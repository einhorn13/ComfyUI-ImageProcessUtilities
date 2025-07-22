import torch
import torch.nn.functional as F
from .coordinates import _parse_ints_from_string, CombineCoords

class CropByCoords:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "width": ("INT", {"default": 64, "min": 1}),
                "height": ("INT", {"default": 64, "min": 1}),
            },
            "optional": {
                "coords": ("COORDS",),
                "x_coords_str": ("STRING", {"multiline": False}),
                "y_coords_str": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"
    CATEGORY = "image/transform"

    def crop(self, images, width, height, coords=None, x_coords_str="", y_coords_str=""):
        if coords is None and not x_coords_str and not y_coords_str:
            return (images,) # No operation if no coordinates are provided

        if coords is None:
            xs = _parse_ints_from_string(x_coords_str) or [0]
            ys = _parse_ints_from_string(y_coords_str) or [0]
            coords, _ = CombineCoords().combine(xs, ys)

        b, H, W, C = images.shape
        
        # Fast path: If all crops are identical, perform a single batch operation.
        if len(coords) == 1:
            x, y = coords[0]
            x1_src, y1_src = max(0, x), max(0, y)
            x2_src, y2_src = min(W, x + width), min(H, y + height)
            
            crop_slice = images[:, y1_src:y2_src, x1_src:x2_src, :]
            slice_h, slice_w = crop_slice.shape[1:3]
            
            # Create a padded canvas to handle out-of-bounds crops.
            padded_crop = torch.zeros((b, height, width, C), device=images.device, dtype=images.dtype)
            paste_y_start, paste_x_start = max(0, -y), max(0, -x)
            padded_crop[:, paste_y_start:paste_y_start+slice_h, paste_x_start:paste_x_start+slice_w, :] = crop_slice
            return (padded_crop,)

        # Iterative path: Process each image with its corresponding coordinate.
        crops = []
        for i in range(b):
            # Use the last coordinate if the list is shorter than the batch size.
            coord = coords[i if i < len(coords) else -1]
            x, y = coord[0], coord[1]
            
            x1_src, y1_src = max(0, x), max(0, y)
            x2_src, y2_src = min(W, x + width), min(H, y + height)
            crop_slice = images[i:i+1, y1_src:y2_src, x1_src:x2_src, :]
            slice_h, slice_w = crop_slice.shape[1:3]

            padded_crop = torch.zeros((1, height, width, C), device=images.device, dtype=images.dtype)
            paste_y_start, paste_x_start = max(0, -y), max(0, -x)
            padded_crop[:, paste_y_start:paste_y_start+slice_h, paste_x_start:paste_x_start+slice_w, :] = crop_slice
            crops.append(padded_crop)
        
        return (torch.cat(crops, dim=0),)


class PasteByCoords:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"base": ("IMAGE",), "inserts": ("IMAGE",)},
            "optional": {
                "coords": ("COORDS",),
                "mask": ("MASK",),
                "x_coords_str": ("STRING", {"multiline": False}),
                "y_coords_str": ("STRING", {"multiline": False}),
                "feathering": ("INT", {"default": 8, "min": 0, "max": 1024}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste"
    CATEGORY = "image/transform"

    def paste(self, base, inserts, coords=None, mask=None, x_coords_str="", y_coords_str="", feathering=0):
        if coords is None and not x_coords_str and not y_coords_str:
            coords = [[0,0]] # Default to top-left corner if no coords provided.

        if coords is None:
            xs = _parse_ints_from_string(x_coords_str) or [0]
            ys = _parse_ints_from_string(y_coords_str) or [0]
            coords, _ = CombineCoords().combine(xs, ys)

        b, H, W, C = base.shape
        result = base.clone()

        num_inserts, num_coords = inserts.shape[0], len(coords)

        for i in range(b):
            # Cycle through coords and inserts if their lists are shorter than the base batch.
            coord = coords[i % num_coords]
            x, y = coord[0], coord[1]
            
            ins = inserts[i % num_inserts].unsqueeze(0)
            _, h, w, _ = ins.shape
            
            # --- Blending Mask Generation ---
            if mask is not None:
                m = mask[i % mask.shape[0]]
                # Resize mask to match insert dimensions if necessary.
                if m.shape[0] != h or m.shape[1] != w:
                    m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
                blending_mask = m.unsqueeze(0).unsqueeze(-1)
            else:
                blending_mask = torch.ones((1, h, w, 1), device=ins.device, dtype=ins.dtype)
                # Apply feathering if specified, creating a soft edge.
                if feathering > 0:
                    f = min(feathering, h // 2, w // 2)
                    if f > 0:
                        fade = torch.linspace(0, 1, f, device=ins.device, dtype=ins.dtype)
                        fade_x = fade.view(1, 1, -1, 1)
                        fade_y = fade.view(1, -1, 1, 1)
                        blending_mask[:, :f, :, :] *= fade_y
                        blending_mask[:, -f:, :, :] *= fade_y.flip(1)
                        blending_mask[:, :, :f, :] *= fade_x
                        blending_mask[:, :, -f:, :] *= fade_x.flip(2)
            
            blending_mask = blending_mask.repeat(1, 1, 1, C)

            # --- Paste Operation ---
            y1_dst, x1_dst = max(0, y), max(0, x)
            y2_dst, x2_dst = min(H, y + h), min(W, x + w)
            y1_src, x1_src = max(0, -y), max(0, -x)
            y2_src, x2_src = y1_src + (y2_dst - y1_dst), x1_src + (x2_dst - x1_dst)

            if y1_dst >= y2_dst or x1_dst >= x2_dst: continue

            # Alpha composite the insert over the base image.
            base_crop = result[:, y1_dst:y2_dst, x1_dst:x2_dst, :]
            mask_crop = blending_mask[:, y1_src:y2_src, x1_src:x2_src, :]
            ins_crop = (ins * blending_mask)[:, y1_src:y2_src, x1_src:x2_src, :]

            result[:, y1_dst:y2_dst, x1_dst:x2_dst, :] = base_crop * (1 - mask_crop) + ins_crop
            
        return (result,)

class ReorderBatch:
    MODES = ["row_to_column", "reverse", "ping-pong"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (cls.MODES, ),
            },
            "optional": {
                "columns": ("INT", {"default": 2, "min": 1, "max": 128}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "new_batch_size")
    FUNCTION = "reorder"
    CATEGORY = "image/transform"

    def reorder(self, images: torch.Tensor, mode: str, columns: int = 2):
        batch_size = images.shape[0]
        if batch_size == 0:
            return (images, 0)
        
        output_images = images
        
        if mode == "reverse":
            output_images = torch.flip(images, dims=[0])

        elif mode == "ping-pong":
            # Creates a seamless loop by appending the reversed sequence (excluding start/end frames).
            if batch_size > 2:
                return_trip = images[1:-1].flip(dims=[0])
                output_images = torch.cat((images, return_trip), dim=0)
            elif batch_size == 2: # For 2 frames, just add the first one at the end.
                output_images = torch.cat((images, images[0:1]), dim=0)

        elif mode == "row_to_column":
            if columns <= 0: return (images, batch_size)
            
            # Transposes the batch order. E.g., for 2 cols: [0,1,2,3,4,5] -> [0,2,4,1,3,5]
            rows = (batch_size + columns - 1) // columns
            indices = [row * columns + col for col in range(columns) for row in range(rows) if row * columns + col < batch_size]
            output_images = images[torch.tensor(indices)]

        return (output_images, output_images.shape[0])