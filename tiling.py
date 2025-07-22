import torch
import torch.nn.functional as F

class ImageTiler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 2, "min": 1, "max": 256}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 256}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 2}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("tiles", "original_width", "original_height")
    FUNCTION = "execute"
    CATEGORY = "image/tiling"

    def execute(self, image, rows, cols, overlap):
        b, h, w, c = image.shape
        original_h, original_w = h, w

        # Ensure overlap is an even number for symmetrical padding.
        overlap = (overlap // 2) * 2
        padding = overlap // 2

        # 1. Pad canvas to be perfectly divisible by rows/cols.
        h_padded = (h + rows - 1) // rows * rows if rows > 0 else h
        w_padded = (w + cols - 1) // cols * cols if cols > 0 else w
        if h_padded != h or w_padded != w:
            pad_h, pad_w = h_padded - h, w_padded - w
            image = F.pad(image.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h), mode='constant', value=0).permute(0, 2, 3, 1)

        # 2. Calculate tile dimensions including overlap.
        base_tile_h = h_padded // rows
        base_tile_w = w_padded // cols
        
        # Prevent overlap from being larger than the tile itself.
        final_padding_y = min(padding, base_tile_h // 2) if rows > 1 else 0
        final_padding_x = min(padding, base_tile_w // 2) if cols > 1 else 0
        
        output_tile_h = base_tile_h + (final_padding_y * 2)
        output_tile_w = base_tile_w + (final_padding_x * 2)
        
        # 3. Slice image into tiles.
        tiles = []
        with torch.no_grad():
            for img in image:
                for i in range(rows):
                    for j in range(cols):
                        y1_src = max(0, i * base_tile_h - final_padding_y)
                        y2_src = min(h_padded, (i + 1) * base_tile_h + final_padding_y)
                        x1_src = max(0, j * base_tile_w - final_padding_x)
                        x2_src = min(w_padded, (j + 1) * base_tile_w + final_padding_x)
                        
                        source_slice = img[y1_src:y2_src, x1_src:x2_src, :]
                        
                        # Pad edge tiles to match the full tile size, ensuring a consistent batch.
                        # 'replicate' mode is used to avoid black borders on edge tiles.
                        pad_top = 0 if i > 0 else final_padding_y - (i * base_tile_h - y1_src)
                        pad_bottom = 0 if i < rows - 1 else final_padding_y - (y2_src - (i + 1) * base_tile_h)
                        pad_left = 0 if j > 0 else final_padding_x - (j * base_tile_w - x1_src)
                        pad_right = 0 if j < cols - 1 else final_padding_x - (x2_src - (j + 1) * base_tile_w)

                        # Correction for single row/col cases
                        if rows == 1: pad_top = pad_bottom = 0
                        if cols == 1: pad_left = pad_right = 0
                        
                        padded_slice = F.pad(source_slice.permute(2, 0, 1).unsqueeze(0), 
                                             (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
                        
                        tiles.append(padded_slice.squeeze(0).permute(1, 2, 0).unsqueeze(0))

        return (torch.cat(tiles, dim=0), original_w, original_h)


class ImageUntiler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "rows": ("INT", {"default": 2, "min": 1, "max": 256}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 256}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 2}),
            },
            "optional": {
                "original_width": ("INT", {"default": 0, "min": 0, "step": 1, "display": "number"}),
                "original_height": ("INT", {"default": 0, "min": 0, "step": 1, "display": "number"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "image/tiling"

    def execute(self, tiles, rows, cols, overlap, original_width=0, original_height=0):
        group_size = rows * cols
        if tiles.shape[0] % group_size != 0:
            raise ValueError(f"Number of tiles ({tiles.shape[0]}) is not divisible by rows*cols ({group_size}).")
        
        overlap = (overlap // 2) * 2
        padding = overlap // 2

        num_groups = tiles.shape[0] // group_size
        output_images = []

        with torch.no_grad():
            for g in range(num_groups):
                group_tiles = tiles[g * group_size:(g + 1) * group_size]
                _, tile_h_full, tile_w_full, c = group_tiles.shape
                
                base_tile_h = tile_h_full - overlap
                base_tile_w = tile_w_full - overlap
                
                # 1. Prepare canvases for image data and blending weights.
                final_h, final_w = rows * base_tile_h, cols * base_tile_w
                canvas_h, canvas_w = final_h + overlap, final_w + overlap
                image_canvas = torch.zeros((1, canvas_h, canvas_w, c), device=tiles.device, dtype=tiles.dtype)
                weights_canvas = torch.zeros_like(image_canvas)

                # 2. Generate blending masks and assemble tiles.
                for i in range(rows):
                    for j in range(cols):
                        # Create a linear fade mask (window) for blending.
                        fade = torch.linspace(0, 1, padding, device=tiles.device, dtype=tiles.dtype)
                        window_y = torch.ones(tile_h_full, device=tiles.device, dtype=tiles.dtype)
                        if padding > 0:
                            window_y[:padding], window_y[-padding:] = fade, fade.flip(0)
                        
                        window_x = torch.ones(tile_w_full, device=tiles.device, dtype=tiles.dtype)
                        if padding > 0:
                            window_x[:padding], window_x[-padding:] = fade, fade.flip(0)
                        
                        # At the final image edges, the mask should be solid, not faded.
                        if i == 0 and rows > 1: window_y[:padding] = 1.0
                        if i == rows - 1 and rows > 1: window_y[-padding:] = 1.0
                        if j == 0 and cols > 1: window_x[:padding] = 1.0
                        if j == cols - 1 and cols > 1: window_x[-padding:] = 1.0
                        
                        mask = (window_y.unsqueeze(1) * window_x.unsqueeze(0)).unsqueeze(0).unsqueeze(-1)
                        
                        # Add weighted tile to canvas.
                        idx = i * cols + j
                        tile = group_tiles[idx:idx+1]
                        y1, x1 = i * base_tile_h, j * base_tile_w
                        image_canvas[:, y1:y1+tile_h_full, x1:x1+tile_w_full, :] += tile * mask
                        weights_canvas[:, y1:y1+tile_h_full, x1:x1+tile_w_full, :] += mask
                
                # 3. Normalize canvas and crop to final dimensions.
                # Use clamp to prevent division by zero in areas with no tile data.
                final_canvas = image_canvas / weights_canvas.clamp(min=1e-6)
                assembled_image = final_canvas[:, padding : padding + final_h, padding : padding + final_w, :]

                # Crop back to original dimensions if provided.
                if original_width > 0 and original_height > 0:
                    final_image = assembled_image[:, :original_height, :original_width, :]
                else:
                    final_image = assembled_image
                
                output_images.append(final_image)

        return (torch.cat(output_images, dim=0),)