from .tiling import ImageTiler, ImageUntiler
from .transform import CropByCoords, PasteByCoords, ReorderBatch
from .coordinates import StringToIntegers, CombineCoords, SplitCoords

NODE_CLASS_MAPPINGS = {
    # From tiling.py
    "ImageTiler": ImageTiler,
    "ImageUntiler": ImageUntiler,
    # From transform.py
    "CropByCoords": CropByCoords,
    "PasteByCoords": PasteByCoords,
    "ReorderBatch": ReorderBatch, 
    # From coordinates.py
    "StringToIntegers": StringToIntegers,
    "CombineCoords": CombineCoords,
    "SplitCoords": SplitCoords,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # From tiling.py
    "ImageTiler": "Tile Image",
    "ImageUntiler": "Untile Image",
    # From transform.py
    "CropByCoords": "Crop Image by Coords",
    "PasteByCoords": "Paste Image by Coords",
    "ReorderBatch": "Reorder Batch",
    # From coordinates.py
    "StringToIntegers": "String to Integers",
    "CombineCoords": "Combine Coords (to COORDS)",
    "SplitCoords": "Split Coords (from COORDS)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']