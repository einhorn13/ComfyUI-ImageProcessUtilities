# ComfyUI-ImageProcessUtilities
This is a collection of custom nodes for ComfyUI designed for image processing workflows. It is useful for high-resolution rendering, complex inpainting, and composition tasks that require breaking down and reassembling images.

Key Features
Seamless Image Tiling & Untiling: Process huge images by breaking them into overlapping tiles and perfectly reassembling them. The process is non-destructive, using padding instead of cropping to preserve all image data.

Coordinate-Based Transforms: Precisely crop and paste images using a flexible coordinate system. Ideal for targeted inpainting or complex composites.

Modular Coordinate System: Convert strings to coordinates and combine/split coordinate lists, enabling complex and dynamic workflows.

Batch Manipulation Utilities: Easily reorder or modify image batches for tasks like creating animations and video processing

  **#Nodes**
**Tiling**
Tile Image (ImageTiler): Splits an image into a batch of overlapping tiles. Uses non-destructive padding to handle images of any size.

Untile Image (ImageUntiler): Reassembles a batch of tiles into a single, seamless final image, correctly cropping it to the original dimensions.

**Transform**
Crop Image by Coords (CropByCoords): Crops one or more regions from an image or image batch at specified coordinates.

Paste Image by Coords (PasteByCoords): Pastes insert images onto a base image at specified coordinates, with optional edge feathering for smooth blending.

Reorder Batch (ReorderBatch): Manipulates the order of images in a batch with modes like reverse, ping-pong, and row_to_column.

**Coordinates**
String to Integers (StringToIntegers): Converts a comma-separated string (e.g., "128, 256, 512") into a list of integers.

Combine Coords (CombineCoords): Combines separate lists of X and Y coordinates into a single COORDS output.

Split Coords (SplitCoords): Splits a COORDS input back into separate X and Y integer lists.

**Installation**
Navigate to your ComfyUI/custom_nodes/ directory.

Clone this repository: git clone <repository_url>

Restart ComfyUI.

Example: High-Resolution Upscaling Workflow

This allows you to upscale images that would otherwise exceed your VRAM limits.
