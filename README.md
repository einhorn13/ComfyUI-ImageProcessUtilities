# ComfyUI-ImageProcessUtilities

A collection of **custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)** designed to enhance image processing workflows — especially for high-resolution rendering, complex inpainting, tiling, and batch manipulation tasks.

---

## 🔑 Key Features

- **🧩 Seamless Image Tiling & Untiling**  
  Process huge images by breaking them into overlapping tiles and perfectly reassembling them. This is **non-destructive**, using padding instead of cropping to preserve all image data.

- **📐 Coordinate-Based Transforms**  
  Precisely crop and paste images using a flexible coordinate system. Ideal for targeted inpainting or complex composites.

- **🔁 Modular Coordinate System**  
  Convert strings to coordinates and combine/split coordinate lists, enabling complex and dynamic workflows.

- **📦 Batch Manipulation Utilities**  
  Reorder or modify image batches — useful for animations, video workflows, or layout transforms.

---

## 🧱 Nodes

### 🟦 Tiling

- **Tile Image (`ImageTiler`)**  
  Splits an image into a batch of overlapping tiles. Uses non-destructive padding to handle images of any size.

- **Untile Image (`ImageUntiler`)**  
  Reassembles a batch of tiles into a single, seamless image — correctly cropping it to original dimensions.

---

### 🔲 Transform

- **Crop Image by Coords (`CropByCoords`)**  
  Crops one or more regions from an image or batch at specified coordinates.

- **Paste Image by Coords (`PasteByCoords`)**  
  Pastes insert images onto a base image at specified coordinates. Supports optional **edge feathering** for smooth blending.

- **Reorder Batch (`ReorderBatch`)**  
  Changes the order of images in a batch. Modes include:
  - `reverse`
  - `ping-pong`
  - `row_to_column`

---

### 🧮 Coordinates

- **String to Integers (`StringToIntegers`)**  
  Converts a comma-separated string like `"128, 256, 512"` into a list of integers.

- **Combine Coords (`CombineCoords`)**  
  Combines two lists of X and Y values into a `COORDS` structure.

- **Split Coords (`SplitCoords`)**  
  Splits a `COORDS` input into two lists: X and Y.

---

## ⚙️ Installation

1. Navigate to your ComfyUI `custom_nodes/` directory:

   ```bash
   cd ComfyUI/custom_nodes/

2. Clone this repository: git clone <repository_url>

3. Restart ComfyUI.

Example usecase: High-Resolution Upscaling Workflow

This allows you to upscale images that would otherwise exceed your VRAM limits.
