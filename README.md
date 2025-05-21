# SAM2 Point Segmentation Demo

This project provides a Tkinter-based GUI for interactive point-based segmentation using the SAM2 model.

## Features

- Load images and interactively select points for segmentation.
- Run SAM2 segmentation and visualize masks.
- Manage mask history (rename, delete, save).
- Pan (WASD) and zoom image view.
- Select multiple masks to combine into custom masks.

## Requirements

- Python 3.8+
- [Pillow](https://python-pillow.org/)
- [numpy](https://numpy.org/)
- [tkinter](https://wiki.python.org/moin/TkInter)
- [samv2](https://github.com/SauravMaheshkar/samv2) (`pip install git+https://github.com/SauravMaheshkar/samv2.git`)

## Setup

1. Clone this repository.
2. Install dependencies:
   ```sh
   pip install pillow numpy
   pip install git+https://github.com/SauravMaheshkar/samv2.git
   ```
3. Download the SAM2 checkpoints:
   ```sh
   pwsh ./download_checkpoints.ps1
   ```

## Usage

Run the GUI:
```sh
python sam2_select.py
```

## Checkpoints

The script `download_checkpoints.ps1` downloads the required model checkpoints:
- `sam2_hiera_tiny.pt`
- `sam2_hiera_small.pt`
- `sam2_hiera_base_plus.pt`
- `sam2_hiera_large.pt`