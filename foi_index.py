#!/usr/bin/env python3
"""
Compute a Ferric Oxide Index (FOI) to highlight metal‐rich (iron oxide) areas:
FOI = (Red₆₅₀ − Green₅₃₁) / (Red₆₅₀ + Green₅₃₁)
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# 1. Filepaths for Red‐650 nm and Green‐531 nm bands
red_fp   = os.path.join("images", "IMG_0401_8.tif")  # Red‐650
green_fp = os.path.join("images", "IMG_0401_7.tif")  # Green‐531

# 2. Load each as 2D DataArray (drop band dim)
red   = xr.open_rasterio(red_fp).squeeze("band").astype(np.float32)
green = xr.open_rasterio(green_fp).squeeze("band").astype(np.float32)

# 3. Compute FOI with small epsilon to avoid division by zero
eps = 1e-6
foi = (red - green) / (red + green + eps)

# 4. Normalize to [0,1] for visualization
vmin, vmax = -1, 1
foi_norm = np.clip((foi.values - vmin) / (vmax - vmin), 0, 1)

# 5. Apply colormap (metal areas → bright yellow/red)
cmap = plt.get_cmap("inferno")
rgb = cmap(foi_norm)[:, :, :3]
rgb = (rgb * 255).astype(np.uint8)

# 6. Save the RGB result
out_png = "foi_metal_highlight.png"
plt.imsave(out_png, rgb)
print(f"Saved metal‐highlight FOI image → {out_png}")

