import os
import xarray as xr
import spyndex
import numpy as np
import matplotlib.pyplot as plt

# file paths
green_fp = os.path.join("images", "IMG_0401_2.tif")  # Green band (531 nm)
nir_fp   = os.path.join("images", "IMG_0401_4.tif")  # NIR band (842 nm)

# load as DataArray
green = xr.open_rasterio(green_fp).squeeze("band")
nir   = xr.open_rasterio(nir_fp).squeeze("band")

# compute GNDVI (note params={"N":..., "G":...})
gndvi = spyndex.computeIndex(
    index  = "GNDVI",
    params = {"N": nir, "G": green}
)

# normalize to [0,1]
arrn = np.clip((gndvi.values + 1) / 2, 0, 1)
rgb  = plt.get_cmap("RdYlGn")(arrn)[:,:,:3]
rgb  = (rgb * 255).astype(np.uint8)

# save
plt.imsave("gndvi_rgb.png", rgb)
print("Saved GNDVI → gndvi_rgb.png")