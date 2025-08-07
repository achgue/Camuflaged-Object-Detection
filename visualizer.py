import numpy as np
import rasterio
import matplotlib.pyplot as plt

# === Load the image ===
tif_path = "im/IMG_0422_4.tif"

with rasterio.open(tif_path) as src:
    band = src.read(1)  # First band
    profile = src.profile

# === Create base figure ===
fig, ax = plt.subplots()
gray_img = ax.imshow(band, cmap='gray')
plt.title("Hover to see pixel value — Red: ±2000 match")

# === Text overlay ===
text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='white',
               bbox=dict(facecolor='black', alpha=0.6), fontsize=10)

# === Red overlay for highlighting ===
highlight = np.zeros((*band.shape, 4), dtype=np.float32)  # RGBA
highlight_img = ax.imshow(highlight, interpolation='none')

# === Update on mouse move ===
def on_mouse_move(event):
    if not event.inaxes or event.xdata is None or event.ydata is None:
        return

    col, row = int(event.xdata), int(event.ydata)
    if 0 <= row < band.shape[0] and 0 <= col < band.shape[1]:
        val = band[row, col]
        msg = f"X: {col}, Y: {row} → Value: {val}"
        text.set_text(msg)

        # Highlight pixels within ±2000 of the hovered value
        diff = np.abs(band - val)
        mask = diff <= 2000

        # Reset highlight layer
        highlight[:, :, :] = 0
        highlight[mask] = [1.0, 0.0, 0.0, 0.5]  # Red with alpha

        highlight_img.set_data(highlight)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
plt.show()
