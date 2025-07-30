import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# === Load the PNG image ===
png_path = "out_overlay/overlay_NHFD.png"  # Replace with your PNG file path

# Load image as RGB
img_rgb = np.array(Image.open(png_path).convert('RGB'))
height, width, channels = img_rgb.shape

# Convert to grayscale for display
img_gray = np.dot(img_rgb[...,:3], [0.2989, 0.5870, 0.1140])

# === Create base figure ===
fig, ax = plt.subplots(figsize=(10, 8))
gray_display = ax.imshow(img_gray, cmap='gray')
plt.title("Hover to see RGB values — Colored areas: similar colors (±20 per channel)")

# === Text overlay ===
text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='white',
               bbox=dict(facecolor='black', alpha=0.7), fontsize=10)

# === Color overlay for highlighting ===
highlight = np.zeros((height, width, 4), dtype=np.float32)  # RGBA
highlight_img = ax.imshow(highlight, interpolation='none')

# === Update on mouse move ===
def on_mouse_move(event):
    if not event.inaxes or event.xdata is None or event.ydata is None:
        return

    col, row = int(event.xdata), int(event.ydata)
    if 0 <= row < height and 0 <= col < width:
        # Get RGB values at cursor position
        r_val, g_val, b_val = img_rgb[row, col]
        
        # Display cursor info
        msg = f"X: {col}, Y: {row} → RGB: ({r_val}, {g_val}, {b_val})"
        text.set_text(msg)

        # Find pixels with similar colors (±20 in each channel)
        r_diff = np.abs(img_rgb[:, :, 0] - r_val)
        g_diff = np.abs(img_rgb[:, :, 1] - g_val)
        b_diff = np.abs(img_rgb[:, :, 2] - b_val)
        
        # Mask for pixels within ±20 range in ALL channels
        mask = (r_diff <= 20) & (g_diff <= 20) & (b_diff <= 20)

        # Reset highlight layer
        highlight[:, :, :] = 0
        
        # Highlight similar pixels with the original color (normalized to 0-1)
        highlight[mask] = [r_val/255.0, g_val/255.0, b_val/255.0, 0.6]  # Original color with alpha

        highlight_img.set_data(highlight)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
plt.tight_layout()
plt.show()