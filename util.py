import os, glob
import micasense.capture as capture
import cv2
import numpy as np
import matplotlib.pyplot as plt
import micasense.imageutils as imageutils
import micasense.plotutils as plotutils
import xarray as xr        # libreria per array multi-dimensionali
import rioxarray           # estensione geospaziale per xarray

def get_aligned_images(image_dir):
    """
    Allinea le immagini multispettrali in una directory e restituisce uno stack numpy delle immagini allineate.

    Args:
        image_dir (str): Percorso della cartella contenente le immagini multispettrali.

    Returns:
        np.ndarray: Array numpy 3D (H, W, Bands) delle immagini allineate.
    """
    imageNames = glob.glob(os.path.join(image_dir, 'IMG_0401_*.tif'))
    if len(imageNames) == 0:
        raise ValueError(f"No images found in {image_dir} matching pattern 'IMG_0401_*.tif'")
    cap = capture.Capture.from_filelist(imageNames)
    if cap.dls_present():
        img_type = 'reflectance'
    else:
        img_type = 'radiance'
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrices = cap.get_warp_matrices()
    cropped_dimensions, edges = imageutils.find_crop_bounds(cap, warp_matrices)
    im_aligned = imageutils.aligned_capture(cap, warp_matrices, warp_mode, cropped_dimensions, 0, img_type=img_type)
    return im_aligned


def get_rgb_from_aligned(im_aligned, img_type, rgb_band_indices=[2,1,0]):
    """
    Restituisce un'immagine RGB normalizzata a partire da uno stack di immagini allineate.

    Args:
        im_aligned (np.ndarray): Array numpy 3D (H, W, Bands) delle immagini allineate.
        img_type (str): Tipo di immagine ('reflectance' o 'radiance').
        rgb_band_indices (list, optional): Indici delle bande da usare per l'RGB (default [2,1,0]).

    Returns:
        np.ndarray: Array numpy 3D (H, W, 3) dell'immagine RGB normalizzata.
    """
    # Create an empty normalized stack for viewing
    im_display = np.zeros((im_aligned.shape[0], im_aligned.shape[1], im_aligned.shape[2]), dtype=np.float32)
    im_min = np.percentile(im_aligned[:,:,0:2].flatten(),  0.1)
    im_max = np.percentile(im_aligned[:,:,0:2].flatten(), 99.9)
    for i in range(0, im_aligned.shape[2]):
        if img_type == 'reflectance':
            im_display[:,:,i] = imageutils.normalize(im_aligned[:,:,i], im_min, im_max)
        elif img_type == 'radiance':
            im_display[:,:,i] = imageutils.normalize(im_aligned[:,:,i])
    rgb = im_display[:,:,rgb_band_indices]
    return rgb


def get_bands_dataarrays(im_aligned, crs="EPSG:32633"):
    """
    Crea un dizionario di DataArray xarray per ogni banda presente nell'immagine allineata.

    Args:
        im_aligned (np.ndarray): Array numpy 3D (H, W, Bands) delle immagini allineate.
        crs (str, optional): Codice EPSG del sistema di riferimento spaziale. Default: "EPSG:32633".

    Returns:
        dict: Dizionario {nome_banda: xr.DataArray} con CRS associato per ogni banda trovata.
    """

    band_indices = {
        "B": 0,
        "G": 1,
        "R": 2,
        "N": 3,
        "RE": 4,
        "Y": 5,
        "R2": 6
        # Aggiungi qui altre bande se necessario
    }
    bands_da = {}
    for band_name, idx in band_indices.items():
        if idx < im_aligned.shape[2]:
            da = xr.DataArray(
                im_aligned[:, :, idx],
                dims=["y", "x"],
                name=band_name
            )
            da.rio.write_crs(crs, inplace=True)
            bands_da[band_name] = da
    return bands_da


def plot_index_overlay(calculated_index, rgb, threshold=0.7, cmap="jet", out_mask_path="SAVI.png", out_overlay_path="overlay_savi.png", title="SAVI Overlay su RGB"):
    """
    Crea e salva una sovrapposizione (overlay) di una heatmap di un indice spettrale su un'immagine RGB.

    Args:
        calculated_index (np.ndarray): Array 2D dell'indice calcolato (es. NDVI, SAVI).
        rgb (np.ndarray): Immagine RGB di sfondo (H, W, 3).
        threshold (float, optional): Soglia per la maschera di trasparenza (default 0.7).
        cmap (str, optional): Nome della colormap matplotlib da usare per la heatmap (default "magma").
        out_mask_path (str, optional): Percorso file per salvare la maschera dell'indice (default "SAVI.png").
        out_overlay_path (str, optional): Percorso file per salvare l'overlay risultante (default "overlay_savi.png").
        title (str, optional): Titolo della figura (default "SAVI Overlay su RGB").

    Returns:
        None. Salva le immagini su disco e mostra la figura.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    # Salva la maschera su disco prima di applicarla alla heatmap
    plt.imsave(out_mask_path, calculated_index, cmap=cmap)
    # Normalizza indice da [-1, 1] a [0, 1]
    arrn = np.clip((calculated_index + 1)/2, 0, 1)
    # Applica soglia per creare una maschera booleana
    mask = arrn > threshold
    # Crea una mappa RGBA (4 canali) con la colormap scelta
    heatmap = plt.get_cmap(cmap)(arrn)
    # Applica trasparenza: sotto soglia invisibile, sopra soglia semitrasparente
    heatmap[~mask, 3] = 0.0  # alpha = 0
    # Overlay della heatmap su immagine RGB
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.imshow(heatmap)
    plt.axis("off")
    plt.title(title)
    plt.savefig(out_overlay_path, bbox_inches="tight")
    plt.show()
