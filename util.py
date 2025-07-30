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
    imageNames = glob.glob(os.path.join(image_dir, 'IMG_0600_*.tif'))
    if len(imageNames) == 0:
        raise ValueError(f"No images found in {image_dir} matching pattern 'IMG_0228_*.tif'")
    cap = capture.Capture.from_filelist(imageNames)
    if cap.dls_present():
        print("DLS irradiance present, using reflectance images.")
        img_type = 'reflectance'
    else:
        print("No DLS irradiance, using radiance images.")
        img_type = 'radiance'
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrices = cap.get_warp_matrices()
    cropped_dimensions, edges = imageutils.find_crop_bounds(cap, warp_matrices)
    im_aligned = imageutils.aligned_capture(cap, warp_matrices, warp_mode, cropped_dimensions, 0, img_type=img_type)
        
    return im_aligned, cap


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


def get_bands_dataarrays(im_aligned, cap):
    bands_da = {}
    for i, band_name in enumerate(cap.band_names()):
        if i < im_aligned.shape[2]:
            da = xr.DataArray(im_aligned[:, :, i])
            bands_da[band_name] = da
    return bands_da


def plot_index_overlay(calculated_index, rgb, out_mask_path, out_overlay_path, title, 
                       threshold=0.7, cmap="jet", remove_outliers=True, pick_range=False, outlier_percentile=50):
    """
    Crea e salva una sovrapposizione (overlay) di una heatmap di un indice spettrale su un'immagine RGB.

    Args:
        calculated_index (np.ndarray): Array 2D dell'indice calcolato (es. NDVI, SAVI).
        rgb (np.ndarray): Immagine RGB di sfondo (H, W, 3).
        threshold (float, optional): Soglia per la maschera di trasparenza (default 0.7).
        cmap (str, optional): Nome della colormap matplotlib da usare per la heatmap (default "jet").
        out_mask_path (str): Percorso file per salvare la maschera dell'indice.
        out_overlay_path (str): Percorso file per salvare l'overlay risultante.
        title (str): Titolo della figura.
        remove_outliers (bool, optional): Se True, rimuove gli outliers prima della normalizzazione (default True).
        outlier_percentile (float, optional): Percentile sopra il quale considerare i valori come outliers (default 95).

    Returns:
        None. Salva le immagini su disco e mostra la figura.
    """
    # Converti a numpy array se necessario e copia per non modificare l'originale
    if hasattr(calculated_index, 'values'):  # pandas DataFrame o xarray DataArray
        processed_index = calculated_index.values.copy()
        original_data = calculated_index.values
    else:  # già un numpy array
        processed_index = np.array(calculated_index).copy()
        original_data = np.array(calculated_index)

    if(pick_range):
        lower, upper = 0.1, 0.3
        min_value = np.nanmin(processed_index)
        mask_keep = (processed_index >= lower) & (processed_index <= upper)
        processed_index = np.where(mask_keep, processed_index, min_value)
        print(f"Applied value filter: keeping [{lower}, {upper}], others set to {min_value:.3f}")

    if(remove_outliers):
         # Trova il minimo dell'indice (escludendo NaN)
        min_value = np.nanmin(processed_index)

        # Calcola il percentile soglia per identificare gli outliers
        outlier_threshold = np.nanpercentile(original_data, outlier_percentile)
        
        # Sostituisci gli outliers con il valore minimo usando np.where
        processed_index = np.where(processed_index > outlier_threshold, min_value, processed_index)
        
        # Conta gli outliers rimossi
        outliers_count = np.sum(original_data > outlier_threshold)
        print(f"Removed {outliers_count} outlier pixels (>{outlier_threshold:.3f}) and set them to {min_value:.3f}")
    
    # Salva la maschera su disco (usando l'array processato)
    plt.imsave("out_masks/" + out_mask_path, processed_index, cmap=cmap)
    
    # Normalizza indice processato a [0, 1]
    vmin = np.nanmin(processed_index)
    vmax = np.nanmax(processed_index)
    arrn = np.clip((processed_index - vmin) / (vmax - vmin), 0, 1)
    
    # Applica soglia per creare una maschera booleana
    mask = (arrn > threshold)
    
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
    plt.savefig("out_overlay/" + out_overlay_path, bbox_inches="tight")
    plt.show()

# Normalizza entrambi gli indici a [0,1]
def normalize_index(idx):
    vmin, vmax = np.nanmin(idx), np.nanmax(idx)
    return np.clip((idx - vmin) / (vmax - vmin), 0, 1)

def compute_index_difference(index1, index2, threshold=0.7, remove_outliers=False, outlier_percentile=95):
    """
    Calcola la differenza tra due indici spettrali normalizzati e applica una soglia.
    
    Args:
        index1 (np.ndarray or xr.DataArray): Primo indice spettrale
        index2 (np.ndarray or xr.DataArray): Secondo indice spettrale  
        threshold (float): Soglia minima per mantenere i valori (default 0.1)
        remove_outliers (bool): Se rimuovere gli outliers prima della normalizzazione (default True)
        outlier_percentile (float): Percentile per identificare outliers (default 95)
    
    Returns:
        np.ndarray: Differenza normalizzata e filtrata per soglia
    """
    # Converti a numpy se necessario
    if hasattr(index1, 'values'):
        idx1 = index1.values.copy()
    else:
        idx1 = np.array(index1).copy()
        
    if hasattr(index2, 'values'):
        idx2 = index2.values.copy()
    else:
        idx2 = np.array(index2).copy()
    
    # Rimuovi outliers se richiesto
    if remove_outliers:
        for idx in [idx1, idx2]:
            outlier_thresh = np.nanpercentile(idx, outlier_percentile)
            min_val = np.nanmin(idx)
            idx = np.where(idx > outlier_thresh, min_val, idx)
    
    # Sostituisci gli outliers con il valore minimo usando np.where
    idx2 = np.where(idx2 < threshold, np.nanmin(idx2), idx2)
    
    # Calcola differenza
    difference = idx1 - idx2

    return difference