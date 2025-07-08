import spyndex             # libreria per il calcolo degli indici spettrali
import util

aligned_capture = util.get_aligned_images("./im/")  # Percorso delle immagini
rgb = util.get_rgb_from_aligned(aligned_capture, img_type="reflectance", rgb_band_indices=[2, 1, 0])
bands_da = util.get_bands_dataarrays(aligned_capture)  # Indici delle bande da usare
# Calcolo di tre indici vegetazionali con Spyndex
indices = spyndex.computeIndex(
    index=["NDVI", "GNDVI", "SAVI"],  # indici da calcolare
    params={
        "N": bands_da["N"],           # banda NIR
        "R": bands_da["R"],           # banda rossa
        "G": bands_da["G"],           # banda verde
        "L": 0.5                      # parametro per il SAVI
    }
)

# Estrae l'indice NDVI dal risultato
calculated_index = indices.sel(index="NDVI")

util.plot_index_overlay(
    calculated_index,
    rgb,
    threshold=0.7,
    cmap="jet",
    out_mask_path="NDVI.png",
    out_overlay_path="overlay_NDVI.png",
    title="NDVI Overlay su RGB"
)