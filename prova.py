import spyndex             # libreria per il calcolo degli indici spettrali
import util

aligned_capture = util.get_aligned_images("./im/")  # Percorso delle immagini
rgb = util.get_rgb_from_aligned(aligned_capture, img_type="reflectance", rgb_band_indices=[2, 1, 0])
bands_da = util.get_bands_dataarrays(aligned_capture)  # Indici delle bande da usare

custom_idx = {
    "short_name": "FII",
    "long_name": "Ferric Iron Index",
    "formula": "R / G",  # example
    "bands": ("R", "G"),  # use Spyndex band names
    "application_domain": "metal",
    "date_of_addition": "2025-07-14",
}


spyndex.indices.FII = spyndex.axioms.SpectralIndex(custom_idx)

# Calcolo di tre indici vegetazionali con Spyndex
indices = spyndex.computeIndex(
    index=["TSAVI", "SAVI", "MCARI", "GEMI", "SR", "NDVI", "FII"],  # indici da calcolare
    params={
        "N": bands_da["N"],           # banda NIR
        "R": bands_da["R"],           # banda rossa
        "RE1": bands_da["RE1"],           # banda rossa
        "G": bands_da["G"],          # banda verde
        "L": 0.5,                      # parametro per il SAVI
        "sla": 0.5,
        "slb": 0.0,                   # parametri per il TSAVI
    }
)

# Estrae l'indice NDVI dal risultato
#print(indices)
indici_da_visualizzare = ["TSAVI", "SAVI", "MCARI", "GEMI", "SR", "NDVI", "FII"]

for indice_nome in indici_da_visualizzare:
    try:
        # Seleziona l'indice calcolato
        calculated_index = indices.sel(index=indice_nome)

        # Costruisci i nomi dei file di output
        mask_path = f"{indice_nome}.png"
        overlay_path = f"overlay_{indice_nome}.png"

        # Titolo dinamico
        titolo = f"{indice_nome} overlay su RGB"

        # Visualizza e salva overlay
        util.plot_index_overlay(
            calculated_index,
            rgb,
            threshold=0.9,
            cmap="jet",
            out_mask_path=mask_path,
            out_overlay_path=overlay_path,
            title=titolo
        )
        print(f"[✓] {indice_nome} processato correttamente.")
    
    except Exception as e:
        print(f"[!] Errore con indice {indice_nome}: {e}")
