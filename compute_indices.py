import spyndex             # libreria per il calcolo degli indici spettrali
import util
import numpy as np

aligned_capture, cap = util.get_aligned_images("./im")  # Percorso delle immagini
rgb = util.get_rgb_from_aligned(aligned_capture, img_type="reflectance", rgb_band_indices=[2, 1, 0])
bands_da = util.get_bands_dataarrays(aligned_capture, cap)  # Indici delle bande da usare

def compute_ferric_index():
    red = bands_da["Red"]
    green = bands_da["Green"]
    
    # Calcolo dell'indice ferrico come rapporto Red/Green
    ferric_index = red / green
    
    # Visualizza e salva l'indice ferrico
    try:
        mask_path = "ferric_index.png"
        overlay_path = "overlay_ferric_index.png"
        titolo = "Ferric Index (Red/Green) overlay su RGB"
        
        # Visualizza e salva overlay
        util.plot_index_overlay(
            ferric_index,
            rgb,
            out_mask_path=mask_path,
            out_overlay_path=overlay_path,
            title=titolo,
            threshold=0.1,
            cmap="jet"
        )
        print(f"[✓] Ferric Index processato correttamente.")
        
    except Exception as e:
        print(f"[!] Errore con Ferric Index: {e}")
    
    return ferric_index
	
def compute_wv_II():
    blue = bands_da["Blue"]
    green = bands_da["Green"]
    
    # Calcolo dell'indice ferrico come rapporto Red/Green
    wv_ratio = blue / green * 1000
    # Visualizza e salva l'indice ferrico
    try:
        mask_path = "iwv_index.png"
        overlay_path = "overlay_wv_index.png"
        titolo = "WV (BLue/Green*1000) overlay su RGB"
        
        # Visualizza e salva overlay
        util.plot_index_overlay(
            wv_ratio,
            rgb,
            threshold=0.1,
            cmap="jet",
            out_mask_path=mask_path,
            out_overlay_path=overlay_path,
            title=titolo
        )
        print(f"[✓] WV Index processato correttamente.")
        
    except Exception as e:
        print(f"[!] Errore con WV Index: {e}")
    
    return wv_ratio	

def compute_iron_oxide():
    red = bands_da["Red"]
    red_edge = bands_da["Red edge-705"]
    blue = bands_da["Blue"]
    blue444 = bands_da["Blue-444"]
    green = bands_da["Green"]
    nir = bands_da["NIR"]
    
    # Calcolo dell'indice ferrico come rapporto Red/Green
    iron_oxide_ratio = (nir - blue)/ nir + blue
    # Visualizza e salva l'indice ferrico
    try:
        mask_path = "iron_oxide_index.png"
        overlay_path = "overlay_iron_oxide_index.png"
        titolo = "Iron oxide index (Red/Blue) overlay su RGB"

        # Visualizza e salva overlay
        util.plot_index_overlay(
            iron_oxide_ratio,
            rgb,
            threshold=0.1,
            cmap="jet",
            out_mask_path=mask_path,
            out_overlay_path=overlay_path,
            title=titolo
        )
        print(f"[✓] Iron Oxide Index processato correttamente.")
        
    except Exception as e:
        print(f"[!] Errore con Iron Oxide Index: {e}")

    return iron_oxide_ratio	

# Calcolo di tre indici vegetazionali con Spyndex
indices = spyndex.computeIndex(
    index=["TSAVI", "SAVI", "MCARI", "NDVI", "GEMI", "BITM", "BIXS", "RI4XS", "NHFD", "PISI", "VgNIRBI"],  # indici da calcolare
    params={
        "N": bands_da["NIR"],         # banda NIR
        "R": bands_da["Red"],         # banda rossa
        "RE1": bands_da["Red Edge"],  # banda rossa
        "G": bands_da["Green"],       # banda verde
        "B": bands_da["Blue"],        # banda blu
        "A": bands_da["Blue-444"],
        "L": 0.5,                     # parametro per il SAVI
        "sla": 0.5,
        "slb": 0.0,                   # parametri per il TSAVI
    }
)

# Compute NHFD - SAVI difference
def compute_nhfd_savi_difference():
    # Calculate SAVI and NHFD indices
    savi_nhfd_indices = spyndex.computeIndex(
        index=["NHFD", "SAVI", "PISI"],
        params={
            "N": bands_da["NIR"],         # banda NIR
            "R": bands_da["Red"],         # banda rossa
            "RE1": bands_da["Red Edge"],  # banda rossa
            "G": bands_da["Green"],       # banda verde
            "B": bands_da["Blue"],        # banda blu
            "A": bands_da["Blue-444"],
            "L": 0.5,                     # parametro per il SAVI
            "sla": 0.5,
            "slb": 0.0,                   # parametri per il TSAVI
        }
    )
    
    # Extract individual indices
    nhfd_index = savi_nhfd_indices.sel(index="NHFD")
    savi_index = savi_nhfd_indices.sel(index="SAVI")
    pisi_index = savi_nhfd_indices.sel(index="PISI")

    # Compute difference: NHFD - SAVI
    nhfd_index = util.normalize_index(nhfd_index)
    savi_index = util.normalize_index(savi_index)
    difference = util.compute_index_difference(nhfd_index, savi_index)
    # pisi_index = util.normalize_index(pisi_index)
    # difference = util.normalize_index(difference)
    #difference = util.compute_index_difference(difference, pisi_index)
    
    # Plot the difference
    try:
        mask_path = "NHFD_minus_SAVI.png"
        overlay_path = "overlay_NHFD_minus_SAVI.png"
        title = "NHFD - SAVI Difference overlay su RGB"
        
        util.plot_index_overlay(
            difference,
            rgb,
            out_mask_path=mask_path,
            out_overlay_path=overlay_path,
            title=title,
            threshold=0.6,
            cmap="jet"
        )
        print(f"[✓] NHFD - SAVI difference processato correttamente.")
        
    except Exception as e:
        print(f"[!] Errore con NHFD - SAVI difference: {e}")
    
    return difference

# Call the function
compute_nhfd_savi_difference()

compute_ferric_index()  # Calcola e visualizza l'indice ferrico

compute_wv_II()  # Calcola e visualizza l'indice degli ossidi di ferro

compute_iron_oxide()  # Calcola e visualizza l'indice degli ossidi di ferro

indici_da_visualizzare = ["TSAVI", "SAVI", "MCARI", "NDVI", "GEMI", "BITM", "BIXS", "RI4XS", "NHFD", "PISI", "VgNIRBI"]

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
            threshold=0.1,
            cmap="jet",
            out_mask_path=mask_path,
            out_overlay_path=overlay_path,
            title=titolo
        )
        print(f"[✓] {indice_nome} processato correttamente.")
    
    except Exception as e:
        print(f"[!] Errore con indice {indice_nome}: {e}")

