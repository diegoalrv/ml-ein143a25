"""Descarga de una vez todos los modelos del Lab 14 (≈4 GB), para no esperar en clase.

    python precargar_modelos.py           # los 5 modelos del notebook
    python precargar_modelos.py --profe   # además sd-turbo para la demo del profe (≈2,5 GB más)

Quedan en la caché de Hugging Face (~/.cache/huggingface) y el notebook ya no los vuelve a bajar.
"""
import sys

import llm_utils

llm_utils.precargar(*llm_utils.MODELOS)
if llm_utils.SIN_RED:
    sys.exit("No se pudieron descargar los modelos: revisa la conexión y vuelve a intentar.")
if "--profe" in sys.argv:
    import texto_a_imagen_profe
    texto_a_imagen_profe.cargar()
print("✔ Modelos del Lab 14 listos")
