"""Demo del profe (Lab 14, P4.3 · diapo 23): de texto a imagen, de ruido a dibujo, con sd-turbo.

Corre en el Mac del profe (GPU del Mac, «mps»; si no hay, en la CPU, mucho más lento).

    python texto_a_imagen_profe.py            # modo interactivo: los estudiantes dictan frases
    python texto_a_imagen_profe.py --pasos    # además muestra los pasos de ruido a dibujo
    python texto_a_imagen_profe.py --galeria  # graba galeria_profe.png y pasos_profe.png (notebook y Release)

Las imágenes quedan en galeria_profe/ con la frase como nombre. Ctrl+C para salir.
"""
import argparse
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import torch

CARPETA = Path(__file__).resolve().parent
SALIDA = CARPETA / "galeria_profe"
REPO = "stabilityai/sd-turbo"
PASOS = 4  # sd-turbo dibuja en 1 paso; con 4 se alcanza a ver la secuencia de ruido a dibujo
FRASES_GALERIA = [
    "un perro surfeando en Cocholgüe al atardecer, estilo acuarela",
    "un robot tomando mote con huesillo, ilustración",
    "un gato astronauta flotando sobre una ciudad de noche, ilustración",
    "una sopaipilla gigante en la playa, fotografía",
    "un pingüino leyendo un libro en una biblioteca, estilo caricatura",
    "un puente hecho de chocolate sobre un río, estilo acuarela",
]
SEMILLAS_GALERIA = [3, 1, 2, 3, 4, 5]  # elegidas mirando 8 semillas del perro: con la 3 se ve sobre la tabla
AZUL, MARINO = "#2e69b2", "#1b2740"


def cargar():
    from diffusers import AutoPipelineForText2Image
    disp = "mps" if torch.backends.mps.is_available() else "cpu"
    tipo = torch.float16 if disp == "mps" else torch.float32
    print(f"Cargando {REPO} en {disp} (la primera vez se descarga ≈2,5 GB)…", flush=True)
    try:  # primero desde la caché, sin internet: una consulta colgada a Hugging Face congela la demo
        pipe = AutoPipelineForText2Image.from_pretrained(REPO, dtype=tipo, variant="fp16", local_files_only=True)
    except Exception:
        pipe = AutoPipelineForText2Image.from_pretrained(REPO, dtype=tipo, variant="fp16")
    pipe = pipe.to(disp)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def dibujar(pipe, frase, semilla=0, con_pasos=False):
    """Devuelve (imagen final, lista de (rótulo, imagen) de ruido a dibujo) y los segundos que tomó."""
    gen = torch.Generator("cpu").manual_seed(semilla)
    ruido = torch.randn((1, 4, 64, 64), generator=gen, dtype=torch.float32).to(pipe.device, pipe.dtype)
    pasos = []

    def a_imagen(latentes):
        with torch.no_grad():
            x = pipe.vae.decode(latentes / pipe.vae.config.scaling_factor).sample
        return pipe.image_processor.postprocess(x, output_type="pil")[0]

    def al_terminar_paso(p, i, t, kw):
        if con_pasos and i < PASOS - 1:  # se achica como lo ve la red, para que el ruido vaya bajando de a poco
            sigma = float(p.scheduler.sigmas[i + 1])
            pasos.append((f"paso {i + 1}", a_imagen(kw["latents"] / (sigma ** 2 + 1) ** 0.5)))
        return kw

    t0 = time.time()
    img = pipe(frase, num_inference_steps=PASOS, guidance_scale=0.0, latents=ruido.clone(),
               callback_on_step_end=al_terminar_paso, callback_on_step_end_tensor_inputs=["latents"]).images[0]
    seg = time.time() - t0
    if con_pasos:
        pasos = [("ruido", a_imagen(ruido))] + pasos + [(f"paso {PASOS}: listo", img)]
    return img, pasos, seg


def nombre_archivo(frase):
    return re.sub(r"[^\wáéíóúñü]+", "_", frase.lower()).strip("_")[:80] + ".png"


def tira_de_pasos(pasos, frase, archivo):
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, len(pasos), figsize=(3.2 * len(pasos), 3.9), facecolor="#f6f6f4")
    for ax, (rotulo, img) in zip(ejes, pasos):
        ax.imshow(img)
        ax.set_title(rotulo, color=MARINO, fontsize=12, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"De ruido a dibujo, paso a paso: «{frase}»", color=MARINO, fontsize=13, fontweight="bold")
    fig.savefig(archivo, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return archivo


def galeria(pipe):
    import matplotlib.pyplot as plt
    SALIDA.mkdir(exist_ok=True)
    fig, ejes = plt.subplots(2, 3, figsize=(12, 9), facecolor="#f6f6f4")
    tiempos = []
    for k, (ax, frase) in enumerate(zip(ejes.flat, FRASES_GALERIA)):
        img, pasos, seg = dibujar(pipe, frase, semilla=SEMILLAS_GALERIA[k], con_pasos=(k == 0))
        tiempos.append(seg)
        img.save(SALIDA / nombre_archivo(frase))
        if pasos:
            tira_de_pasos(pasos, frase, CARPETA / "pasos_profe.png")
        ax.imshow(img)
        ax.set_title(textwrap.fill(f"«{frase}»", 34), color=MARINO, fontsize=10.5)
        ax.axis("off")
        print(f"  {seg:4.1f} s · {frase}", flush=True)
    fig.suptitle("Galería del profe: cada imagen salió de la frase de arriba (sd-turbo, 4 pasos)",
                 color=MARINO, fontsize=13, fontweight="bold")
    fig.savefig(CARPETA / "galeria_profe.png", dpi=80, bbox_inches="tight")
    print(f"✔ galeria_profe.png y pasos_profe.png listos · {sum(tiempos[1:]) / (len(tiempos) - 1):.1f} s por imagen "
          f"(la primera, {tiempos[0]:.1f} s, incluye calentar el modelo)")


def abrir(archivo):
    if sys.platform == "darwin":
        subprocess.run(["open", str(archivo)])
    else:
        print(f"Imagen guardada en {archivo}")


def interactivo(pipe, con_pasos):
    SALIDA.mkdir(exist_ok=True)
    print("Escribe una frase y presiona Enter (Ctrl+C para salir). Sugerencia: sin personas reales ni marcas.")
    semilla = 0
    while True:
        frase = input("\nFrase: ").strip()
        if not frase:
            continue
        img, pasos, seg = dibujar(pipe, frase, semilla, con_pasos)
        archivo = SALIDA / nombre_archivo(frase)
        img.save(archivo)
        print(f"✔ {seg:.1f} s · {archivo.name}")
        abrir(tira_de_pasos(pasos, frase, SALIDA / ("pasos_" + archivo.name)) if con_pasos else archivo)
        semilla += 1


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--pasos", action="store_true", help="mostrar los pasos de ruido a dibujo")
    args.add_argument("--galeria", action="store_true", help="grabar la galería para el notebook y la Release")
    a = args.parse_args()
    pipe = cargar()
    try:
        galeria(pipe) if a.galeria else interactivo(pipe, a.pasos)
    except KeyboardInterrupt:
        print("\nChao 👋")
