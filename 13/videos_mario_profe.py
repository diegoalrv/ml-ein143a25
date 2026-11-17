"""Arma los videos del Mario del profe y deja lista la carpeta de la Release lab13-mario.

Uso (desde Laboratorios/13, después de entrenar_mario_profe.py):
    python videos_mario_profe.py --modelos ~/mario_lab13 --salida ~/mario_lab13/release

Con cada modelo juega una partida determinista y deja en --salida:
  - mario_profe_XXXXk.zip, aligerado (sin el estado del optimizador: pesa ~3 veces menos y juega igual)
  - mario_profe_XXXXk.mp4, la partida de ese modelo (para el modo sin juego del notebook)
  - mario_progreso.mp4, el video para proyectar: «Después de N pasos de práctica» + clip,
    modelo por modelo, y al final la mejor partida completa
  - progreso_profe.csv y el log del entrenamiento, copiados
"""
import argparse
import shutil
from pathlib import Path

from stable_baselines3 import PPO

from mario_utils import cartel, guardar_video, jugar_partida, miles, nombre_profe

FPS = 15
MAX_CLIP = 15 * FPS  # cada clip intermedio, máximo 15 s (los primeros Marios mueren antes)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", default=str(Path.home() / "mario_lab13"))
    ap.add_argument("--salida", default=str(Path.home() / "mario_lab13" / "release"))
    a = ap.parse_args()
    origen, salida = Path(a.modelos), Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)

    zips = sorted(origen.glob("mario_profe_*k.zip"))
    assert zips, f"No hay modelos mario_profe_*k.zip en {origen}"
    video, partidas = [], []
    for z in zips:
        pasos = int(z.stem.split("_")[-1][:-1]) * 1000
        modelo = PPO.load(z, device="cpu")
        # aligerar: un optimizador recién creado no guarda estado (el de Adam pesa el doble que la red)
        modelo.policy.optimizer = modelo.policy.optimizer_class(modelo.policy.parameters(), lr=2.5e-4,
                                                                **modelo.policy.optimizer_kwargs)
        modelo.save(salida / z.name)
        p = jugar_partida(modelo, etiqueta=f"Mario del profe · {miles(pasos)} pasos")
        p["pasos_practica"] = pasos
        partidas.append(p)
        guardar_video(p["cuadros"], salida / f"{nombre_profe(pasos)}.mp4", FPS)
        video += cartel(f"Después de {miles(pasos)}\npasos de práctica") + p["cuadros"][:MAX_CLIP]
        print(f"{z.stem}: x = {p['x']:5d} · bandera {'sí' if p['bandera'] else 'no'} · "
              f"{p['pasos']} instantes · {(salida / z.name).stat().st_size / 1e6:.1f} MB", flush=True)

    mejor = max(partidas, key=lambda p: (p["bandera"], p["x"], -p["pasos_practica"]))
    video += cartel(f"La mejor partida\n{miles(mejor['pasos_practica'])} pasos de práctica", 2) + mejor["cuadros"]
    guardar_video(video, salida / "mario_progreso.mp4", FPS, escala=3)
    print(f"mario_progreso.mp4: {len(video) / FPS:.0f} s · mejor: {mejor['etiqueta']} (x = {mejor['x']})")

    for extra in ["progreso_profe.csv", "entrenamiento_mario.log"]:
        if (origen / extra).exists():
            shutil.copy(origen / extra, salida / extra)
