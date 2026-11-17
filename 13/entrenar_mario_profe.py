"""Entrena al Mario del profe con PPO (antes de clase) y guarda fotos de su aprendizaje.

Uso (desde Laboratorios/13):
    nohup python entrenar_mario_profe.py > ~/mario_lab13/entrenamiento_mario.log 2>&1 &
    python entrenar_mario_profe.py --pasos 20000 --salida /tmp/prueba     # prueba corta

Deja en --salida (por defecto ~/mario_lab13, fuera de OneDrive: su sincronización corta las escrituras):
  - mario_profe_0000k.zip, _0050k, _0150k, _0300k, _0600k, _1000k, _2000k (y 3000k-5000k si siguió)
  - progreso_profe.csv: cada 50.000 pasos, 5 partidas deterministas (cada una parte esperando
    0-30 instantes al azar, si no saldrían idénticas): distancia media, máxima y proporción con bandera.
Si en 2 M Mario todavía no toca la bandera, sigue hasta 5 M; si ya la toca, se detiene en 2 M.
"""
import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from mario_utils import SEED, crear_vec, jugar_partida, nombre_profe

HITOS = [50_000, 150_000, 300_000, 600_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000]
CADA = 50_000
META = 2_000_000


class Profe(BaseCallback):
    """Guarda los modelos en los hitos, evalúa cada 50.000 pasos y decide si seguir después de 2 M."""

    def __init__(self, carpeta, hitos):
        super().__init__()
        self.carpeta, self.hitos, self.proxima_eval, self.bandera = carpeta, list(hitos), 0, 0.0
        self.csv = carpeta / "progreso_profe.csv"
        if not self.csv.exists() or not CONTINUAR:  # al continuar se agregan filas al CSV existente
            with open(self.csv, "w", newline="") as f:
                csv.writer(f).writerow(["pasos", "x_media", "x_max", "prop_bandera", "minutos"])
        self.t0 = time.time()

    def evaluar(self):
        partidas = [jugar_partida(self.model, grabar=False, espera_inicial=30, seed=s) for s in range(5)]
        xs = [p["x"] for p in partidas]
        self.bandera = float(np.mean([p["bandera"] for p in partidas]))
        minutos = (time.time() - self.t0) / 60
        with open(self.csv, "a", newline="") as f:
            csv.writer(f).writerow([self.num_timesteps, np.mean(xs), max(xs), self.bandera, round(minutos, 1)])
        print(f"[{time.strftime('%H:%M:%S')}] {self.num_timesteps:>9,} pasos · x media {np.mean(xs):6.0f} · "
              f"x máx {max(xs):5d} · bandera {self.bandera:.0%} · {minutos:.1f} min", flush=True)

    def _on_training_start(self):
        if self.num_timesteps == 0:
            self.evaluar()
        self.proxima_eval = (self.num_timesteps // CADA + 1) * CADA

    def _on_step(self):
        while self.hitos and self.num_timesteps >= self.hitos[0]:
            self.model.save(self.carpeta / nombre_profe(self.hitos.pop(0)))
        if self.num_timesteps >= self.proxima_eval:
            self.evaluar()
            self.proxima_eval += CADA
            if self.num_timesteps >= META and self.bandera > 0:
                print("Ya toca la bandera después de 2 M: me detengo aquí.", flush=True)
                return False
        return True


def hardware():
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        cpu = ""
    return f"{cpu or platform.processor()} · {platform.platform()} · Python {platform.python_version()} · torch {torch.__version__}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasos", type=int, default=5_000_000, help="tope; se detiene en 2 M si ya toca la bandera")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "auto")
    ap.add_argument("--salida", default=str(Path.home() / "mario_lab13"))
    ap.add_argument("--continuar", help="zip de un hito para seguir desde ahí (por ejemplo, si se reinició el equipo)")
    a = ap.parse_args()
    CONTINUAR = bool(a.continuar)

    carpeta = Path(a.salida)
    carpeta.mkdir(parents=True, exist_ok=True)
    print(f"Hardware: {hardware()} · dispositivo {a.device} · {a.n_envs} entornos", flush=True)
    venv = crear_vec(a.n_envs, subprocesos=True)
    if CONTINUAR:
        modelo = PPO.load(a.continuar, env=venv, device=a.device)
        print(f"Sigo desde {a.continuar} ({modelo.num_timesteps:,} pasos)", flush=True)
    else:
        # valores habituales de SB3 para Atari (libreto); los mismos del notebook
        modelo = PPO("CnnPolicy", venv, learning_rate=2.5e-4, n_steps=128, batch_size=256, n_epochs=4,
                     clip_range=0.1, ent_coef=0.01, gamma=0.99, device=a.device, seed=SEED, verbose=0)
        modelo.save(carpeta / nombre_profe(0))  # el Mario que todavía no sabe nada
    t0, inicio = time.time(), modelo.num_timesteps
    modelo.learn(a.pasos - modelo.num_timesteps, reset_num_timesteps=not CONTINUAR,
                 callback=Profe(carpeta, [h for h in HITOS if modelo.num_timesteps < h <= a.pasos]))
    if a.pasos not in HITOS:  # prueba corta
        modelo.save(carpeta / nombre_profe(a.pasos))
    minutos = (time.time() - t0) / 60
    print(f"Listo: {modelo.num_timesteps:,} pasos en {minutos:.1f} min "
          f"({(modelo.num_timesteps - inicio) / (minutos * 60):.0f} pasos/s) · {hardware()}", flush=True)
    venv.close()
