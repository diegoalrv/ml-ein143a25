"""Herramientas del Lab 13 · Mario aprende solo (PPO sobre SuperMarioBros-1-1-v0).

Las usan el notebook, entrenar_mario_profe.py y videos_mario_profe.py, para que todos
jueguen exactamente el mismo Mario. Si el juego no se puede instalar (por ejemplo, Python
menor que 3.13), HAY_JUEGO queda en False y igual funcionan mostrar_video y la descarga.
"""
import math
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    import gym_super_mario_bros  # noqa: F401  (registra los entornos SuperMarioBros-*)
    from gym_super_mario_bros.actions import RIGHT_ONLY
    from nes_py.wrappers import JoypadSpace
    HAY_JUEGO = True
except ImportError:
    HAY_JUEGO = False

NIVEL = "SuperMarioBros-1-1-v0"
NOMBRES_BOTONES = ["nada", "derecha", "derecha + saltar", "derecha + correr", "derecha + correr + saltar"]
MARINO = "#1b2740"
SEED = 42

URL_RELEASE = "https://github.com/diegoalrv/ml-ein143a25/releases/download/lab13-mario/"
PASOS_PROFE = [0, 50_000, 150_000, 300_000, 600_000, 1_000_000, 2_000_000]
CARPETA_RELEASE_LOCAL = Path(__file__).resolve().parent.parent / "release_lab13"  # copia del profe


def nombre_profe(pasos):
    """0 -> 'mario_profe_0000k', 2_000_000 -> 'mario_profe_2000k'."""
    return f"mario_profe_{pasos // 1000:04d}k"


def miles(n):
    """1240 -> '1.240' (separador de miles con punto, como en la prosa)."""
    return f"{int(n):,}".replace(",", ".")


# --- El entorno ---------------------------------------------------------------

class Grabador(gym.Wrapper if HAY_JUEGO else object):
    """Va antes del preprocesamiento: guarda la pantalla a color (240x256) y el x_pos de cada instante."""

    def __init__(self, env, cada=4):
        super().__init__(env)
        self.cada, self.cuadros, self.xs, self._n = cada, [], [], 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cuadros, self.xs, self._n = [obs.copy()], [info["x_pos"]], 0
        return obs, info

    def step(self, accion):
        obs, r, terminado, truncado, info = self.env.step(accion)
        self._n += 1
        if self._n % self.cada == 0:  # 1 de cada 4 cuadros = un instante de juego, a velocidad real con 15 fps
            self.cuadros.append(obs.copy())
            self.xs.append(info["x_pos"])
        return obs, r, terminado, truncado, info


class CortaSiSeTraba(gym.Wrapper if HAY_JUEGO else object):
    """Corta la partida si Mario no avanza en `limite` instantes (trabado contra un tubo)."""

    def __init__(self, env, limite=300):
        super().__init__(env)
        self.limite = limite

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.x, self.sin_avanzar = info["x_pos"], 0
        return obs, info

    def step(self, accion):
        obs, r, terminado, truncado, info = self.env.step(accion)
        if info["x_pos"] > self.x:
            self.x, self.sin_avanzar = info["x_pos"], 0
        else:
            self.sin_avanzar += 1
        return obs, r, terminado, truncado or self.sin_avanzar >= self.limite, info


def crear_entorno(para_entrenar=True):
    """Mario 1-1 con 5 botones, 1 instante = 4 cuadros, gris 84x84 y las últimas 4 pantallas apiladas.

    La partida termina al perder una vida: en gym-super-mario-bros 9.x un nivel suelto ya
    termina cuando Mario muere (info["death"] = True) o toca la bandera; el autochequeo de
    abajo lo verifica. Además se corta si no avanza en 300 instantes (trabado contra un tubo).
    Con para_entrenar=False se agrega el Grabador para hacer videos.
    """
    env = JoypadSpace(gym.make(NIVEL), RIGHT_ONLY)
    if not para_entrenar:
        env = Grabador(env)
    env = gym.wrappers.MaxAndSkipObservation(env, skip=4)  # repite el botón 4 cuadros y suma los puntos
    env = CortaSiSeTraba(env, 300)
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    return gym.wrappers.FrameStackObservation(env, 4)


def crear_vec(n_envs=4, subprocesos=False, gamma=0.99):
    """Varios Marios en paralelo, para entrenar.

    VecMonitor anota distancia y bandera de cada partida terminada. VecNormalize achica los
    puntos (solo al entrenar): este juego da hasta ±15 por instante y los valores de PPO para
    Atari esperan puntos cercanos a ±1. Sin esto, a los 200.000 pasos Mario no mejoraba.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

    clase = SubprocVecEnv if subprocesos else DummyVecEnv
    venv = VecMonitor(clase([crear_entorno] * n_envs), info_keywords=("x_pos", "flag_get"))
    return VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=gamma)


def jugar_partida(modelo=None, max_pasos=2000, deterministico=True, etiqueta="", grabar=True,
                  espera_inicial=0, seed=SEED):
    """Juega una partida. modelo=None -> botones al azar.

    Corta si Mario no avanza en 300 instantes (lo hace crear_entorno). espera_inicial=N hace que
    parta esperando entre 0 y N instantes al azar: sirve para que partidas deterministas no sean
    idénticas (el juego es siempre igual, la semilla no lo cambia).
    Devuelve un dict: cuadros (a color, con la franja de la etiqueta si se dio), x, bandera, pasos, puntos.
    """
    env = crear_entorno(para_entrenar=not grabar)
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    if modelo is not None:  # así las partidas no deterministas también se pueden repetir
        import torch
        torch.manual_seed(seed)
    for _ in range(np.random.default_rng(seed).integers(espera_inicial + 1)):
        obs, *_ = env.step(0)
    x, puntos, pasos = info["x_pos"], 0.0, 0
    while pasos < max_pasos:
        if modelo is None:
            accion = env.action_space.sample()
        else:
            accion, _ = modelo.predict(np.asarray(obs), deterministic=deterministico)
        obs, r, terminado, truncado, info = env.step(int(accion))
        pasos, puntos = pasos + 1, puntos + float(r)
        x = max(x, info["x_pos"])
        if terminado or truncado:
            break
    cuadros = []
    if grabar:
        cuadros = env.get_wrapper_attr("cuadros")
        if etiqueta:
            xs = np.maximum.accumulate(env.get_wrapper_attr("xs"))
            cuadros = rotular(cuadros, [f"{etiqueta} · x = {miles(xi)}" for xi in xs])
    env.close()
    return {"cuadros": cuadros, "x": int(x), "bandera": bool(info["flag_get"]),
            "pasos": pasos, "puntos": round(puntos, 1), "etiqueta": etiqueta}


# --- Videos --------------------------------------------------------------------

ALTO_FRANJA = 32  # 240 + 32 = 272 = 17 x 16: el mp4 queda sin reescalar


def _fuente(tam):
    from PIL import ImageFont
    from matplotlib import font_manager  # DejaVu Sans viene con matplotlib y tiene tildes
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), tam)


def rotular(cuadros, texto):
    """Agrega arriba de cada cuadro una franja azul marino con el texto (uno para todos o uno por cuadro)."""
    from PIL import Image, ImageDraw

    textos = [texto] * len(cuadros) if isinstance(texto, str) else texto
    fuente, salida = _fuente(14), []
    for cuadro, t in zip(cuadros, textos):
        img = Image.new("RGB", (cuadro.shape[1], cuadro.shape[0] + ALTO_FRANJA), MARINO)
        img.paste(Image.fromarray(cuadro), (0, ALTO_FRANJA))
        ImageDraw.Draw(img).text((8, 8), t, fill="white", font=fuente)
        salida.append(np.asarray(img))
    return salida


def cartel(texto, segundos=1.5, fps=15):
    """Pantalla azul marino con un texto grande al centro (separa los clips del video de progreso)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (256, 240 + ALTO_FRANJA), MARINO)
    dibujo = ImageDraw.Draw(img)
    dibujo.multiline_text((128, 136), texto, fill="white", font=_fuente(20), anchor="mm", align="center", spacing=8)
    return [np.asarray(img)] * int(segundos * fps)


def guardar_video(cuadros, archivo, fps=15, escala=1):
    import imageio.v2 as imageio

    if escala > 1:  # pixel art: agrandar repitiendo píxeles, sin borronear
        cuadros = [np.repeat(np.repeat(c, escala, 0), escala, 1) for c in cuadros]
    imageio.mimwrite(archivo, cuadros, fps=fps, macro_block_size=16)
    return archivo


def mostrar_video(cuadros, fps=15, texto="", max_seg=30, ancho=512):
    """Escribe un mp4 y lo incrusta en el notebook. Si dura más de max_seg, se acelera."""
    from IPython.display import Video

    if texto:
        cuadros = rotular(cuadros, texto)
    if len(cuadros) > fps * max_seg:
        cuadros = cuadros[::math.ceil(len(cuadros) / (fps * max_seg))]
    archivo = guardar_video(cuadros, tempfile.mktemp(suffix=".mp4"), fps)
    return Video(archivo, embed=True, width=ancho, html_attributes="controls loop")


def mostrar_mp4(archivo, ancho=512):
    """Incrusta un mp4 ya grabado (modo sin juego)."""
    from IPython.display import Video
    return Video(str(archivo), embed=True, width=ancho, html_attributes="controls loop")


# --- Modelos del profe ------------------------------------------------------------

def descargar_modelos_profe(carpeta="modelos_mario", solo_videos=False):
    """Baja de la Release lab13-mario los modelos, sus videos y progreso_profe.csv a `carpeta`.

    Lo que ya está en la carpeta no se vuelve a bajar. Sin internet, usa lo que encuentre
    (en `carpeta` o en la copia local del profe) y dice con claridad qué falta.
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(exist_ok=True)
    archivos = ["progreso_profe.csv", "mario_progreso.mp4"] + [nombre_profe(p) + ".mp4" for p in PASOS_PROFE]
    if not solo_videos:
        archivos += [nombre_profe(p) + ".zip" for p in PASOS_PROFE]
    faltan = []
    for nombre in archivos:
        destino = carpeta / nombre
        if destino.exists():
            continue
        if (CARPETA_RELEASE_LOCAL / nombre).exists():
            shutil.copy(CARPETA_RELEASE_LOCAL / nombre, destino)
            continue
        try:
            urllib.request.urlretrieve(URL_RELEASE + nombre, destino)
        except Exception:
            destino.unlink(missing_ok=True)
            faltan.append(nombre)
    hay = sorted(p.name for p in carpeta.iterdir() if p.suffix in (".zip", ".mp4", ".csv"))
    print(f"En {carpeta}/ hay {len(hay)} archivos del profe: "
          f"{sum(n.endswith('.zip') for n in hay)} modelos, {sum(n.endswith('.mp4') for n in hay)} videos"
          f"{' y progreso_profe.csv' if 'progreso_profe.csv' in hay else ''}.")
    if faltan:
        print(f"NO SE PUDIERON BAJAR {len(faltan)} ARCHIVOS (¿sin internet?): {', '.join(faltan)}")
        print(f"Bájalos a mano desde {URL_RELEASE.replace('download/', 'tag/')} y déjalos en {carpeta}/")
    return carpeta


if __name__ == "__main__":  # autochequeo: python mario_utils.py
    assert HAY_JUEGO, "No se pudo importar gym_super_mario_bros"
    env = crear_entorno()
    obs, info = env.reset(seed=SEED)
    assert np.asarray(obs).shape == (4, 84, 84), np.asarray(obs).shape
    assert env.action_space.n == 5 and {"x_pos", "flag_get", "life"} <= info.keys()
    vidas = info["life"]
    while True:  # al azar hasta que termine: debe ser por perder una vida, no por game over
        obs, r, terminado, truncado, info = env.step(env.action_space.sample())
        if terminado or truncado:
            break
    assert info["death"] or info["flag_get"], info
    print(f"✔ la partida terminó al morir (death={info['death']}, life {vidas} -> {info['life']}, x={info['x_pos']})")
    p = jugar_partida(None, etiqueta="prueba")
    assert p["x"] > 0 and p["cuadros"][0].shape == (272, 256, 3), (p["x"], p["cuadros"][0].shape)
    guardar_video(p["cuadros"] + cartel("Después de 50.000\npasos de práctica"), tempfile.mktemp(suffix=".mp4"))
    print(f"✔ mario_utils OK · al azar: x = {p['x']}, {p['pasos']} instantes, {len(p['cuadros'])} cuadros")
