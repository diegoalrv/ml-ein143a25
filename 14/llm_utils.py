"""Herramientas del Lab 14 · Por dentro de una IA (pedacitos, autocompletar, mapa de significados e imágenes).

El notebook solo llama estas funciones. Cada modelo se descarga recién cuando se usa por primera vez.
Si una descarga falla, se entra en MODO SIN RED: las funciones devuelven las salidas que grabó el profe
(resultados_profe.json) para los ejemplos del notebook.

Variables de entorno: LLM_GRABAR=1 graba cada salida en resultados_profe.json (lo usa el profe al
ejecutar la docente) · LLM_SIN_RED=1 parte directo en modo sin red (para probarlo).
"""
import html
import json
import os
import sys
import threading
import time
import urllib.request
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # los estudiantes no tienen cuenta: sin avisos de token
try:
    from huggingface_hub.utils import logging as _registro_hf
    _registro_hf.set_verbosity_error()
except ImportError:
    pass

SEED = 42
RAPIDO = False  # el notebook lo cambia: True = textos generados más cortos, para ensayar el código
URL_RELEASE = "https://github.com/diegoalrv/ml-ein143a25/releases/download/lab14-llm/"
CARPETA = Path(__file__).resolve().parent
ARCHIVO_GRABADOS = CARPETA / "resultados_profe.json"
GRABANDO = os.environ.get("LLM_GRABAR") == "1"
SIN_RED = os.environ.get("LLM_SIN_RED") == "1"
GRABADOS = {}

# Paleta del deck «Clase 14 - Transformers y LLM»
AZUL, NARANJA, MARINO, GRIS = "#2e69b2", "#ba6826", "#1b2740", "#8b929c"
FONDO_PEDACITOS = ["#dbe7f6", "#f6e3d4", "#e9e7e1"]
COLORES_GRUPOS = [AZUL, NARANJA, "#4f9a62", GRIS, MARINO, "#d9a066"]  # verde: el marino se confundía con el azul

MODELOS = {  # nombre: (repositorios en Hugging Face, cómo lo llamamos, tamaño de la descarga)
    "base": (["Qwen/Qwen2.5-0.5B"], "el que solo completa", "≈1 GB"),
    "asistente": (["Qwen/Qwen2.5-0.5B-Instruct"], "el asistente", "≈1 GB"),
    "fichas": (["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"], "el modelo de fichas", "≈0,5 GB"),
    "clip": (["sentence-transformers/clip-ViT-B-32", "sentence-transformers/clip-ViT-B-32-multilingual-v1"],
             "el modelo de fotos y frases (CLIP)", "≈1 GB"),
    "describir": (["HuggingFaceTB/SmolVLM-256M-Instruct"], "el que describe fotos", "≈1 GB"),
}

PREGUNTAS_CURSO = [  # Diego: revisar antes de clase
    "El Certamen 3 de Machine Learning es la última evaluación del semestre: el miércoles 2 de diciembre de 2026, toda la sesión.",
    "Los laboratorios se entregan por AULA-USM, como notebook ejecutado.",
    "Los notebooks de los laboratorios están en el repositorio de GitHub ml-ein143a25.",
    "Las clases de Machine Learning (EIN143A25, paralelo 701) son los miércoles en el Campus Concepción.",
    "En la Clase 13 vimos aprendizaje por refuerzo: un Mario que aprendió a jugar solo.",
    "En la Clase 14 vimos transformers, LLM, embeddings y modelos multimodales.",
    "El proyecto final asigna a cada estudiante un problema de uno de seis tópicos: MLP, CNN, RNN, GAN, aprendizaje por refuerzo o transformers.",
    "En la Clase 4 vimos PCA, que resume muchas columnas en solo dos para poder dibujarlas en un gráfico.",
    "En la Clase 9 vimos cómo medir un modelo con métricas como la exactitud.",
    "En la Clase 11 vimos redes convolucionales (CNN), que aprenden a reconocer imágenes con miles de ejemplos etiquetados.",
    "En la Clase 12 vimos redes recurrentes para pronosticar series y una GAN que aprendió a dibujar dígitos.",
]
DISTRACTORES = [
    "Local de completos y churrascos en el centro.",
    "Paradero de micros a Talcahuano.",
    "Pronóstico: lluvia en Concepción el fin de semana.",
    "Farmacia de turno cerca del campus.",
]


# --- Modo sin red y salidas grabadas ---------------------------------------------

class SinGrabar(Exception):
    """Sin red y esta entrada no está entre las grabadas por el profe."""


class _SinModelo(Exception):
    pass


def _leer_grabados():
    """Busca resultados_profe.json junto a este archivo, en la carpeta actual o en la Release."""
    for ruta in (ARCHIVO_GRABADOS, Path("resultados_profe.json")):
        if ruta.exists():
            GRABADOS.update(json.loads(ruta.read_text(encoding="utf-8")))
            return True
    try:
        urllib.request.urlretrieve(URL_RELEASE + "resultados_profe.json", "resultados_profe.json")
        GRABADOS.update(json.loads(Path("resultados_profe.json").read_text(encoding="utf-8")))
        return True
    except Exception:
        print("⚠ Tampoco se pudo bajar resultados_profe.json. Descárgalo a mano desde\n  "
              + URL_RELEASE.replace("download/", "tag/") + " y déjalo junto al notebook.")
        return False


def activar_sin_red(motivo="se pidió a mano"):
    """Pasa al modo sin red: desde aquí, todo sale de resultados_profe.json."""
    global SIN_RED
    SIN_RED = True
    print("=" * 78 + f"\nMODO SIN RED ({motivo}).\n"
          "DESDE AHORA CADA FUNCIÓN MUESTRA LAS SALIDAS QUE GRABÓ EL PROFE PARA LOS EJEMPLOS\n"
          "DEL NOTEBOOK (resultados_profe.json). LAS PREGUNTAS SE PUEDEN RESPONDER IGUAL.\n" + "=" * 78)
    if not GRABADOS:
        _leer_grabados()


if SIN_RED:  # pedido con LLM_SIN_RED=1: que se vea el mismo aviso que cuando falla la descarga
    activar_sin_red("se pidió con LLM_SIN_RED=1")


def _grabado(funcion, args, calcular):
    """Calcula (y si GRABANDO, guarda) o, en modo sin red, busca la salida grabada."""
    clave = json.dumps([funcion, *args], ensure_ascii=False)
    if not SIN_RED:
        try:
            r = calcular()
        except _SinModelo:
            pass  # la descarga falló: activar_sin_red ya avisó
        else:
            if GRABANDO:
                GRABADOS[clave] = r
                ARCHIVO_GRABADOS.write_text(json.dumps(GRABADOS, ensure_ascii=False), encoding="utf-8")
            return r
    if not GRABADOS:
        _leer_grabados()
    if clave in GRABADOS:
        return GRABADOS[clave]
    raise SinGrabar


def _amable(f):
    """En modo sin red, si piden algo que no está grabado, lo dice sin traceback."""
    def envoltura(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except SinGrabar:
            print(f"⚠ MODO SIN RED: «{f.__name__}» con estos datos no está entre las salidas grabadas por el profe.\n"
                  "  Prueba con los ejemplos que trae el notebook; la pregunta se puede responder igual.")
    envoltura.__name__, envoltura.__doc__ = f.__name__, f.__doc__
    return envoltura


# --- Carga perezosa de los modelos ---------------------------------------------

_CARGADOS = {}
_CANDADO = threading.RLock()  # ponytail: un candado para todos los modelos; basta con 5 modelos


def _cargar(nombre):
    with _CANDADO:
        if nombre in _CARGADOS:
            return _CARGADOS[nombre]
        if SIN_RED:
            raise _SinModelo
        repos, quien, tam = MODELOS[nombre]
        print(f"Cargando {quien} ({tam}; la primera vez se descarga y demora ~1 min)…", flush=True)
        t0 = time.time()
        try:  # primero desde la caché, sin preguntarle nada a internet (una consulta colgada congela la celda)
            _CARGADOS[nombre] = _CARGADORES[nombre](*repos, local_files_only=True)
        except Exception:
            try:
                _CARGADOS[nombre] = _CARGADORES[nombre](*repos)
            except Exception as e:
                activar_sin_red(f"NO SE PUDO DESCARGAR {quien.upper()}: {type(e).__name__}")
                raise _SinModelo from e
        print(f"✔ {quien[0].upper() + quien[1:]} listo ({time.time() - t0:.0f} s)", flush=True)
        return _CARGADOS[nombre]


def _cargar_llm(repo, **kw):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    transformers.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    tok = AutoTokenizer.from_pretrained(repo, **kw)
    # eager: la atención "a la antigua", la única que deja mirar los pesos (Extra E1)
    modelo = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, attn_implementation="eager",
                                                  **kw).eval()
    return tok, modelo


def _cargar_st(*repos, **kw):
    from sentence_transformers import SentenceTransformer
    m = [SentenceTransformer(r, device="cpu", **kw) for r in repos]
    return m[0] if len(m) == 1 else m


def _cargar_vlm(repo, **kw):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    return (AutoProcessor.from_pretrained(repo, **kw),
            AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.float32, **kw).eval())


_CARGADORES = {"base": _cargar_llm, "asistente": _cargar_llm, "fichas": _cargar_st, "clip": _cargar_st,
               "describir": _cargar_vlm}


def precargar(*nombres):
    """Carga ya los modelos pedidos (así las partes siguientes no esperan)."""
    for n in nombres:
        try:
            _cargar(n)
        except _SinModelo:
            return


def _perilla(t):
    """La temperatura como número, con un mensaje claro si viene mal escrita."""
    if isinstance(t, tuple):
        raise TypeError("La perilla se escribe con punto, no con coma: 0.2 y no 0,2.")
    if t is Ellipsis:
        raise TypeError("Falta un valor: reemplaza los ... por un número, por ejemplo 0.2.")
    try:
        return round(float(t), 2)
    except (TypeError, ValueError):
        raise TypeError(f"La perilla tiene que ser un número, por ejemplo 0.2 (llegó {t!r}).") from None


def _n(n):
    return min(n, 12) if RAPIDO and not SIN_RED else n


# --- Mostrar texto ----------------------------------------------------------------

def _caja(titulo, texto, color, inicio=""):
    cuerpo = html.escape(texto)
    if inicio and texto.startswith(inicio):
        cuerpo = f"<b>{html.escape(inicio)}</b>{html.escape(texto[len(inicio):])}"
    cab = f"<div style='font-weight:bold;color:{color};margin-bottom:4px'>{html.escape(titulo)}</div>" if titulo else ""
    return (f"<div style='border-left:5px solid {color};background:#fffefb;padding:8px 12px;margin:4px 0;"
            f"color:{MARINO};font-size:14px;line-height:1.45;white-space:pre-wrap;flex:1'>{cab}{cuerpo}</div>")


class Texto(str):
    """Un texto común y corriente que en el notebook se ve como un recuadro de color."""

    def __new__(cls, texto, color=AZUL, titulo="", inicio=""):
        t = super().__new__(cls, texto)
        t.color, t.titulo, t.inicio = color, titulo, inicio
        return t

    def _repr_html_(self):
        return _caja(self.titulo, self, self.color, self.inicio)


def lado_a_lado(*columnas):
    """Muestra textos uno al lado del otro: lado_a_lado(("título", texto, color), ...)."""
    from IPython.display import HTML, display
    cajas = "".join(_caja(t, x if x is not None else "(sin salida)", c) for t, x, c in columnas)
    display(HTML(f"<div style='display:flex;gap:12px;align-items:stretch'>{cajas}</div>"))


# --- Pedacitos (P1) ---------------------------------------------------------------

@_amable
def mostrar_pedacitos(texto):
    """Dibuja el texto como cajitas de colores, con el número de cada pedacito debajo."""
    from IPython.display import HTML, display

    def calcular():
        tok = _cargar("base")[0]
        e = tok(texto, return_offsets_mapping=True)
        piezas, fin = [], 0
        for i, (a, b) in zip(e.input_ids, e.offset_mapping):
            piezas.append([texto[a:b] if a >= fin else tok.decode([i]), i])  # una letra partida en 2 pedacitos
            fin = max(fin, b)
        return piezas

    piezas = _grabado("pedacitos", [texto], calcular)
    cajas = "".join(
        f"<span style='display:inline-block;background:{FONDO_PEDACITOS[j % 3]};border-radius:6px;padding:3px 7px;"
        f"margin:3px 2px;text-align:center;color:{MARINO};font-family:monospace;font-size:16px'>"
        f"{html.escape(p.replace(' ', '·'))}<br><span style='font-size:10px;color:{GRIS}'>{i}</span></span>"
        for j, (p, i) in enumerate(piezas))
    display(HTML(f"<div style='line-height:1.2'>{cajas}</div><div style='color:{MARINO};margin:2px 0 10px'>"
                 f"<b>{len(piezas)} pedacitos</b> para {len(texto)} caracteres</div>"))


# --- El autocompletar (P2) --------------------------------------------------------

@lru_cache(maxsize=64)
def _logits(texto, modelo):
    import torch
    tok, m = _cargar(modelo)
    with torch.no_grad():
        return m(**tok(texto, return_tensors="pt")).logits[0, -1].float()


def _siguiente(texto, k, modelo, temperatura):
    temperatura = round(float(temperatura), 2)

    def calcular():
        import torch
        tok = _cargar(modelo)[0]
        p = torch.softmax(_logits(texto, modelo) / temperatura, dim=-1)
        top = torch.topk(p, k)
        return [[tok.decode([int(i)]), round(float(v) * 100, 2)] for v, i in zip(top.values, top.indices)]

    return _grabado("siguiente", [texto, k, modelo, temperatura], calcular)


@_amable
def porcentajes_siguiente(texto, k=10, modelo="base"):
    """Tabla con los k pedacitos más probables para lo que sigue y su porcentaje."""
    import pandas as pd
    filas = _siguiente(texto, k, modelo, 1.0)
    return pd.DataFrame({"pedacito": [p.replace(" ", "·") for p, _ in filas], "porcentaje": [v for _, v in filas]},
                        index=pd.RangeIndex(1, len(filas) + 1, name="lugar"))


def _dibujar_barras(ax, texto, filas, temperatura):
    piezas = [repr(p.replace(" ", "·"))[1:-1] for p, _ in filas][::-1]
    valores = [v for _, v in filas][::-1]
    colores = ["#9fb8da"] * (len(valores) - 1) + [AZUL]
    ax.barh(piezas, valores, color=colores)
    for y, v in enumerate(valores):
        ax.text(v, y, f" {v:.1f} %".replace(".", ","), va="center", fontsize=9, color=MARINO)
    ax.set_xlim(0, max(valores) * 1.25 + 1)
    ax.set_xlabel("porcentaje")
    extra = f" · perilla en {temperatura:.1f}".replace(".", ",") if temperatura != 1.0 else ""
    ax.set_title(f"«{texto} …» ¿qué pedacito sigue?{extra}", loc="left", fontsize=11)


@_amable
def barras_siguiente(texto, k=10, modelo="base", temperatura=1.0):
    """Barras con los k pedacitos más probables (como la diapo 3), aplanadas o afiladas con la temperatura."""
    import matplotlib.pyplot as plt
    filas = _siguiente(texto, k, modelo, temperatura)
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * k + 1.2))
    _dibujar_barras(ax, texto, filas, round(float(temperatura), 2))
    plt.show()


@_amable
def completar(texto, n_pedacitos=30, temperatura=0.7, semilla=0, modelo="base"):
    """Continúa el texto eligiendo pedacito por pedacito con la perilla en `temperatura`."""
    n, temperatura = _n(n_pedacitos), _perilla(temperatura)

    def calcular():
        import torch
        tok, m = _cargar(modelo)
        ent = tok(texto, return_tensors="pt")
        torch.manual_seed(semilla)
        with torch.no_grad():
            g = m.generate(**ent, max_new_tokens=n, do_sample=temperatura > 0, temperature=temperatura or None,
                           top_k=0, top_p=1.0, repetition_penalty=1.0, pad_token_id=tok.eos_token_id)
        return tok.decode(g[0, ent.input_ids.shape[1]:], skip_special_tokens=True)

    seguir = _grabado("completar", [texto, n, temperatura, semilla, modelo], calcular)
    return Texto(texto + seguir, AZUL if modelo == "asistente" else GRIS, inicio=texto)


SISTEMA = "Eres un asistente útil. Responde siempre en español, en pocas frases."
INSTRUCCION_CONTEXTO = ("Responde solo con la información de este texto. Si la respuesta no está en el texto, "
                        "di que no lo sabes.\n\nTexto: {contexto}\n\nPregunta: {pregunta}")


@_amable
def preguntar(pregunta, contexto=None, modelo="asistente", temperatura=0.3, semilla=0, n_pedacitos=120):
    """Le hace una pregunta al asistente (con su plantilla de chat). Con `contexto`, debe responder solo con ese texto."""
    if modelo != "asistente":
        return completar(pregunta, n_pedacitos, temperatura, semilla, modelo)
    if isinstance(contexto, (list, tuple)):
        contexto = "\n".join(contexto)
    n, temperatura = _n(n_pedacitos), _perilla(temperatura)

    def calcular():
        import torch
        tok, m = _cargar("asistente")
        usuario = INSTRUCCION_CONTEXTO.format(contexto=contexto, pregunta=pregunta) if contexto else pregunta
        ent = tok.apply_chat_template([{"role": "system", "content": SISTEMA}, {"role": "user", "content": usuario}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True)
        torch.manual_seed(semilla)
        with torch.no_grad():
            g = m.generate(**ent, max_new_tokens=n, do_sample=temperatura > 0, temperature=temperatura or None,
                           top_k=0, top_p=1.0, repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
        return tok.decode(g[0, ent["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    r = _grabado("preguntar", [pregunta, contexto, n, temperatura, semilla], calcular)
    return Texto(r, AZUL, titulo="El asistente (con documento)" if contexto else "El asistente")


def comparar_etapas(pregunta):
    """La misma pregunta al que solo completa (texto crudo) y al asistente (plantilla de chat), lado a lado."""
    base = completar(pregunta, n_pedacitos=40, temperatura=0.3, semilla=0, modelo="base")
    asistente = preguntar(pregunta, n_pedacitos=80)
    lado_a_lado(("El que solo completa (etapa 1)", base, GRIS), ("El asistente (etapas 2 y 3)", asistente, AZUL))


def perilla_temperatura(texto, temperaturas=(0.2, 1.0, 1.8)):
    """Deslizador de la perilla de la creatividad: redibuja las barras y 3 continuaciones cortas.

    Sin ipywidgets (o en modo sin red sin todos los valores grabados), muestra 3 perillas fijas.
    """
    import matplotlib.pyplot as plt
    from IPython.display import HTML, display

    def mostrar(t):
        t = round(float(t), 1)
        try:
            filas = _siguiente(texto, 8, "base", t)
            seguidas = [completar(texto, 12, t, s) for s in range(3)]
        except SinGrabar:
            print(f"⚠ MODO SIN RED: la perilla en {t} no está grabada para «{texto}».")
            return
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        _dibujar_barras(ax, texto, filas, t)
        plt.show()
        display(HTML("<div style='margin-left:8px'><b>3 continuaciones con esta perilla:</b></div>"
                     + "".join(s._repr_html_() for s in seguidas if s)))

    grilla = [round(0.1 * i, 1) for i in range(1, 21)]
    if GRABANDO:  # así el deslizador también funciona en modo sin red
        for t in grilla:
            _siguiente(texto, 8, "base", t)
            [completar(texto, 12, t, s) for s in range(3)]
    try:
        import ipywidgets as w
        perilla = w.FloatSlider(value=0.7, min=0.1, max=2.0, step=0.1, description="perilla", continuous_update=False,
                                readout_format=".1f", layout=w.Layout(width="420px"))
        salida = w.interactive_output(mostrar, {"t": perilla})
        display(w.VBox([perilla, salida]))
    except ImportError:
        print("(sin ipywidgets: tres perillas fijas)")
        for t in temperaturas:
            mostrar(t)


# --- Mapa de significados (P3) ----------------------------------------------------

def _ficha(texto):
    def calcular():
        f = _cargar("fichas").encode([texto], normalize_embeddings=True)[0]
        return [round(float(x), 5) for x in f]
    return _grabado("ficha", [texto], calcular)


@_amable
def fichas(textos):
    """La ficha de cada texto: una fila de números por texto (en los libros, un embedding)."""
    textos = [textos] if isinstance(textos, str) else list(textos)
    return np.array([_ficha(t) for t in textos])


def _parecido(f1, f2):
    return round(max(0.0, float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2)))) * 100, 1)


@_amable
def parecido(a, b):
    """Porcentaje de parecido entre dos textos (0 % = nada que ver, 100 % = mismo significado)."""
    return _parecido(_ficha(a), _ficha(b))


@_amable
def mapa_significados(textos, grupos=None):
    """Achica las fichas a 2 números con PCA (la de la Clase 4) y dibuja cada frase como un punto."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2, random_state=SEED).fit_transform(np.array([_ficha(t) for t in textos]))
    grupos = list(grupos) if grupos is not None else [""] * len(textos)
    nombres = list(dict.fromkeys(grupos))
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for j, g in enumerate(nombres):
        idx = [i for i, gi in enumerate(grupos) if gi == g]
        ax.scatter(xy[idx, 0], xy[idx, 1], s=70, color=COLORES_GRUPOS[j % len(COLORES_GRUPOS)], label=g or None)
        for i in idx:
            t = textos[i] if len(textos[i]) <= 30 else textos[i][:28] + "…"
            ax.annotate(t, xy[i], xytext=(5, 4), textcoords="offset points", fontsize=8.5, color=MARINO)
    ax.set(xticks=[], yticks=[], title="El mapa de significados: frases parecidas quedan cerca")
    if any(nombres):
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    plt.show()


class Resultados(list):
    """Lista de documentos encontrados (el primero es el más parecido); en el notebook se ve como tabla."""

    def _repr_html_(self):
        filas = "".join(f"<tr><td style='text-align:right'><b>{p:.0f} %</b></td><td style='text-align:left'>"
                        f"{html.escape(d)}</td></tr>" for d, p in zip(self, self.parecidos))
        return (f"<table><tr><th>parecido</th><th style='text-align:left'>documento</th></tr>{filas}</table>"
                f"<div style='color:{GRIS};font-size:12px'>Pregunta: «{html.escape(self.pregunta)}»</div>")


@_amable
def buscar(pregunta, documentos, k=3):
    """Los k documentos con la ficha más parecida a la de la pregunta."""
    fp = _ficha(pregunta)
    puntaje = sorted(((_parecido(fp, _ficha(d)), d) for d in documentos), key=lambda x: -x[0])[:k]
    r = Resultados(d for _, d in puntaje)
    r.parecidos, r.pregunta = [p for p, _ in puntaje], pregunta
    return r


# --- Imágenes (P4) -----------------------------------------------------------------

def _con_nombre(foto, nombre):
    foto.info["nombre"] = nombre
    return foto


def fotos_ejemplo():
    """Fotos que vienen con scikit-image (sin internet): gato, cafe, cohete y hubble."""
    from PIL import Image
    from skimage import data
    return {n: _con_nombre(Image.fromarray(f()), n) for n, f in
            [("gato", data.chelsea), ("cafe", data.coffee), ("cohete", data.rocket), ("hubble", data.hubble_deep_field)]}


def subir_foto(ruta=None):
    """En Colab abre el botón para subir; en tu computador pide la ruta. Devuelve la foto (máx. 768 px).

    Si algo falla (se canceló la subida, la foto es HEIC…), avisa y devuelve la foto del cohete.
    """
    from PIL import Image
    try:
        if ruta is None and "google.colab" in sys.modules:
            from google.colab import files
            ruta = next(iter(files.upload()))
        elif ruta is None:
            ruta = input("Ruta de tu foto (por ejemplo, mi_foto.jpg): ").strip().strip("'\"")
        foto = Image.open(ruta).convert("RGB")
    except Exception as e:
        print(f"⚠ No se pudo abrir la foto ({type(e).__name__}). Sigo con la del cohete; "
              "si quieres, vuelve a intentarlo con una foto .jpg o .png.")
        return fotos_ejemplo()["cohete"]
    foto.thumbnail((768, 768))
    return _con_nombre(foto, Path(ruta).name)


def _nombre(foto):
    return getattr(foto, "info", {}).get("nombre", "foto sin nombre")


@_amable
def clasificar_foto(foto, etiquetas):
    """Con CLIP: qué frase está más cerca de la foto en el mapa compartido de fotos y frases."""
    import matplotlib.pyplot as plt
    etiquetas = list(etiquetas)

    def calcular():
        imagenes, textos = _cargar("clip")
        fi = imagenes.encode([foto], normalize_embeddings=True)[0]
        ft = textos.encode(etiquetas, normalize_embeddings=True)
        z = 100 * ft @ fi  # CLIP compara con esta escala
        p = np.exp(z - z.max())
        return [round(float(x) * 100, 2) for x in p / p.sum()]

    pct = _grabado("clasificar", [_nombre(foto), etiquetas], calcular)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.3]})
    a1.imshow(foto)
    a1.axis("off")
    orden = np.argsort(pct)
    a2.barh([etiquetas[i] for i in orden], [pct[i] for i in orden],
            color=["#9fb8da"] * (len(orden) - 1) + [AZUL])
    for y, i in enumerate(orden):
        a2.text(pct[i], y, f" {pct[i]:.1f} %".replace(".", ","), va="center", fontsize=9, color=MARINO)
    a2.set_xlim(0, 118)
    a2.set(xlabel="porcentaje", title="¿Qué frase se parece más a la foto?")
    plt.tight_layout()
    plt.show()


@_amable
def describir_foto(foto, pregunta="Describe la imagen.", n_pedacitos=60):
    """Una IA pequeña (SmolVLM) mira la foto y escribe una descripción pedacito a pedacito."""
    import matplotlib.pyplot as plt
    n = _n(n_pedacitos)

    def calcular():
        import torch
        proc, m = _cargar("describir")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": pregunta}]}]
        ent = proc(text=proc.apply_chat_template(msgs, add_generation_prompt=True), images=[foto],
                   return_tensors="pt", do_image_splitting=False)  # sin cortar en trozos: 5 veces más rápido
        with torch.no_grad():
            g = m.generate(**ent, max_new_tokens=n, do_sample=False)
        return proc.batch_decode(g[:, ent["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    r = _grabado("describir", [_nombre(foto), pregunta, n], calcular)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.imshow(foto)
    ax.axis("off")
    plt.show()
    return Texto(r, AZUL, titulo=f"La IA mira la foto y escribe ({pregunta})")


def mostrar_galeria():
    """La galería grabada del profe (texto a imagen): imágenes con su frase y los pasos de ruido a dibujo."""
    from IPython.display import Image as Img, display
    for nombre in ("galeria_profe.png", "pasos_profe.png"):
        ruta = next((r for r in (CARPETA / nombre, Path(nombre)) if r.exists()), None)
        if ruta is None:
            try:
                urllib.request.urlretrieve(URL_RELEASE + nombre, nombre)
                ruta = Path(nombre)
            except Exception:
                print(f"⚠ No se encontró {nombre}: se publica en la Release lab14-llm "
                      f"({URL_RELEASE.replace('download/', 'tag/')}).")
                continue
        display(Img(filename=str(ruta)))


# --- Extra E1: mirar la atención -----------------------------------------------------

@_amable
def mapa_atencion(frase, capa=0, cabeza=0):
    """Cuánto mira cada pedacito (fila) a cada pedacito anterior (columna), en una capa y cabeza del que solo completa."""
    import matplotlib.pyplot as plt

    def calcular():
        import torch
        tok, m = _cargar("base")
        ent = tok(frase, return_tensors="pt")
        with torch.no_grad():
            att = m(**ent, output_attentions=True).attentions[capa][0, cabeza]
        return [[tok.decode([int(i)]) for i in ent.input_ids[0]], np.round(att.numpy(), 4).tolist()]

    piezas, att = _grabado("atencion", [frase, capa, cabeza], calcular)
    piezas = [p.replace(" ", "·") for p in piezas]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(att, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(piezas)), piezas, rotation=70, fontsize=9)
    ax.set_yticks(range(len(piezas)), piezas, fontsize=9)
    ax.grid(False)
    ax.set(xlabel="pedacito al que mira", ylabel="pedacito que está mirando",
           title=f"Capa {capa}, cabeza {cabeza}: ¿a quién mira cada pedacito?")
    plt.show()


if __name__ == "__main__":  # autochequeo: python llm_utils.py
    assert parecido("Me encanta el completo italiano", "Adoro los hot dogs con palta") > \
        parecido("Me encanta el completo italiano", "El banco me cobró una comisión") + 10
    docs = PREGUNTAS_CURSO + DISTRACTORES
    assert buscar("¿Qué día es la última prueba?", docs)[0] == PREGUNTAS_CURSO[0]
    assert PREGUNTAS_CURSO[0] in \
        buscar("¿Cuándo es el Certamen 3 del curso Machine Learning de la USM Concepción y qué entra?", docs)[:2]
    filas = _siguiente("La capital de Chile es la ciudad de", 5, "base", 1.0)
    assert any("Santiago" in pz for pz, _ in filas), filas
    # entradas que también graba el notebook, para que el autochequeo funcione en modo sin red
    plano, afilado = (_siguiente("Mi mascota favorita es el", 8, "base", t) for t in (2.0, 0.5))
    assert plano[0][1] < afilado[0][1], "la temperatura alta debería aplanar los porcentajes"
    assert len(completar("Mi mascota favorita es el", 12, 0.7, 0)) > len("Mi mascota favorita es el")
    for mala in ((0, 2), ..., "alta"):
        try:
            completar("Hola", 5, mala)
        except TypeError as e:
            assert "perilla" in str(e) or "..." in str(e), e
        else:
            raise AssertionError(f"la perilla {mala!r} debió dar un mensaje claro")
    print("✔ llm_utils OK" + (" (MODO SIN RED)" if SIN_RED else ""))
