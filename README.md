# Laboratorios — Machine Learning (EIN143A25)

Notebooks de laboratorio del curso **Machine Learning** (EIN143A25), Paralelo 701,
Universidad Técnica Federico Santa María, semestre **2026-2**.

Docente: Diego Ramírez · diego.ramirezvas@usm.cl

Cada laboratorio vive en su propia carpeta (`01/`, `02/`, …) y viene en dos versiones:
`*_estudiante.ipynb` (con las tareas por resolver) y `*_docente.ipynb` (resuelto, con notas de
conducción de la clase).

| # | Sesión | Tema | Dataset |
|---|--------|------|---------|
| 01 | 05-08-2026 | Introducción a ML: ambiente, exploración y primer modelo end-to-end | Wine (`sklearn.datasets.load_wine`) |
| 02 | 12-08-2026 | Supervisado vs no supervisado: K-means sobre Wine sin etiquetas | Wine (`sklearn.datasets.load_wine`) |
| 03 | 19-08-2026 | Preprocesamiento: escalamiento y regla de oro fit/transform, nulos y categóricas | Wine + Titanic (`seaborn` o `titanic.csv` local) |
| 04 | 26-08-2026 | Selección vs extracción de características: `SelectKBest`, PCA y maldición de la dimensionalidad | Wine + Digits (`sklearn.datasets`) |
| 05 | 02-09-2026 | Train-test split y validación cruzada: varianza del split simple, curva de sobre/subajuste, `cross_val_score`, pipeline completo de cierre de la Unidad 1 | Wine (`sklearn.datasets.load_wine`) |
| 06 | 23-09-2026 | Regresión lineal y logística: coeficientes, MSE y R², `predict_proba`, frontera de decisión lineal en 2D, logística vs KNN | Diabetes + Breast Cancer (+ Wine en ejercicio propuesto) (`sklearn.datasets`) |
| 07 | 30-09-2026 | Árboles de decisión y Random Forest: `plot_tree` y Gini, curva de sobre/subajuste con `max_depth` (split y CV), `feature_importances_`, comparación KNN vs logística vs árbol vs bosque | Wine (+ Breast Cancer en ejercicio propuesto) (`sklearn.datasets`) |
| 10a | 11-11-2026 | Armar la red (Lab 1 del día doble): perceptrón y XOR, apilar neuronas a mano, regla de la cadena (a mano, diferencias finitas y `autograd`), MLP 2-4-1 desde cero en NumPy y red de dígitos con PyTorch con SGD simple (requiere `torch`) | XOR + Digits (`sklearn.datasets.load_digits`) |
| 10b | 11-11-2026 | Entrenar bien la red (Lab 2 del día doble): GD, momentum, RMSprop y Adam a mano en un valle 2D, optimizadores en la red de dígitos, sobreajuste y remedios uno por uno (L2, L1, BatchNorm, Dropout, early stopping) e informe con 3 semillas e IC95 % bootstrap (requiere `torch`) | Digits (`sklearn.datasets.load_digits`) |
| 11 | 18-11-2026 | Redes convolucionales: MLP vs CNN, formas capa por capa, filtros y mapas de características, qué hace la ReLU, convolución a mano en NumPy, BatchNorm, Dropout y aumento de datos por lote, transferencia de aprendizaje con ResNet-18 (opcional) (requiere `torch` y `torchvision`) | Fashion-MNIST + CIFAR-10 (opcional) (`torchvision.datasets`, descarga) |
| 12a | 11-11-2026 | Pronóstico con redes recurrentes (Lab A de la clase alternada): serie inventada cortada en orden y «mañana = hoy», ventanas deslizantes (animación), RNN, LSTM y GRU con la mejor época restaurada, pronóstico recursivo de 100 pasos y cuánta historia mirar (requiere `torch`) | Serie sintética (generada en el notebook) |
| 12b | 11-11-2026 | Una mini-GAN que aprende a dibujar dígitos (Lab B de la clase alternada): falsificador contra policía por turnos, del ruido a los dígitos con el mismo ruido fijo (animación desde el paso 0, antes y después, grilla y curvas de pérdida), ¿salen todos los dígitos? (colapso parcial de modos), reflexión y tabla de tres escenarios; extra opcional: GAN en 1D con dos montes y un policía 10 veces más rápido (requiere `torch`) | Digits (`sklearn.datasets.load_digits`) + dos montes sintéticos (extra) |
| 13 | 18-11-2026 | Aprendizaje por refuerzo: **el jugador aprende a cruzar el lago** (`Lab13_FrozenLake_*.ipynb`, lab de la clase). El jugador al azar, la tabla Q de la diapo 9 y cómo se llena desde la meta hacia atrás, el hielo resbaloso, el desafío del lago de 8×8 (eliges tú los valores), explorar vs. explotar, de qué están hechos los puntos y, al cierre, el video del Mario del profe (cuando la tabla no alcanza). Corre en Colab en segundos. *Para la casa (opcional):* **Mario aprende a jugar solo** (`Lab13_Mario_*.ipynb`): entrenar tu propio Mario con PPO durante 15 minutos y compararlo con el del profe (requiere Python 3.13, `gym-super-mario-bros` y `stable-baselines3`; tiene modo sin juego) | `FrozenLake-v1` de Gymnasium (4×4 y 8×8) · nivel 1-1 de `gym-super-mario-bros` 9.1 + modelos del profe en la Release `lab13-mario` |
| 14 | 25-11-2026 | Por dentro de una IA de lenguaje (`Lab14_LLM_*.ipynb`): los **pedacitos** (tokens) y por qué a una IA le cuesta contar letras, el **autocompletar gigante** escrito a mano con el modelo real (porcentajes con `softmax`, un `for` que elige el pedacito más probable, la perilla de la temperatura, la plantilla oculta del asistente, una alucinación en vivo), el **mapa de significados** (embeddings y PCA), **tu propio buscador** y el mensaje de RAG armado a mano, y **fotos**: clasificar con frases propias (CLIP), describir una foto (SmolVLM) y la demo de texto a imagen del profe. Sin fórmulas; corre en la CPU (se recomienda Colab) y tiene modo sin red. *Extra opcional:* la atención hecha a mano con NumPy y DistilGPT-2 (`_anterior/Lab14_DistilGPT2_*.ipynb`) | Modelos de Hugging Face (Qwen2.5-0.5B base e Instruct, paraphrase-multilingual-MiniLM-L12-v2, CLIP ViT-B/32 multilingüe, SmolVLM-256M-Instruct) · fotos de `skimage.data` · salidas grabadas y galería en la Release `lab14-llm` |

## Ambiente

Python 3 (probado con 3.13, solo CPU). Todas las dependencias de los Labs 01 a 14 están en
`requirements.txt`; desde el Lab 10 se necesita además PyTorch (`torch`). Los datasets vienen
incluidos en scikit-learn o seaborn; el Lab 03 trae además `titanic.csv` de respaldo en su carpeta
por si no hay internet. El Lab 04 usa `load_digits`, que también viene empaquetado con
scikit-learn (sin descarga).
El Lab 11 descarga Fashion-MNIST (unos 30 MB; si fallan los espejos de `torchvision`, lo baja de
`storage.googleapis.com`) y, en la parte opcional, CIFAR-10 (unos 170 MB) y los pesos de ResNet-18 (unos 45 MB).
El lab del lago (Lab 13) corre con cualquier Python reciente. El notebook de Mario (Lab 13, para la casa) necesita **Python 3.13 o más** (lo pide `gym-super-mario-bros` 9.x, que ya trae el juego: no hay que conseguir nada aparte) y baja de la Release [`lab13-mario`](https://github.com/diegoalrv/ml-ein143a25/releases/tag/lab13-mario) los modelos y videos del Mario del profe (unos 50 MB). Si el juego no se puede instalar (por ejemplo, en un Colab con Python más antiguo), el notebook entra en modo sin juego y muestra los videos ya grabados.
El Lab 14 descarga de Hugging Face 5 modelos pequeños (unos 4 GB en total, sin cuenta ni licencia que aceptar): conviene correrlo en Colab, o bajarlos antes con `python 14/precargar_modelos.py`. Si una descarga falla, entra en **modo sin red** y muestra las salidas grabadas por el profe (`resultados_profe.json`, en la Release [`lab14-llm`](https://github.com/diegoalrv/ml-ein143a25/releases/tag/lab14-llm)). La demo de texto a imagen (`14/texto_a_imagen_profe.py`, sd-turbo) es solo para el profe.

```bash
pip install -r requirements.txt
jupyter lab
```

Verificación rápida del ambiente:

```bash
python3 -c "import sklearn, pandas, numpy, matplotlib, torch; print('ok')"
```

Alternativa sin instalar nada: subir el notebook a [Google Colab](https://colab.research.google.com).
