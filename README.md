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

## Ambiente

Python 3 (probado con 3.13, solo CPU). Todas las dependencias de los Labs 01 a 14 están en
`requirements.txt`; desde el Lab 10 se necesita además PyTorch (`torch`). Los datasets vienen
incluidos en scikit-learn o seaborn; el Lab 03 trae además `titanic.csv` de respaldo en su carpeta
por si no hay internet. El Lab 04 usa `load_digits`, que también viene empaquetado con
scikit-learn (sin descarga).
El Lab 11 descarga Fashion-MNIST (unos 30 MB; si fallan los espejos de `torchvision`, lo baja de
`storage.googleapis.com`) y, en la parte opcional, CIFAR-10 (unos 170 MB) y los pesos de ResNet-18 (unos 45 MB).

```bash
pip install -r requirements.txt
jupyter lab
```

Verificación rápida del ambiente:

```bash
python3 -c "import sklearn, pandas, numpy, matplotlib, torch; print('ok')"
```

Alternativa sin instalar nada: subir el notebook a [Google Colab](https://colab.research.google.com).
