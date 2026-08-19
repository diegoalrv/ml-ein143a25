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

## Ambiente

Python 3 con scikit-learn, pandas, numpy, matplotlib, seaborn y Jupyter. Los datasets vienen
incluidos en scikit-learn o seaborn; el Lab 03 trae además `titanic.csv` de respaldo en su carpeta
por si no hay internet.

```bash
pip install scikit-learn pandas numpy matplotlib seaborn jupyter
jupyter notebook
```

Verificación rápida del ambiente:

```bash
python3 -c "import sklearn, pandas, numpy, matplotlib; print('ok')"
```

Alternativa sin instalar nada: subir el notebook a [Google Colab](https://colab.research.google.com).
