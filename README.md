# Statistical training with the scipy library
## 1. What is SciPy and what is it used for?

SciPy is built on top of NumPy—it leverages NumPy’s powerful numeric arrays and adds a vast collection of scientific and numerical computing tools. Whenever “numerical and scientific computing” is involved, SciPy is almost certainly part of the stack.

### Key capabilities of SciPy

In general, SciPy is used for tasks such as:

- **Numerical integration and solving equations**  
- **Optimization** (minimizing/maximizing functions—e.g., for model fitting)  
- **Advanced linear algebra** (matrix operations, decompositions, etc.)  
- **Statistics and probability** (via `scipy.stats`)  
- **Signal processing** (filtering, FFT, etc.)  
- **Image processing** (basic image operations)  
- **Spatial data analysis** (distances, k-d trees, nearest neighbors, etc.)

In short: SciPy provides the mathematical and statistical backbone for **data science, engineering, and classical machine learning**—anything beyond what basic NumPy offers.

### Important SciPy submodules (subpackages)

Some of the most widely used include:

| Submodule            | Purpose |
|----------------------|--------|
| `scipy.linalg`       | Linear algebra (more advanced and faster than `numpy.linalg`) |
| `scipy.optimize`     | Optimization, nonlinear regression, function minimization/maximization |
| `scipy.integrate`    | Numerical integration and solving differential equations |
| `scipy.fft`          | Fast Fourier Transform |
| `scipy.signal`       | Signal processing (filtering, convolution, etc.) |
| `scipy.spatial`      | Distance metrics, KDTree, geometric computations |
| `scipy.interpolate`  | Interpolation (filling gaps between data points) |
| `scipy.stats`        | Statistics and probability (core of statistical analysis in SciPy) |
| `scipy.sparse`       | Sparse matrices for large, memory-efficient datasets |
| `scipy.cluster`      | Basic clustering algorithms |
| `scipy.io`           | Reading/writing special file formats (e.g., MATLAB files) |

### SciPy’s role in Data Science

A typical data science workflow follows this path:

1. **Data collection**  
2. **Data preprocessing & cleaning** → `pandas` / `NumPy`  
3. **Statistical analysis, hypothesis testing, simple modeling** → **SciPy** (especially `stats` and `optimize`)  
4. **Machine learning modeling** → `scikit-learn`, `PyTorch`, `TensorFlow`  
5. **Model evaluation & visualization** → combination of `NumPy`, `SciPy`, `matplotlib`, etc.

Thus, **SciPy acts as the mathematical/statistical toolkit** that enables serious quantitative analysis whenever you need more than just data wrangling.

---

## 2. What exactly is the `scipy.stats` module?

This is likely what you meant by “statis” 😊  
Its correct name is **`scipy.stats`**.

### Main functions of `scipy.stats`

`scipy.stats` is used for a wide range of statistical tasks:

- **Descriptive statistics**: mean, variance, skewness, kurtosis, etc.  
- **Probability distributions**: normal, Poisson, binomial, uniform, gamma, and dozens more  
- **Random sample generation** from these distributions  
- **Fitting distributions** to observed data  
- **Statistical hypothesis tests**:  
  - t-test, chi-square test, ANOVA  
  - Mann-Whitney U, Wilcoxon signed-rank, etc.  
- **Correlation measures**: Pearson, Spearman, Kendall, etc.  
- **Classical regression and inferential statistics**

In plain terms:  
**Whenever you need to “speak the language of statistics” with your data—rather than just observing numbers—`scipy.stats` is the tool you reach for.**





# Statistical training with the Scipi library

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

In this repository, we cover the scipy library in Python.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         statistical_training_with_the_scipi_library and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── statistical_training_with_the_scipi_library   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes statistical_training_with_the_scipi_library a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

