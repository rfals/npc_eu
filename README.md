# npc eu

Estimating Phillips Curve and the output gap in the Eurozone by using explainable neural networks.

## About

This is the working repository of Reinis Fals and Igors Tatarinovs for the development of Bachelor's thesis about Estimating Phillips Curve and the output gap in the Eurozone by using explainable neural networks. The thesis expands on Columbe's, 2022 paper titled [A Neural Phillips Curve and a Deep Output Gap](https://doi.org/10.48550/arxiv.2202.04146) and extends the Hemisphere Neural Network model to EU macroeconomic climate.



### Introduction

In the realm of economic forecasting, accurately modeling the Phillips Curve (further in the research - PC)—the relationship between inflation and unemployment—remains a formidable challenge. Traditional approaches, such as Ordinary Least Squares (OLS) linear regressions, have offered limited insights, primarily due to their inability to capture the complex, nonlinear dynamics inherent within economic data. This research proposes a novel approach by exploring the application of explainable neural networks (NN) to estimate the PC and the output gap within the Eurozone, aiming to enhance both predictive accuracy and interpretability.


## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
