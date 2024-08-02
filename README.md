# lit-mlflow


[![Build](https://github.com/twsl/lit-mlflow/actions/workflows/build.yaml/badge.svg)](https://github.com/twsl/lit-mlflow/actions/workflows/build.yaml)
[![Documentation](https://github.com/twsl/lit-mlflow/actions/workflows/docs.yaml/badge.svg)](https://github.com/twsl/lit-mlflow/actions/workflows/docs.yaml)
[![Docs with MkDocs](https://img.shields.io/badge/MkDocs-docs?style=flat&logo=materialformkdocs&logoColor=white&color=%23526CFE)](https://squidfunk.github.io/mkdocs-material/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](.pre-commit-config.yaml)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/twsl/lit-mlflow/releases)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-border.json)](https://github.com/copier-org/copier)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)


An improved Lightning mlflow logger. Works seamlessly with PyTorch Lightning on Databricks and offers more control compared to the `mlflow.pytorch.autolog` function.


## Features

- Makes `MLflow` logging work with `lightning` and [Databricks](https://www.databricks.com/)


## Installation

With `pip`:
```bash
python -m pip install lit-mlflow
```

With [`poetry`](https://python-poetry.org/):
```bash
poetry add lit-mlflow
```


## How to use it
Replace `mlflow.autolog()` with the `MlFlowAutoCallback`:

```python
from lit_mlflow import MlFlowAutoCallback
import lightning.pytorch as pl

trainer = pl.Trainer(
    callbacks=[
        MlFlowAutoCallback()
    ]
)
```

To support Databricks mlflow, use the `DbxMLFlowLogger` instead of the `MlFlowLogger`:

```python
from lit_mlflow import DbxMLFlowLogger
import lightning.pytorch as pl

trainer = pl.Trainer(
    logger=[
        DbxMLFlowLogger()
    ]
)
```

## Docs

```bash
poetry run mkdocs build -f ./docs/mkdocs.yaml -d ./docs/_build/
```


## Update template

```bash
copier update --trust
```


## Credits

This project was generated with [![🚀 A generic python project template.](https://img.shields.io/badge/python--project--template-%F0%9F%9A%80-brightgreen)](https://github.com/twsl/python-project-template)
