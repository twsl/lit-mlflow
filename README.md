# lit-mlflow


[![Build](https://github.com/twsl/lit-mlflow/actions/workflows/build.yaml/badge.svg)](https://github.com/twsl/lit-mlflow/actions/workflows/build.yaml)
[![Documentation](https://github.com/twsl/lit-mlflow/actions/workflows/docs.yaml/badge.svg)](https://github.com/twsl/lit-mlflow/actions/workflows/docs.yaml)
<!--- [![PyPI - Package Version](https://img.shields.io/pypi/v/lit-mlflow?logo=pypi&style=flat&color=orange)](https://pypi.org/project/lit-mlflow/) -->
<!--- [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lit-mlflow?logo=pypi&style=flat&color=blue)](https://pypi.org/project/lit-mlflow/) -->
[![Docs with MkDocs](https://img.shields.io/badge/MkDocs-docs?style=flat&logo=materialformkdocs&logoColor=white&color=%23526CFE)](https://squidfunk.github.io/mkdocs-material/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/j178/prek)
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

With [`uv`](https://docs.astral.sh/uv/):
```bash
uv add lit-mlflow
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
uv run mkdocs build -f ./mkdocs.yml -d ./_build/
```


## Update template

```bash
copier update --trust -A --vcs-ref=HEAD
```


## Credits

This project was generated with [![🚀 python project template.](https://img.shields.io/badge/python--project--template-%F0%9F%9A%80-brightgreen)](https://github.com/twsl/python-project-template)
