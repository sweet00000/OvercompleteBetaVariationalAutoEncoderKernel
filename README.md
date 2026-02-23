# obvaekernel

`obvaekernel` is a NumPy-first implementation of an overcomplete beta-VAE kernel feature map designed for PCA and kernel-method workflows.

## Install

```bash
pip install .
```

Benchmark/data extras:

```bash
pip install .[benchmark]
```

Development/build extras:

```bash
pip install .[dev]
```

## Quick Usage

```python
import numpy as np
from obvaekernel import OBVAEKernel

X = np.random.randn(256, 12).astype("float32")
model = OBVAEKernel(latent_dim="auto", epochs=25, random_state=42)
Z = model.fit_transform(X)
K = model.kernel_matrix(X)
```

## Benchmark Scripts

Run the root benchmark harness:

```bash
python main.py --mode quick --no-plots
```

Run the examples copy:

```bash
python examples/benchmark_main.py --mode full --include-weather
```

## Build and Publish Checks

Build wheel + sdist:

```bash
python -m build
```

Validate distribution metadata:

```bash
python -m twine check dist/*
```

