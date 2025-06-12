# PyTerp

A 3D interpolator for **Python** designed for maximum speed on large datasets. It accelerates the IDW algorithm with a parallelized C++ core (`OpenMP`) and optimized k-NN searches (`nanoflann`).

## Theoretical Summary

The interpolation is performed in a two-step process that combines the k-NN and IDW algorithms.

1. **Neighbor Selection (k-NN)**: For each point where a value is to be estimated, the _k-Nearest Neighbors_ algorithm first finds the k closest known source points in space. The efficiency of this search is ensured by an optimized data structure (`k-d tree`).

2. **Value Calculation (IDW)**: Next, the _Inverse Distance Weighting_ method calculates the final value as a weighted average of the k found neighbors. The weight of each neighbor is inversely proportional to its distance (weight = 1/distanceᵖ, where `p` is a power parameter), causing closer points to have a much greater influence on the result.

## Prerequisites

Before you begin, ensure you have the following software installed:

* **Python 3.10+**
* **Git**
* **A C++ compiler**: This package contains C++ code that needs to be compiled during installation.
    * **Windows**: Install Visual Studio Build Tools (select the "Desktop development with C++" workload).
    * **Linux (Debian/Ubuntu)**: Install build-essential with: sudo apt-get install build-essential.

## Installation

### PyPI

#### Install the package:

```bash
pip install pyterp
```

---

### GitHub

#### 1. Clone the repository:

```bash
git clone https://github.com/jgmotta98/PyTerp.git
cd PyTerp
```

#### 2. Create and activate a virtual environment:

```bash
# Create the environment
python -m venv .venv

# Activate the environment
# On Windows (cmd.exe):
.venv\Scripts\activate
# On macOS/Linux (bash/zsh):
source .venv/bin/activate
```

#### 3. Install the requirements:

```bash
pip install -r requirements.txt
```

#### 4. Install the package:

```bash
pip install .
```

## Usage Example

For a complete and runnable example, including the creation and preparation of input data, please see the scripts in the [examples](examples/basic_usage.py) folder.

## Acknowledgements

This project uses `nanoflann`, a high-performance C++ library for the _k-Nearest Neighbors_ algorithm. The efficiency of nanoflann's k-d tree implementation is fundamental to this interpolator's performance.

* **Official Repository:** [Nanoflann](https://github.com/jlblancoc/nanoflann)