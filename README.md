# Slope Stability Calculation Software

## Required Libraries

Install these dependencies before running the project:

- Python 3.10+ (recommended)
- `numpy`
- `matplotlib`
- `PyQt6`

Install with:

```bash
pip install numpy matplotlib PyQt6
```

## Run

Main entry:

```bash
python main.py
```

This launches the GUI application for slope stability analysis.

## Overview

### `main.py` (Main Program)

`main.py` is the core application entry and the GUI controller.  
It does not re-implement each method. Instead, it organizes inputs, calls method modules, and presents results.

Key responsibilities:

- Builds the full PyQt6 interface (input panel, method selector, result area, plotting canvas).
- Provides two operation modes:
  - **Main**: normal engineering calculation mode.
  - **Evaluation**: comparison/sensitivity mode for parameter studies.
- Dispatches calculation requests to:
  - Fellenius method
  - Bishop method
  - Taylor method
  - Janbu GPS method
- Visualizes outputs:
  - FoS contour maps and critical slip circles for circle-search methods.
  - Taylor chart plots with highlighted solution points.
  - Result summary text (FoS, circle center, radius, stability number `N`, etc.).
- Supports evaluation result export (figures and CSV), useful for thesis reporting and reproducibility.

In short, `main.py` is the integration layer of the software: GUI + workflow + algorithm orchestration.

