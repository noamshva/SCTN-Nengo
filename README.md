# SCTN Resonator Simulation

This repository contains a minimal Nengo simulation using **SCTN neurons** to build a resonator driven by 4 phase-shifted input signals.

##  Features

- Spiking resonator network based on the SCTN model
- 4 phase-shifted sinusoidal input signals
- Visualizes neuron spiking activity

##  Getting Started

### 1. Create a virtual environment

```bash
python -m venv sctn-env
```

### 2. Activate the environment

**Windows:**
```bash
sctn-env\Scripts\activate
```

**Linux/macOS:**
```bash
source sctn-env/bin/activate
```

### 3. Install dependencies

```bash
pip install nengo matplotlib numpy
```

### 4. Run the simulation

```bash
python main.py
```
