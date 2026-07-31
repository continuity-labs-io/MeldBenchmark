# MeldBenchmark

MeldBenchmark is a research repository for benchmarking continuous-time multiscale biological datasets using State Space Models (SSMs). The project focuses on fusing high-frequency electrophysiological data with lower-frequency optical imaging, orthogonalizing hardware artifacts, and performing real-time biological anomaly detection using self-supervised predictive coding.

## Core Concepts

### The "Drowning Signal" Environment
The `ToyBiologicalEnvironment` simulates a realistic, complex multiscale biological recording:
- **GEVI (Genetically Encoded Voltage Indicator) Data:** Sampled at 20kHz, containing sparse 1ms biological spikes (action potentials).
- **Optical Data:** Sampled at 100Hz, representing a continuous macro-biological state.
- **Artifacts:** Both modalities are corrupted by a massive 2Hz sine wave representing a mechanical pump vibration (the "Drowning Signal").
- **Anomalies:** The environment supports injecting "Corrosion" (hardware failure causing baseline drift) and "Toxic Shock" (biological crashes causing variance explosions).

### State Space Engine (Fusion Core)
The core architecture (`src/models/state_space_engine.py`) uses a **Mamba SSM** to model the continuous kinetic trajectory of the biological state:
1. **Edge Compression:** A 1D Convolution processes the 20kHz GEVI data into a lower-dimensional latent representation.
2. **Fusion:** The compressed GEVI latents are fused with the 100Hz optical stream.
3. **Forward Predictive Coding:** The Mamba SSM performs self-supervised forward prediction, outputting a "Surprise" metric (Cosine Distance) between predicted and actual future states.
4. **Orthogonal Veto:** The network is trained on homeostasis data to mathematically isolate biological spikes and orthogonalize (veto) the massive 2Hz pump artifact.

### Thermodynamic Metrics & Latency Benchmarks
The repository evaluates the thermodynamic stability and phase space geometry of the biological manifold (`src/metrics/thermodynamics.py`):
- **PALC (Pseudo-Arclength Continuation):** Exact Jacobian-based phase space volume tracking.
- **DMD (Dynamic Mode Decomposition):** A sliding-window approximation of the Koopman operator.
- **Latency Benchmarks:** `src/metrics/bifurcation_benchmark.py` compares the latency of exact Jacobian computation (~730ms) vs. DMD (~1.3ms), validating DMD as a viable proxy for 100Hz live-streaming applications.

### Interpretability (SPD)
The repository includes experimental integration with the Goodfire AI **Stochastic Parameter Decomposition (SPD)** library (`src/metrics/spd_interpreter.py`). It applies structural interpretability techniques specifically to the internal 1D Convolutional components of the Mamba architecture to decompose and understand its feature isolation capabilities.

## Getting Started

### Installation
Ensure you have PyTorch and the required dependencies installed (including `mamba_ssm` and `transformers==4.40.1`):
```bash
pip install -r requirements.txt
```
*(Note: Mamba SSM has known precision issues on Apple Silicon (MPS). The demo scripts automatically default to CPU for the Mamba training loop to prevent NaN instabilities.)*

### Running the Demos
The repository contains two main demonstration scripts:

#### 1. End-to-End Compiler Demo (`0_concat_demo.py`)
This script demonstrates the full multi-modal pipeline of the MELD system. It ingests realistic AO-LLSM optical telemetry, injects synthetic high-frequency GEVI bioelectric data, and processes the fused temporal sequence through the continuous Mamba model. It outputs three core thermodynamic metrics:
- Koopman-Stability-Metric (KSM) via Dynamic Mode Decomposition
- Critical Slowing Down (CSD) for structural wobble
- Morphological Hysteresis (Scar Area) during biological rescue

It also includes a hardware telemetry benchmark comparing VRAM scaling of Mamba vs. a legacy Transformer.
```bash
python src/demo/0_concat_demo.py
```

#### 2. Orthogonal Veto Training (`1_hssm_demo.py`)
To train the Fusion Core and generate the inference dashboard:
```bash
python src/demo/1_hssm_demo.py
```
This will train the model and output a visual dashboard to `output/1_hssm_veto_proof.png` demonstrating the network's ability to zero out the pump artifact and detect true biological crashes vs. hardware failures.

### Running the Latency Benchmark
To compare the execution speed of exact Jacobian metrics vs. DMD approximations:
```bash
python src/metrics/bifurcation_benchmark.py
```
