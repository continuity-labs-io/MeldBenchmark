Write a highly optimized Python module to ingest the 'mysteriousauthor/spikeprophecy-steinmetz' dataset using the Hugging Face 'datasets' library. Implement a PyTorch 'IterableDataset' that downloads and streams the 50ms binned spike counts. The dataset must yield sliding history windows of shape (batch, time_steps, num_neurons). Prioritize execution speed and simplicity, ensuring the host memory footprint remains flat regardless of the dataset size.

dependencies:
  - python=3.10
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=12.1
  - huggingface_hub
  - datasets
  - numpy
  - scipy
  - pandas
  - pip
  - pip:
    - mamba-ssm
    - causal-conv1d
    - loguru

# requirements.txt
torch
huggingface-hub
datasets
numpy
scipy
pandas
mamba-ssm
causal-conv1d
loguru
