import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import spikeinterface.extractors as se

import logging
logger = logging.getLogger(__name__)

class ContinuousHDMEADataset(Dataset):
    """
    A PyTorch Dataset for loading continuous high-density electrophysiology data
    from raw 3Brain (.brw) files. This is designed to simulate data topology of 
    a 1,024-channel MaxOne CMOS array streaming at 20kHz for continuous-time 
    state-space models (e.g., Mamba-2).
    
    The .brw (BrainWave) format is a proprietary data standard developed by 3Brain.
    Under the hood, it is an HDF5 (Hierarchical Data Format version 5) file, which
    is an efficient binary format designed to store and organize large amounts of data.
    It contains hierarchical groups and datasets including:
      - Raw voltage traces (typically stored as a massive 2D array: Time x Channels)
      - Metadata such as sampling frequency, resolution, and recording configuration
      - Spatial layouts mapping channels to physical electrode coordinates
      
    By using spikeinterface, we abstract away the low-level HDF5 reading, allowing us
    to lazily slice directly into this binary blob without loading the entire multi-GB
    recording into memory.
    """
    def __init__(self, brw_file_path: str, sequence_length: int = 10000, target_channels: int = 1024):
        """
        Args:
            brw_file_path: Path to the .brw file.
            sequence_length: Number of time frames per sequence chunk.
            target_channels: Number of channels to subsample from the recording.
        """
        super().__init__()
        self.brw_file_path = brw_file_path
        self.sequence_length = sequence_length
        self.target_channels = target_channels
        
        # Load the recording lazily using spikeinterface
        self.recording = se.read_biocam(self.brw_file_path)
        
        # Extract total frames and sampling rate
        self.total_frames = self.recording.get_num_frames()
        self.sampling_rate = self.recording.get_sampling_frequency()
        
        # Calculate number of available non-overlapping chunks
        self.num_chunks = int(self.total_frames // self.sequence_length)
        
    def __len__(self):
        """Returns the total number of non-overlapping sequence chunks."""
        return self.num_chunks
        
    def __getitem__(self, idx):
        """
        Fetches a temporal chunk of traces and subsamples spatially.
        
        Args:
            idx: Index of the non-overlapping chunk.
            
        Returns:
            torch.Tensor of shape [Sequence_Length, Channels] in float32.
        """
        if idx >= self.num_chunks or idx < 0:
            raise IndexError("Dataset index out of range.")
            
        start_frame = idx * self.sequence_length
        end_frame = start_frame + self.sequence_length
        
        # Fetch the temporal chunk of traces using get_traces
        # get_traces returns shape [num_frames, num_channels]
        traces = self.recording.get_traces(start_frame=start_frame, end_frame=end_frame)
        
        # Subsample the spatial dimension to the target_channels
        traces = traces[:, :self.target_channels]
        
        # Convert to float32 and return as PyTorch tensor
        return torch.tensor(traces, dtype=torch.float32)

if __name__ == "__main__":
    # Define default directory relative to the repository root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    default_dir = os.path.join(repo_root, "dataset", "ephys")
    
    # Path to the dummy example file
    dummy_file_path = os.path.join(default_dir, "example.brw")
    
    if not os.path.exists(dummy_file_path):
        logger.warning(f"Warning: Dummy file not found at {dummy_file_path}.")
        logger.info("Please place a valid 'example.brw' file there to run the test.")
    
    try:
        # Instantiate the dataset
        dataset = ContinuousHDMEADataset(
            brw_file_path=dummy_file_path, 
            sequence_length=10000, 
            target_channels=1024
        )
        
        logger.info(f"Dataset successfully initialized.")
        logger.info(f"Total frames: {dataset.total_frames}")
        logger.info(f"Sampling rate: {dataset.sampling_rate} Hz")
        logger.info(f"Total chunks: {len(dataset)}")
        
        # Instantiate DataLoader with batch size 4
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Iterate over the first batch and print out the resulting tensor shape
        for batch_idx, batch in enumerate(dataloader):
            logger.info(f"Batch {batch_idx + 1} tensor shape: {batch.shape}")
            break
            
    except Exception as e:
        logger.error(f"Failed to run the dataset test: {e}")
