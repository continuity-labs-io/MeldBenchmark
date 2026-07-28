import torch

def get_optimal_device(verbose: bool = False, allow_mps: bool = True) -> torch.device:
    """
    Detects and returns the best available PyTorch device (CUDA, MPS, or CPU).
    
    Args:
        verbose (bool): If True, prints the selected device.
        allow_mps (bool): If False, ignores MPS and falls back to CPU.
        
    Returns:
        torch.device: The selected PyTorch device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and allow_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    if verbose:
        print(f"Using device: {device}")
        
    return device
