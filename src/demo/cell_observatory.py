

import torch
import torch.nn as nn
from mamba_ssm import Mamba 
import time

def benchmark_data_08():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initiating Data 08 Telemetry Benchmark on: {device.upper()}\n")
    
    # CHRONOS Data 08 Architecture: 114 Dimensions (100 Phase + 14 Chem)
    d_model = 114  
    batch_size = 1 
    
    # Simulating continuous optical video (Frames / Sequence Length)
    # We step up the temporal window until the Transformer explodes.
    seq_lengths = [1024, 4096, 16384, 65536, 131072] 

    # The competing backends
    transformer_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=6, batch_first=True).to(device)
    mamba_layer = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2).to(device)

    print(f"{'Temporal Context (Frames)':<28} | {'Transformer VRAM (O(L^2))':<28} | {'Mamba SSM VRAM (O(L))':<25}")
    print("-" * 85)

    for L in seq_lengths:
        # Generate synthetic Data 08 tensor
        x = torch.randn(batch_size, L, d_model).to(device)
        
        # 1. Test Transformer (His AOVIFT proxy)
        try:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            _ = transformer_attn(x, x, x)
            if device == "cuda":
                t_mem = f"{torch.cuda.max_memory_allocated() / (1024**2):.1f} MB"
            else:
                t_mem = "N/A (CPU)"
        except RuntimeError as e:
            if "OutOfMemory" in str(e) or "OOM" in str(e):
                t_mem = "💥 OOM CRASH"
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                t_mem = "FAILED"
                
        # 2. Test Mamba SSM (Your CHRONOS proxy)
        try:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            _ = mamba_layer(x)
            if device == "cuda":
                m_mem = f"{torch.cuda.max_memory_allocated() / (1024**2):.1f} MB"
            else:
                m_mem = "N/A (CPU)"
        except RuntimeError as e:
            if "OutOfMemory" in str(e) or "OOM" in str(e):
                m_mem = "OOM CRASH"
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                m_mem = "FAILED"

        print(f"{L:<28} | {t_mem:<28} | {m_mem:<25}")

if __name__ == "__main__":
    benchmark_data_08()
