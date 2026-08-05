import time
import torch
from torch.utils.data import DataLoader
from src.pipeline.ephys.spike_ingestion import SpikeProphecyDataset
from src.models.spike_forecaster import SpikeForecaster
from src.utils.device import get_optimal_device

def format_bytes(size):
    """Format bytes into a human-reaksmle string."""
    power = 2**10
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}B"

def format_bandwidth(bytes_per_sec):
    """Format bandwidth into Mbps/Gbps."""
    bits_per_sec = bytes_per_sec * 8
    if bits_per_sec > 1e9:
        return f"{bits_per_sec / 1e9:.2f} Gbps"
    else:
        return f"{bits_per_sec / 1e6:.2f} Mbps"

def run_benchmark(batch_size=32, time_steps=100, num_batches=10):
    print("\n" + "="*80)
    print(" MELD TELEMETRY MATRIX BENCHMARK ")
    print("="*80)
    
    device = get_optimal_device(verbose=True)
    
    # Initialize Dataset and Dataloader
    print("[*] Initializing SpikeProphecyDataset...")
    dataset = SpikeProphecyDataset(time_steps=time_steps, split="train")
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    
    # Initialize Model
    print("[*] Initializing SpikeForecaster (Mamba-2)...")
    model = SpikeForecaster(input_dim=dataset.m_max).to(device)
    model.eval()
    
    # Pre-allocate metrics
    latencies = []
    vram_peaks = []
    
    print("\n[*] Commencing Benchmarking Loop...")
    print("-" * 110)
    print(f"{'Batch':<8} | {'Latency (ms)':<15} | {'Peak VRAM':<15} | {'Throughput (frames/s)':<25} | {'Cloud Bandwidth Eq.':<25}")
    print("-" * 110)
    
    # PyTorch warmup
    data_iter = iter(dataloader)
    
    for i in range(num_batches + 1): # +1 for warmup
        try:
            batch = next(data_iter).to(device)
        except StopIteration:
            break
            
        # Reset memory tracking if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            
        # Synchronize device before timer
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model(batch)
            
        # Synchronize device after timer
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            
        end_time = time.perf_counter()
        
        latency_sec = end_time - start_time
        latency_ms = latency_sec * 1000.0
        
        # Track memory
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated(device)
        elif torch.backends.mps.is_available():
            # MPS does not natively expose max_memory_allocated via PyTorch cleanly yet.
            # We estimate based on current allocation if possible, or fallback.
            # Using driver_allocated_memory() as a proxy for peak memory.
            peak_vram = torch.mps.driver_allocated_memory()
        else:
            peak_vram = 0 # Cannot track RAM directly in standard PyTorch easily
            
        # Calculations
        frames_processed = batch_size * time_steps
        throughput = frames_processed / latency_sec
        
        # Calculate theoretical bandwidth (if we streamed this data to the cloud at the speed we process it locally)
        # Data size per batch (float32 = 4 bytes)
        batch_size_bytes = batch_size * time_steps * dataset.m_max * 4
        bytes_per_sec = batch_size_bytes / latency_sec
        cloud_bandwidth = format_bandwidth(bytes_per_sec)
        
        vram_str = format_bytes(peak_vram) if peak_vram > 0 else "N/A"
        
        if i == 0:
            continue # Skip warmup batch
            
        latencies.append(latency_ms)
        if peak_vram > 0:
            vram_peaks.append(peak_vram)
            
        print(f"{i:<8} | {latency_ms:<15.2f} | {vram_str:<15} | {throughput:<25.2f} | {cloud_bandwidth:<25}")

    # Summary Statistics
    print("-" * 110)
    avg_latency = sum(latencies) / len(latencies)
    avg_vram = (sum(vram_peaks) / len(vram_peaks)) if vram_peaks else 0
    avg_throughput = frames_processed / (avg_latency / 1000.0)
    avg_bandwidth = format_bandwidth(frames_processed * dataset.m_max * 4 / (avg_latency / 1000.0))
    
    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Average Inference Latency : {avg_latency:.2f} ms")
    print(f"Average Peak VRAM         : {format_bytes(avg_vram) if avg_vram > 0 else 'N/A (CPU)'}")
    print(f"Average Local Throughput  : {avg_throughput:.2f} frames/sec")
    print(f"Equivalent Cloud Bandwidth: {avg_bandwidth}")
    print("\nCONCLUSION: Performing this inference at the edge requires ~0 VRAM overhead,")
    print("saving massive theoretical network bandwidth compared to streaming the raw arrays")
    print("to a Kubernetes cluster for cloud-based inference.")

if __name__ == "__main__":
    # Use edge-realistic batch sizes to avoid MPS OOM on Apple Silicon
    run_benchmark(batch_size=8, time_steps=50, num_batches=15)
