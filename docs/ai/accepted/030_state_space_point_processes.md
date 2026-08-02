Target File: src/pipeline/sim2real/psi_rna_dataloader.py

Context: We need to fundamentally alter how we handle sparse transcriptomic reads. The current `align_to_master_clock` method injects 12-D Poisson RNA counts on the 5-minute mark and pads the rest of the 500Hz grid with NaNs. We are moving to a Deep Continuous-Time State-Space Point Process (S2P2) architecture.

Task: Rewrite the data alignment logic to utilize a Marked Temporal Point Process instead of discrete NaN grids.
1. Create a `StateSpacePointProcess` class that treats each of the 12 Waddington RNA anchor flashes as discrete events occurring in continuous time.
2. Instead of returning a Pandas DataFrame full of NaNs, generate and return an Event Tensor of shape [Num_Events, 3], where dimension 0 is the exact continuous time of the flash (t), dimension 1 is the RNA anchor index (1 to 12), and dimension 2 is the observed transcriptomic intensity λ(t).
3. Update the Waddington crash simulation (at the 10-minute mark) to manifest as an intense compounding of the baseline hazard rate for the panic genes (TP53, IL6, CASP3), causing an explosion of asynchronous event markers rather than a single discrete spike.
4. Include basic unit tests
