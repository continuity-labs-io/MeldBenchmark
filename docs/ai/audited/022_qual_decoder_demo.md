Create a rigorous benchmarking and training script `5_qualia_decoder_demo.py` in `src/demo`.
Context: This ties the EMI (Extracellular-Membrane-Intracellular) decoding pipeline together, tracking exactly when the continuous wave phase-locks into visual consensus (The ~300ms Ignition).

Requirements:
1. Setup:
   - Import `ContinuousLFPDataset`, `QualiaDecoder`, `QualiaContrastiveLoss` (from `src.models.meld_loss`), and `ThermodynamicMetrics` (from `src.metrics.thermodynamics`).
   - Initialize the dataloader (batch_size=8), model, and loss on the optimal device using `src.utils.device.get_optimal_device()`.
2. Training Loop (Simulated):
   - Run a 50-iteration training loop to optimize the `QualiaDecoder` using the `QualiaContrastiveLoss` (AdamW optimizer, lr=1e-3).
3. Inference & The Physics Proof:
   - After training, pass a single continuous LFP sequence `[1, 500, 2, 64, 64]` through the model with `return_hidden=True`. Extract the full sequence of Mamba hidden states.
   - Instantiate `ThermodynamicMetrics(alpha=500.0)`. Run `calculate_dab` over the hidden states to map the eigenvalue divergence over the 500ms window.
4. Visualization:
   - Generate a 2-panel Matplotlib dashboard (save to `output/4_qualia_decoder_proof.png`). Use the dark-background aesthetic from `0_concat_demo.py`.
   - Top Panel: Plot the Contrastive Loss Convergence over the 50 training iterations to prove the geometric alignment is being learned.
   - Bottom Panel: Plot the DAB metric over the 500ms temporal sequence. Add a vertical dashed line at the 300ms mark to highlight the "Ignition Phase Transition" (the exact moment the eigenvalues converge to 1.0, indicating the standing wave has locked into the visual consensus).


