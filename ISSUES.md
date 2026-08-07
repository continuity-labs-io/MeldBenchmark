# MeldBenchmark Issues & Technical Debt

This document tracks known issues, technical debt, and open TODOs within the
codebase.

## Add Neurospike data integration.

FinalSpark Neuroplatform: Standard Intan MEA, Low: 32 to 64 channels per well,
20 kHz to 30 kHz, Continuous 24/7 streaming for months. Use for Time: Perfect
for testing Mamba-2's ability to handle infinite sequence lengths and compute
thermodynamic drift over weeks, but it lacks the spatial density to show off
Mamba's capacity.

## V2 Architecture: The 129-D Quintet Tensor

We need data engineers to build actual dataloaders for single-cell epigenetic
clocks (Gamma) and in-line electrochemical sensors (Mu). Mamba-2's
data-dependent step size ($\Delta t$) will gracefully absorb the massive `NaN`
gaps of hourly epigenetic reads.

## The ATP Metabolic Checkbook Loss Constraint

We need to update `src/models/meld_loss.py`. The physics constraint is: If the
model predicts high-frequency action potentials or massive RNA transcription, it
must mathematically subtract from the global ATP reserve dimension. If the model
hallucinates a high-energy repair cascade while the ATP checkbook is empty, the
loss function must geometrically explode.
