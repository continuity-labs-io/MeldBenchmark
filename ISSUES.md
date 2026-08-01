# MeldBenchmark Issues & Technical Debt

This document tracks known issues, technical debt, and open TODOs within the codebase.

## 1. Technical Debt: Interpolation Method in DataLoader
**Location:** `src/pipeline/sim2real/sigma_phase_structure_dataloader.py:73`
**Description:**
Currently uses `scipy` cubic splines for "lazy interpolation." Biologically, cells don't deform like polynomials.
**Action Item:**
Upgrade this interpolation to a Latent ODE, Gaussian Process, or SDE Brownian bridge.

## 2. Incomplete Feature: Veto Proof Verification
**Location:** `src/demo/1_hssm_veto_proof.py`
**Description:**
There is a comment: `TODO: Output Expectation to verify` detailing the expected output (top panel shows raw signal, bottom panel shows flat surprise metric for first 50 frames). Currently, the script visually outputs this but does not programmatically verify/assert this condition.
**Action Item:**
Implement an automated assertion to mathematically verify the Surprise metric is flat during the initial frames despite structural wobble.

## 3. Empty Implementations: Hardware Substrate
**Location:** `src/core/substrate.py`
**Description:**
The `HardwareSubstrate` abstract base class and `CPUSubstrate` have methods containing only `pass`.
**Action Item:**
Review if `pass` should be replaced with `NotImplementedError` or if concrete no-op implementations (like `CPUSubstrate.synchronize`) should be documented as intentional.
