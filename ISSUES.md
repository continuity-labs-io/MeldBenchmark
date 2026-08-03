# MeldBenchmark Issues & Technical Debt

This document tracks known issues, technical debt, and open TODOs within the codebase.

## 1. Empty Implementations: Hardware Substrate
**Location:** `src/core/substrate.py`
**Description:**
The `HardwareSubstrate` abstract base class and `CPUSubstrate` have methods containing only `pass`.
**Action Item:**
Review if `pass` should be replaced with `NotImplementedError` or if concrete no-op implementations (like `CPUSubstrate.synchronize`) should be documented as intentional.
