"""Minimal numpy-only stand-in for the three scipy calls used by the eFOV pipeline.

Used only on the analysis VM, which cannot install scipy. Each function is
restricted to the exact call the pipeline makes and was checked against
scipy 1.x (see test_shim.py). Anything else raises immediately.
"""
