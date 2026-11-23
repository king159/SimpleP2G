"""STFlow baseline (cross-patch flow matching, ICML'25 Spotlight).

Adapted from the official repository:
    https://github.com/Graph-and-Geometric-Learning/STFlow

We port the SpatialTransformer denoiser + flow-matching interpolant to run on
the simpleP2G brain HMHVG protocol with our precomputed spot embeddings
(ResNet50 / Virchow2) and a single LOSO held-out slide (NCBI667).

Differences from the official code:
* No ``torch_geometric`` dependency — ``to_dense_batch`` is reimplemented.
* Gaussian prior only — ZINB prior (which requires ``scvi``) is dropped.
* Coords are 2-D (their default supports 2 or 3); coord dim is fixed in config.
"""
