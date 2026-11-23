"""MERGE baseline (CVPR'25): hierarchical-graph GAT for ST.

Adapted from the official repository:
    https://github.com/ags3927/merge

Differences from the original:
* Their stage-1 fine-tunes a ResNet18-SimCLR on patches to produce 256-d spot
  features. We instead consume the project's precomputed spot embeddings
  (ResNet50 / Virchow2) and project them via a learnable ``Linear`` to 256-d,
  then run their stage-2 GATNet verbatim. This isolates the *graph-head*
  contribution of MERGE on the brain HMHVG protocol.
* Hierarchical edges are built once per slide on the fly (spatial + feature
  KMeans, plus 8-NN spatial graph), matching their construction.
"""
