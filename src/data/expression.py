"""Read selected genes from dense, sparse and backed AnnData matrices."""

def read_panel_counts(adata, genes):
    """Select ordered genes without AnnData's version-sensitive backed slicing."""
    matrix = adata.X
    if adata.isbacked and hasattr(matrix, "to_memory"):
        columns = adata.var_names.get_indexer(genes)
        if (columns < 0).any():
            missing = [gene for gene, column in zip(genes, columns) if column < 0]
            raise KeyError(f"Genes missing from expression matrix: {missing}")
        # Keep the full matrix sparse; only the selected panel is densified by
        # the caller. This bypasses private SciPy APIs used by backed slicing.
        return matrix.to_memory()[:, columns]
    return adata[:, genes].X

