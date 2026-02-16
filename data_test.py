import numpy as np, scanpy as sc, scipy.sparse as sp
ad = sc.read_h5ad("data/hlca_minified.D20000.V1024.log1p.h5ad")  # or the raw HLCA before minify if you tap the hub directly
print("layers:", list(ad.layers.keys()))
def _lib(X):
    return np.asarray(X.sum(axis=1)).ravel() if sp.issparse(X) else X.sum(axis=1)
for k, X in [("X", ad.X),
             ("layers['counts']", ad.layers.get("counts")),
             ("layers['raw_counts']", ad.layers.get("raw_counts")),
             ("raw.X", ad.raw.X if (ad.raw is not None) else None)]:
    if X is None:
        continue
    lib = _lib(X)
    print(k, "min", lib.min(), "max", lib.max(), "pct>0", (lib>0).mean())
