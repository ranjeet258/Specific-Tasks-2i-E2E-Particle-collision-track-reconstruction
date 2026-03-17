import os, glob, gc
import numpy as np
import torch
import pyarrow.parquet as pq
from torch_geometric.data import Dataset
from tqdm import tqdm
from graph_utils import build_physics_graph


class JetGraphDataset(Dataset):

    def __init__(self, root, k=8, max_jets=None,
                 transform=None, pre_transform=None):
        self.k        = k
        self.max_jets = max_jets
        self._data    = []
        super().__init__(root, transform, pre_transform)
        self._load()

    @property
    def raw_file_names(self): return []

    @property
    def processed_file_names(self):
        tag = f"_cap{self.max_jets}" if self.max_jets else "_full"
        return [f"jets{tag}.pt"]

    def download(self): pass

    def process(self):
        parquet_dir   = os.path.join(self.root, "data")
        parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files in {parquet_dir}")

        data_list   = []
        total       = 0
        skipped     = 0

        for fpath in parquet_files:
            if self.max_jets and total >= self.max_jets:
                break

            print(f"\nProcessing: {os.path.basename(fpath)}")
            pf        = pq.ParquetFile(fpath)
            n_groups  = pf.metadata.num_row_groups
            print(f"  Row groups (jets): {n_groups}")

            for rg_idx in tqdm(range(n_groups), desc="  Jets"):
                if self.max_jets and total >= self.max_jets:
                    break

                # Read exactly ONE row group = ONE jet — minimal RAM
                rg    = pf.read_row_group(rg_idx, columns=["X_jets", "y"])
                x_raw = rg["X_jets"][0].as_py()   # nested Python list (3,125,125)
                label = rg["y"][0].as_py()         # float 0.0 or 1.0

                graph = build_physics_graph(x_raw, label, k=self.k)

                if graph is None:
                    skipped += 1
                else:
                    data_list.append(graph)
                    total += 1

                del rg, x_raw; 

            del pf; gc.collect()
            print(f"  Running total: {total} graphs built, {skipped} skipped")

        out = os.path.join(self.processed_dir, self.processed_file_names[0])
        torch.save(data_list, out)
        print(f"\nSaved {len(data_list)} graphs → {out}")

    def _load(self):
        pt = os.path.join(self.processed_dir, self.processed_file_names[0])
        if os.path.exists(pt):
            self._data = torch.load(pt, weights_only=False)

    def len(self):          return len(self._data)
    def get(self, idx):     return self._data[idx]


def split_dataset(dataset, train_frac=0.70, val_frac=0.15, seed=42):
    n   = len(dataset)
    idx = np.random.default_rng(seed).permutation(n)
    n_t = int(n * train_frac)
    n_v = int(n * val_frac)
    t = [dataset[i] for i in idx[:n_t]]
    v = [dataset[i] for i in idx[n_t:n_t+n_v]]
    s = [dataset[i] for i in idx[n_t+n_v:]]
    print(f"Split → train:{len(t)}  val:{len(v)}  test:{len(s)}")
    return t, v, s
