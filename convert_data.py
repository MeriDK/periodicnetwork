import joblib
from pathlib import Path
import numpy as np
import polars as pl

"""
Data conversion script for OGLE-III, MACHO, and ASAS-SN variable star datasets.
Steps to run:
1. Download the raw datasets:
    cd data
    bash download.sh
    cd ..
2. Set up the environment with required dependencies + polars
3. Preprocess and generate filtered data and split indices:
    python train.py --min_sample 50 --L 200 --filename ogle3_raw.pkl --K 8 --data-only
    python train.py --min_sample 50 --L 200 --filename macho_raw.pkl --K 8 --data-only
    python train.py --min_sample 50 --L 200 --filename asassn_raw.pkl --K 8 --data-only
4. Finally, run this script to convert the data to standardized parquet files:
    python convert_data.py
"""


root = Path('./data')
for ds in ('asassn', 'macho', 'ogle3'):
    print(ds)
    for seed in range(8):
        print(seed)
        data = joblib.load(root / f'{ds}_filtered.pkl')
        train_index, test_index = joblib.load(root / 'seeds' / f'{ds}_{seed}' / 'index.pkl')
        train_idx, _, _, _ = joblib.load(root / 'seeds' / f'{ds}_{seed}' / 'train.pkl')
        val_idx, _, _, _ = joblib.load(root / 'seeds' / f'{ds}_{seed}' / 'val.pkl')

        train_index = np.stack(train_index)
        test_index = np.stack(test_index)
        train_idx = np.stack(train_idx)
        val_idx = np.stack(val_idx)

        L = 200
        train_split = [chunk for i in train_index for chunk in data[i].split(L, L) if data[i].label is not None]
        test = [chunk for i in test_index for chunk in data[i].split(L, L) if data[i].label is not None]

        train = [train_split[i] for i in train_idx]
        val = [train_split[i] for i in val_idx]

        train_df = pl.DataFrame({
            'mjd': [el.times for el in train],
            'mag': [el.measurements for el in train],
            'magerr': [el.errors for el in train],
            'period': [el.p for el in train],
            'label': [el.label for el in train]
        })
        val_df = pl.DataFrame({
            'mjd': [el.times for el in val],
            'mag': [el.measurements for el in val],
            'magerr': [el.errors for el in val],
            'period': [el.p for el in val],
            'label': [el.label for el in val]
        })
        test_df = pl.DataFrame({
            'mjd': [el.times for el in test],
            'mag': [el.measurements for el in test],
            'magerr': [el.errors for el in test],
            'period': [el.p for el in test],
            'label': [el.label for el in test]
        })

        out_path = root / 'parquet' / ds / str(seed)
        out_path.mkdir(parents=True, exist_ok=True)
        train_df.write_parquet(out_path / 'train.parquet')
        val_df.write_parquet(out_path / 'val.parquet')
        test_df.write_parquet(out_path / 'test.parquet')
