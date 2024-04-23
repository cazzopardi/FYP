import os

import pandas as pd
import dask.dataframe as dd

from data.utils import dtype_cic_ids2018, dtype_nsl_kdd


def load_cic_ids_2018(meta_labelled=True, n_workers=None) -> dd.DataFrame:
    if not meta_labelled:
        raise NotImplementedError('TODO: implement raw csv loading')

    dataset_path: str = os.path.join('data', 'CSE-CIC-IDS-2018', 'meta_labelled_data', '*.part')
    data: dd.DataFrame = dd.read_csv(dataset_path, dtype=dtype_cic_ids2018)
    data['Timestamp'] = dd.to_datetime(data['Timestamp'])
    return data    

def load_nsl_kdd() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path: str = os.path.join('data', 'NSL-KDD', 'KDDTrain+.txt')
    test_path: str = os.path.join('data', 'NSL-KDD', 'KDDTest+.txt')
    train: pd.DataFrame = pd.read_csv(train_path, names=list(dtype_nsl_kdd.keys()), dtype=dtype_nsl_kdd)
    test: pd.DataFrame = pd.read_csv(test_path, names=list(dtype_nsl_kdd.keys()), dtype=dtype_nsl_kdd)
    return train, test
