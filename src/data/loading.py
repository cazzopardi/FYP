import os

import pandas as pd
import dask.dataframe as dd
from distributed import Client, LocalCluster

from data.utils import dtype_cic_ids2018, dtype_nsl_kdd


class CIC_IDS_2018:
    client: Client|None = None
    count = 0

    def __init__(self, dataset_path, n_workers=None) -> None:
        # setup cluster
        if CIC_IDS_2018.client is None:
            if n_workers is None:
                n_workers = os.cpu_count()
            cluster = LocalCluster(n_workers=n_workers)
            CIC_IDS_2018.client = Client(cluster)
        CIC_IDS_2018.count += 1

        self.data: dd.DataFrame = dd.read_csv(dataset_path, dtype=dtype_cic_ids2018)
        self.data['Timestamp'] = dd.to_datetime(self.data['Timestamp'])

    def __del__(self):
        CIC_IDS_2018.count -= 1
        if CIC_IDS_2018.count == 0:
            assert CIC_IDS_2018.client is not None
            CIC_IDS_2018.client.close()
            CIC_IDS_2018.client = None


def load_cic_ids_2018(meta_labelled=True, n_workers=None) -> CIC_IDS_2018:
    if not meta_labelled:
        raise NotImplementedError('TODO: implement raw csv loading')

    dataset_path: str = os.path.join('data', 'CSE-CIC-IDS-2018', 'meta_labelled_data', '*.part')
    return CIC_IDS_2018(dataset_path, n_workers=n_workers)    

def load_nsl_kdd() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path: str = os.path.join('data', 'NSL-KDD', 'KDDTrain+.txt')
    test_path: str = os.path.join('data', 'NSL-KDD', 'KDDTest+.txt')
    train: pd.DataFrame = pd.read_csv(train_path, names=list(dtype_nsl_kdd.keys()), dtype=dtype_nsl_kdd)
    test: pd.DataFrame = pd.read_csv(test_path, names=list(dtype_nsl_kdd.keys()), dtype=dtype_nsl_kdd)
    return train, test
