import os

import pandas as pd
import dask.dataframe as dd
from distributed import Client, LocalCluster

from data.utils import dtype


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

        self.data: dd.DataFrame = dd.read_csv(dataset_path, dtype=dtype)
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
