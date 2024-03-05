import dask.dataframe as dd

from data.utils import dtype, open_cluster, close_cluster

def load_dataset_dask(path) -> dd.DataFrame:
    # setup dask cluster
    open_cluster()

    # load dataset
    dataset: dd.DataFrame = dd.read_csv(path, dtype=dtype)
    dataset['Timestamp'] = dd.to_datetime(dataset['Timestamp'], dayfirst=True)
    
    close_cluster()
