import os

from filtered_dataset import FilteredDataset
from load_data import load_dataset_dask

raw_dataset = load_dataset_dask(os.path.join('..', '..', 'data', 'meta_labelled_data', '*.part'))
dataset = FilteredDataset(raw_dataset, category_col_name='Label', attack_col_name='attack name')
