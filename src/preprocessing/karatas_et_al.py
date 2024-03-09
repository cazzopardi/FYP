# type: ignore
import dask.dataframe as dd
import pandas as pd
import numpy as npd
from data.loading import load_cic_ids_2018

def preprocess(data: dd.DataFrame):
    data = data.fillna(0)  # cahnge NaN values to 0
    for col in  data.select_dtypes(include='number').columns:  # replace inf with highest value in column
        max_val = data[col][~np.isinf(data[col])].max()
        data[col] = data[col].replace(np.inf, max_val)

    data['Date'] = data['Timestamp'].dt.date  # split datetime column
    data['Time'] = data['Timestamp'].dt.time
    data = data.drop(columns=['Timestamp'])

    data['Init Fwd Win Byts Neg'] = data['Init Fwd Win Byts'] < 0  # add Negative columns
    data['Init Bwd Win Byts Neg'] = data['Init Bwd Win Byts'] < 0

    label_encoding = {'Benign': 0, 'Bot': 1, 'Brute Force': 2, 'DoS': 3, 'Infiltration': 4, 'SQL injection': 5}
    data['Label'] = data['Label'].replace(label_encoding)  # encode labels

    data = data.sample(frac=1)  # shuffle for randomness

if __name__ == '__main__':
    # debugging driver
    dataset = load_cic_ids_2018(na_values=0)  # change NaNs to 0
    preprocess(dataset.data)
