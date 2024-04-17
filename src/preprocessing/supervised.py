from functools import partial
import logging
import math
from multiprocessing import Pool
import random
import sys
import os
from typing import TypeVar

from sklearn.neighbors import NearestNeighbors
sys.path.append('./src')

import pandas as pd
import dask.dataframe as dd
import numpy as np
from sklearn.model_selection import train_test_split

from data.loading import load_cic_ids_2018, CIC_IDS_2018

# Replication of the methodology proposed by Karatas et al. in doi.org/10.1109/ACCESS

TARGET = 'attack category'

class SMOTE:
    label_col: str = '_label columns'
    def __init__(self, k, n_jobs:int=-1):
        self.k = k
        if n_jobs == -1:
            self.n_jobs = os.cpu_count()
        else:
            self.n_jobs = n_jobs

    def fit_resample(self, X: pd.DataFrame, y: np.ndarray, amounts: dict[int, int]) -> tuple[pd.DataFrame, np.ndarray]:
        """
        data: original dataset
        amounts: dict mapping minority class to number of samples to generate
        """
        acc: list[pd.DataFrame] = []
        for minority_class, num_samples in amounts.items():
            samples: pd.DataFrame = X[y == minority_class]
            synth: np.ndarray = self._fit(samples, math.ceil(num_samples/len(samples))*100)
            df = pd.DataFrame(synth, columns=X.columns)
            df[self.label_col] = minority_class
            acc.append(df)
        data: pd.DataFrame = pd.concat([X,pd.Series(y, name=self.label_col, index=X.index)], axis=1)
        resampled_data: pd.DataFrame = pd.concat([data, *acc], ignore_index=True).sample(frac=1, random_state=42)
        return resampled_data.drop(columns=[self.label_col]), resampled_data[self.label_col].to_numpy()

    def _fit(self, minority_samples: pd.DataFrame, N: int) -> np.ndarray:
        """
        T: Number of minority class samples
        N: Amount of SMOTE (as percentage) Likely need to be integral multiple of 100
        k: number of nearest neighbours
        """
        T: int = len(minority_samples)
        if N < 100:
            minority_samples.sample(frac=1)  # shuffle
            if N % 100 != 0:
                raise ValueError("N must be integral multiple of 100")
            T = int((N/100)*T)
            N = 100
            logging.warn("SMOTE received amount lower than 100% this may influence the shape of the resulting resampled data")
        N = int(N/100)
        # numattrs = len(data.columns)  # num attributes
        # self.newindex: int = 0 # tracks number of synthetic samples
        # synthetic: np.ndarray = np.empty((N*T, len(minority_samples.columns)))  # new samples  # TODO: verify N*T
        knn: NearestNeighbors = NearestNeighbors(n_neighbors=self.k, n_jobs=self.n_jobs)
        knn.fit(minority_samples)
        neighbours = knn.kneighbors(minority_samples, return_distance=False)
        # for i in range(T):  # TODO: parallelise
        #     nnarray: np.ndarray = neighbours[i]
        #     self.populate(nnarray, i, self.k, N, minority_samples)
        with Pool(processes=self.n_jobs) as p:
            f = partial(populate, k=self.k, N=N, minority_samples=minority_samples)
            synthetic: np.ndarray = np.vstack(p.starmap(f, zip(neighbours, range(T))))
        return synthetic

def populate(nnarray, i, k, N, minority_samples: pd.DataFrame) -> np.ndarray:
    """
    Generates a sample
    """
    synthetic: np.ndarray = np.empty((N, len(minority_samples.columns)))
    for newindex in range(N):
        nn: int = random.randint(0, k-1)
        for col_index, col in enumerate(minority_samples.columns):
            dif: float = minority_samples[col].iloc[nnarray[nn]] - minority_samples[col].iloc[i]
            gap: float = random.random()
            synthetic[newindex, col_index] = minority_samples[col].iloc[i] + gap*dif
    return synthetic

T = TypeVar('T', bound=pd.DataFrame|dd.DataFrame)
def reassign_xss(data: T) -> T:
    # Karatas et al. use a slightly differnet categorisation scheme to this study, hence, we match to replicate
    if type(data) == dd.DataFrame:
        def adjust_categories(part):
            part.loc[part['Label'] == 'Brute Force -XSS', 'attack category'] = 'Brute Force' 
            return part
        data = data.map_partitions(adjust_categories)
    else:
        data.loc[data['Label'] == 'Brute Force -XSS', 'attack category'] = 'Brute Force'
    return data

def preprocess(data: pd.DataFrame, target: str='attack category', smote_amounts:dict[str,int]|None=None) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    data = data.drop(columns=['Src IP', 'Dst IP', 'Flow ID'])  # remove string columns
    data = data.fillna(0)  # change NaN values to 0

    data_numeric = data.select_dtypes(include='number')
    max_vals = data_numeric[~np.isinf(data_numeric)].max()  # replace inf with highest value in column
    replace_dict = {col: {np.inf: max_vals[col]} for col in data_numeric.columns}
    data = data.replace(replace_dict)

    data['Date'] = data['Timestamp'].dt.strftime('%Y%m%d').astype(int)  # split datetime column
    data['Time'] = data['Timestamp'].dt.strftime('%H%M%S').astype(int)
    data = data.drop(columns=['Timestamp'])

    data['Init Fwd Win Byts Neg'] = (data['Init Fwd Win Byts'] < 0).astype('int8')  # add Negative columns
    data['Init Bwd Win Byts Neg'] = (data['Init Bwd Win Byts'] < 0).astype('int8')
    
    label_encoding = {'Benign': '0', 'Bot': '1', 'Brute Force': '2', 'DoS': '3', 'Infiltration': '4', 'Injection': '5'}
    data[target] = data[target].replace(label_encoding).astype('int8')  # encode labels

    # split into X and y
    y = data[target].values
    X = data.drop(columns=['Label', 'attack name', 'attack category'])  # remove all label columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)  # shuffle and split
    print('Applying SMOTE...')
    if smote_amounts is not None:
        smote = SMOTE(5)
        X_train, y_train = smote.fit_resample(X_train, y_train, amounts={int(label_encoding[k]):v for k, v in smote_amounts.items()})

    return X_train, X_test, y_train, y_test
