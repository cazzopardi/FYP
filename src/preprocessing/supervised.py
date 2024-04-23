from functools import partial
import logging
import math
from multiprocessing import Pool
import random
import sys
import os
from typing import Any, TypeVar

# from sklearn.neighbors import NearestNeighbors
from cuml import NearestNeighbors
sys.path.append('./src')

import pandas as pd
import dask.dataframe as dd
import numpy as np
from sklearn.model_selection import train_test_split

# Replication of the methodology proposed by Karatas et al. in doi.org/10.1109/ACCESS
# TODO: this file needs to be refactored in two

TARGET = 'attack category'

class SMOTE:
    label_col: str = '_label columns'
    def __init__(self, k, n_jobs:int=-1):
        self.k = k
        if n_jobs == -1:
            self.n_jobs = os.cpu_count()
        else:
            self.n_jobs = n_jobs

    def fit_resample(self, X: pd.DataFrame, y: pd.Series, amounts: dict[Any, int]) -> tuple[pd.DataFrame, pd.Series]:
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
        data: pd.DataFrame = pd.concat([X,y.rename(self.label_col)], axis=1)
        resampled_data: pd.DataFrame = pd.concat([data, *acc], ignore_index=True).sample(frac=1, random_state=42)
        return resampled_data.drop(columns=[self.label_col]), resampled_data[self.label_col]

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
        knn: NearestNeighbors = NearestNeighbors(n_neighbors=self.k) # n_jobs=self.n_jobs
        knn.fit(minority_samples)
        neighbours: np.ndarray = knn.kneighbors(minority_samples, return_distance=False).values

        with Pool(processes=self.n_jobs) as p:
            f = partial(populate, k=self.k, N=N, minority_samples=minority_samples)
            synthetic: np.ndarray = np.vstack(p.starmap(f, zip(neighbours, range(T))))
        # synthetic = np.empty((T*N, len(minority_samples.columns)))
        # for item in zip(neighbours, range(T)):
        #     synth = populate(item[0], item[1], k=self.k, N=N, minority_samples=minority_samples)
        return synthetic

def populate(nnarray: np.ndarray, i: int, k: int, N: int, minority_samples: pd.DataFrame) -> np.ndarray:
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

def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.drop(columns=['Src IP', 'Dst IP', 'Flow ID'])  # remove string columns
    X = X.fillna(0)  # change NaN values to 0

    data_numeric = X.select_dtypes(include='number')
    max_vals = data_numeric[~np.isinf(data_numeric)].max() + 1  # replace inf with highest value in column +1
    replace_dict = {col: {np.inf: max_vals[col]} for col in data_numeric.columns}
    X = X.replace(replace_dict)

    X['Date'] = X['Timestamp'].dt.strftime('%Y%m%d').astype(int)  # split datetime column
    X['Time'] = X['Timestamp'].dt.strftime('%H%M%S').astype(int)
    X = X.drop(columns=['Timestamp'])

    X['Init Fwd Win Byts Neg'] = (X['Init Fwd Win Byts'] < 0).astype('int8')  # add Negative columns
    X['Init Bwd Win Byts Neg'] = (X['Init Bwd Win Byts'] < 0).astype('int8')  # TODO: this makes no sense surely they also invert

    # TODO: consider replacing the above with this
    # get_neg_col_mask: Callable[[pd.DataFrame],pd.Series] = lambda df: (df < 0).any(axis=0)
    # feature_neg: pd.Series = get_neg_col_mask(X_train)|get_neg_col_mask(X_test)  # mask indicating which columns contain large values
    
    # # create a boolean column for each column with negative values to represent the minus sign (necessary to apply logarithm later)
    # for df in [X_train, X_test]:
    #     for col in df.columns[feature_neg]: 
    #         df[col+' Neg'] = (df[col] < 0).astype('int8')
    #         df[col] = df[col].abs()

    return X


# label_encoding = {'Benign': '0', 'Bot': '1', 'Brute Force': '2', 'DoS': '3', 'Infiltration': '4', 'Injection': '5'}
def preprocess_labels(y: pd.Series, save_encoding=True) -> tuple[pd.Series, dict[str,int]]:
    labels: np.ndarray = y.unique()
    labels = labels[labels != 'Benign']
    label_encoding: dict[str,int] = {label:i+1 for i, label in enumerate(labels)}
    label_encoding['Benign'] = 0  # this is kept consistent for convenience
    if save_encoding:
        file = open('label encoding.txt', 'a')
        file.write(str(label_encoding))
    y = y.map(label_encoding).astype('int8')  # encode labels
    return y, label_encoding

def smote(X: pd.DataFrame, y: pd.Series, smote_amounts:dict[Any,int]) -> tuple[pd.DataFrame, pd.Series]:
    print('Applying SMOTE...')
    smote = SMOTE(5)
    # {k:v for k, v in smote_amounts.items()}
    return smote.fit_resample(X, y, amounts=smote_amounts)

def split(data: pd.DataFrame, target: str='attack category') -> tuple[pd.DataFrame,pd.Series]:
    y: pd.Series = data[target]
    X: pd.DataFrame = data.drop(columns=['Label', 'attack name', 'attack category'])  # remove all label columns
    return X, y

def preprocess(data: pd.DataFrame, target: str='attack category', smote_amounts:dict[str,int]|None=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = split(data, target)

    X = preprocess_features(X)
    y, label_encoding = preprocess_labels(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)  # shuffle and split
    if smote_amounts is not None:
        X_train, y_train = smote(X_train, y_train, {label_encoding[k]:v for k,v in smote_amounts.items()})

    return X_train, X_test, y_train, y_test
