from functools import partial
from multiprocessing import Pool, cpu_count, current_process
import pickle
from typing import Any, Generator, Tuple
import itertools
import math

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_curve
from thundersvm import OneClassSVM

# Replication of the methodology proposed by Pu et al. in doi.org/10.26599/TST.2019.9010051

def _partition(subspace: pd.DataFrame, nu: float, gpus:list[int]=[0]) -> OneClassSVM:
    proc_id = current_process()._identity[0]
    ocsvm = OneClassSVM(nu=nu, gpu_id=gpus[proc_id-1])
    ocsvm.fit(subspace)
    # name = subspace.columns[0] + subspace.columns[1]
    # pickle.dump(ocsvm, open(f'/home/calvin/FYP/{name}.pkl','wb'))
    return ocsvm

def _decision_function(args: tuple[OneClassSVM, pd.DataFrame], gpu_id:int=0, batch_size=50000) -> np.ndarray:
    # proc_id = current_process()._identity[0]
    
    # similarity: np.ndarray|None = args[0].decision_function(args[1])
    # assert similarity is not None
    # return similarity

    similarities = []
    for i in range(math.ceil(len(args[1])/batch_size)):
        start = i*batch_size
        end = min(i*batch_size+batch_size, len(args[1]))
        similarities.append(args[0].decision_function(args[1][start:end]))  # type: ignore
    return np.concatenate(similarities)

class SSC_OCSVM:
    def __init__(self, gpus:list[int]=[0]) -> None:
        """
        Initialise SSC-OCSVM Model

        Parameters:
        gpus (list[int]): GPU IDs to train with
        """
        self.gpus = gpus
        self.threshold: int|None = None
        self.anomaly_rate: float|None = None
        self.models: list[OneClassSVM]|None = None

    def ssc(self, X: pd.DataFrame) -> Generator[pd.DataFrame,None,None]:
        for i, feat1 in enumerate(X.columns):
            for feat2 in X.columns[i+1:]:
                yield X[[feat1,feat2]]

    def decision_function(self, X: pd.DataFrame, subspaces:Generator[pd.DataFrame,None,None]|None=None) -> np.ndarray:
        if self.models is None:
            raise RuntimeError('Model not fit')
        if subspaces is None:
            subspaces = self.ssc(X)
    
        # with Pool(processes=len(self.gpus)) as p:
        #     f = partial(_decision_function, gpus=self.gpus)
        #     similarity: np.ndarray = np.sum(p.map(f, zip(self.models, subspaces)), axis=0)
        similarity = np.zeros((len(X),1))
        for m, s in zip(self.models, subspaces):
            similarity = np.sum([similarity, _decision_function((m,s), gpu_id=self.gpus[0])], axis=0)
        pickle.dump(similarity, open('sim.pkl', 'wb'))
        # similiarity = pickle.load(open('dissim.pkl', 'rb'))
        return similarity

    def fit(self, X: pd.DataFrame, y: pd.Series, nu=0.1, no_threshold=False):
        subspaces: Generator[pd.DataFrame, None, None] = self.ssc(X)
        with Pool(processes=len(self.gpus)) as p:
            partition = partial(_partition, nu=nu, gpus=self.gpus)
            self.models = list(tqdm(p.imap(partition, subspaces)))
        pickle.dump(self.models, open('/mnt/d/Calvin/FYP/temp/models.pkl', 'wb'))  # debug
        # self.models = pickle.load(open('/mnt/d/Calvin/FYP/temp/models.pkl', 'rb'))  # debug
        # self.models = []
        # for (i,s) in enumerate(subspaces):
        #     self.models.append(_partition(s, nu=nu, gpus=self.gpus))
        #     if (i+1) % 50 == 0:
        #         print(f'Fitted {i+1} models')

        similarities: np.ndarray = self.decision_function(X, subspaces=subspaces)
        FPR, TPR, thresholds = roc_curve(y == 0, similarities)
        distances = np.sqrt((1 - TPR)**2 + FPR**2)
        self.threshold = thresholds[np.argmin(distances)]

    def predict(self, X, threshold=None):
        if self.threshold is None and threshold is None:
            raise RuntimeError("Model threshold not specified")
        return self.decision_function(X) > self.threshold
