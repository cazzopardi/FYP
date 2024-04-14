from functools import partial
from multiprocessing import Pool, cpu_count, current_process
import pickle
import numpy as np
import pandas as pd
# from sklearn.svm import OneClassSVM
from thundersvm import OneClassSVM
from matplotlib import pyplot as plt

def _partition(subspace: pd.DataFrame, anomaly_rate, gpus:list[int]=[0]) -> np.ndarray:
    proc_id = current_process()._identity[0]
    # proc_id = 1
    ocsvm = OneClassSVM(nu=anomaly_rate, gpu_id=gpus[proc_id - 1])
    ocsvm.fit(subspace)  # TODO: experiment with excluding anomalies when fitting
    # name = subspace.columns[0] + subspace.columns[1]
    # pickle.dump(ocsvm, open(f'/home/calvin/FYP/{name}.pkl','wb'))
    y_pred: np.ndarray = ocsvm.predict(subspace)

    return y_pred == -1

class SSC_OCSVM:
    
    def __init__(self, gpus:list[int]=[0]) -> None:
        """
        Initialise SSC-OCSVM Model

        Parameters:
        gpus (list[int]): GPU IDs to train with
        """
        self.gpus = gpus
        self.threshold: int|None = None

    def calc_dissimilarity(self, X: pd.DataFrame, anomaly_rate: float) -> np.ndarray:
        subspaces: list[pd.DataFrame] = []
        for i, feat1 in enumerate(X.columns):
            for feat2 in X.columns[i+1:]:
                subspaces.append(X[[feat1,feat2]])
        with Pool(processes=len(self.gpus)) as p:
            f = partial(_partition, anomaly_rate=anomaly_rate, gpus=self.gpus)
            dissimilarity: np.ndarray = np.sum(p.map(f, subspaces), axis=0)
        # dissimilarity = np.empty(len(X))
        # for s in subspaces:
        #     dissimilarity += _partition(s, anomaly_rate=anomaly_rate, gpus=self.gpus)
        return dissimilarity

    def fit(self, X: pd.DataFrame, y: pd.Series):
        recall: list[float] = []
        far: list[float] = []
        precision: list[float] = []
        f1: list[float] = []
        anomalies: pd.Series = y != 0
        anomaly_rate: float = anomalies.sum()/len(y)
        dissimilarity: np.ndarray = self.calc_dissimilarity(X, anomaly_rate)
        pickle.dump(dissimilarity, open('dissim.pkl', 'wb'))
        # dissimilarity: np.ndarray = pickle.load(open('dissim.pkl', 'rb'))
        for threshold in range(0, np.max(dissimilarity)):
            preds: np.ndarray = dissimilarity > threshold
            TP = np.logical_and((preds == anomalies), anomalies).sum()
            FN = np.logical_and((preds != anomalies), anomalies).sum()
            FP = np.logical_and((preds != anomalies), preds).sum()
            TN = np.logical_and((preds == anomalies), np.logical_not(anomalies)).sum()
            recall_ = TP/(TP+FN)
            recall.append(recall_)
            precision_: float = TP/(TP+FP)
            precision.append(precision_)
            far.append(FP/(TN+FP))
            f1.append(2*precision_*recall_/(precision_+recall_))
        f = open('/home/calvin/FYP/res.txt','w')
        f.write('far: '+str(far)+'\n'+'rec: '+str(recall)+'\n'+'f1: '+str(f1))
        f.close()

        plt.plot(far, recall)
        plt.xlabel('Flase Alarm Rate')
        plt.ylabel('Detection Rate')
        plt.savefig('ROC_Pu.png')

    def predict(self, X):
        if self.threshold is None:
            raise RuntimeError("Model has not been fitted")
        return self.calc_dissimilarity(X) > self.threshold
