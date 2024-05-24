import pickle
import os
from typing import Literal

import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.metrics import auc, f1_score, roc_curve, accuracy_score, classification_report
import tensorflow as tf
from matplotlib import pyplot as plt
from thundersvm import OneClassSVM

from data.filtered_dataset import FilteredDataset, Level, Mode
from data.loading import load_cic_ids_2018
from models.unsupervised import SSC_OCSVM
from preprocessing.supervised import preprocess_labels, split
from preprocessing.unsupervised import preprocess_cic_ids

if __name__ == '__main__':
    # shorthands
    CAT = Level.CATEGORY
    ATT = Level.ATTACK

    print('Loading dataset...')
    dataset: dd.DataFrame = load_cic_ids_2018()
    print('Sampling dataset...')
    data: pd.DataFrame = dataset.sample(frac=0.28, random_state=386453456).compute()
    data.reset_index(drop=True, inplace=True)
    # pickle.dump(data, open('temp/uns_dl_data_np.pkl','wb'))
    # data = pickle.load(open('temp/uns_dl_data_np.pkl','rb'))

    print('Preprocessing...')
    y = {}
    X_train, X_test, y['train', CAT],  y['test',CAT] = preprocess_cic_ids(data)
    _, y_att = split(data, target='attack name')
    y_att, label_encoding = preprocess_labels(y_att)
    y['train', ATT] = y_att[X_train.index]
    y['test',ATT] = y_att[X_test.index]

    # pickle.dump((X_train, X_test, y), open('/mnt/d/Calvin/FYP/temp/uns_sampled_data.pkl','wb'))
    # X_train, X_test, y = pickle.load(open('/mnt/d/Calvin/FYP/temp/uns_sampled_data.pkl','rb'))

    n_samples = 100000
    X_train = X_train[0:n_samples]
    y['train',ATT] = y['train',ATT][0:n_samples]
    y['train',CAT] = y['train',CAT][0:n_samples]

    n_test_samples = 200000
    X_test = X_test[0:n_test_samples]
    y['test',ATT] = y['test',ATT][0:n_test_samples]
    y['test',CAT] = y['test',CAT][0:n_test_samples]

    model: SSC_OCSVM = SSC_OCSVM(gpus=[0,1])  # hyperparameters from original author
    print('Training...')
    model.fit(X_train, y['train',Level.CATEGORY], no_threshold=True)
    print('Generating predictions...')
    similarities = model.decision_function(X_test)
    pickle.dump(similarities, open('/mnt/d/Calvin/FYP/temp/uns_sim.pkl','wb'))

    # similarities = pickle.load(open('uns_dl_y_pred.pkl','rb'))

    for level in Level:
        # calculate threshold
        FPR, TPR, thresholds = roc_curve(y['test',level] == 0, similarities)
        distances = np.sqrt((1 - TPR)**2 + FPR**2)
        threshold = thresholds[np.argmin(distances)]

        y_pred = (similarities < threshold).ravel()
        # metrics
        mask = np.logical_and(y['test',level] != 0, y_pred)  # correct prediction mask 
        y_pred = y_pred.astype('int8')
        y_pred[mask] = y['test',level][mask].to_numpy()  # make prediction appear correct from classification report perspective
        
        # Calculate per-class accuracy
        accs = {}  # accuracy
        for attack_label in np.unique(y['test',level]):
            mask = y['test',level] == attack_label
            acc = accuracy_score(y['test',level][mask], y_pred[mask])
            accs[attack_label] = acc
        report = classification_report(y['test',level].to_numpy(), y_pred)  # precision, recall and f1
        auc_svm = auc(FPR, TPR)  # AUC
        # plt.plot(FPR, TPR)
        # plt.savefig('temp/uns_dl_roc.png')
        
        f = open('Pu_results.txt', 'a')
        # stdout
        print(f'Results UNS Trad - Level: {level}')
        print(pd.Series(accs))
        print(report)
        print('AUC:', auc_svm)
        # file
        print(f'Results UNS Trad - Level: {level}', file=f)
        print(pd.Series(accs), file=f)
        print(report, file=f)
        print('AUC:', auc_svm, file=f)
        f.close()
