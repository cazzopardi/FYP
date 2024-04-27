import pickle

import numpy as np
import pandas as pd
from dask import dataframe as dd 

from data.loading import load_cic_ids_2018
from data.filtered_dataset import Level
from preprocessing.supervised import split, preprocess_features, preprocess_labels, smote

if __name__ == '__main__':
    print('Loading dataset...')
    dataset: dd.DataFrame = load_cic_ids_2018(n_workers=10)

    # Sampling
    print('Sampling dataset...')
    value_counts = dataset['attack category'].value_counts().compute()
    # sampling_strategy = {'Benign': 2856035, 'Bot': 286191, 'Brute Force': 513, 'DoS': 1289544, 'Infiltration': 93063, 'Injection': 53}
    sampled_data: pd.DataFrame = dataset.sample(frac=0.28, random_state=386453456).compute()

    print('Preprocessing...')
    X, y_att = split(sampled_data, target='attack name')
    _, y_cat = split(sampled_data)
    X = preprocess_features(X)
    y = {}
    label_encodings = {}
    y[Level.CATEGORY], label_encodings[Level.CATEGORY] = preprocess_labels(y_cat)
    y[Level.ATTACK], label_encodings[Level.ATTACK] = preprocess_labels(y_att)

    # y = {}
    # X_train, X_test, y['train',Level.ATTACK], y['test',Level.ATTACK] = train_test_split(X, y_att, test_size=0.2, random_state=42, shuffle=True)  # shuffle and split
    # _, _, y['train',Level.CATEGORY], y['test',Level.CATEGORY] = train_test_split(X, y_cat, test_size=0.2, random_state=42, shuffle=True)  # shuffle and split
    # smote_amounts = {'Brute Force': 286191-513, 'Infiltration': 286191-93063, 'Injection': 286191-53}

    # TEMP_DATA = 'interim_data.pkl'
    # pickle.dump((X_train, X_test, y, label_encodings), open(TEMP_DATA, 'wb'))
    # _, X_test, y, label_encodings = pickle.load(open(TEMP_DATA, 'rb'))
    print(sampled_data['attack category'].value_counts())
    min_percentage = {Level.CATEGORY: 0.05, Level.ATTACK: 0.01}
    for level in Level:
        n = len(X)
        # vc = y[level].value_counts()
        # min_amount = min_percentage[level] * n  # set min_amount as number of samples in the smallest class above 5% of the dataset
        # # min_amount = 0.0256 * n  # note: 0.0256 is the fraction of the dataset the third most frequent class constitutes
        # smote_amounts = {label: int(min_amount - len(X[y[level] == label])) for label in np.unique(y[level]) if len(X[y[level] == label]) < min_amount}
        # sampled_X, sampled_y, index = smote(X, y[level], smote_amounts)
        # print("SMOTE complete")
        sampled_X, sampled_y, _ = pickle.load(open(f'/mnt/d/Calvin/FYP/SMOTE/cicids2018_smote_{level.value}.pkl','rb'))
        tot = len(sampled_X)
        pickle.dump((sampled_X,sampled_y,pd.RangeIndex(start=n,stop=tot,step=1)), open(f'/mnt/d/Calvin/FYP/SMOTE/cicids2018_smote_{level.value}.pkl','wb'))
