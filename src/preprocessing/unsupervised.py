import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Replication of the methodology proposed by Pu et al. in doi.org/10.26599/TST.2019.9010051

# label encoding maps
category_encoding = {'normal':0, 'DoS':1, 'probe':2, 'r2l':3, 'u2r':4}
category_decoding = {v:k for k,v in category_encoding.items()}
output_encoding = {'category': category_encoding}
output_decoding = {'category': category_decoding}

def _one_hot_encode(data: pd.DataFrame, feature: str) -> pd.DataFrame:
    levels: np.ndarray =  data[feature].unique()
    for level in levels:
        data = data.assign(**{feature+'_'+str(level):data[feature] == level})

    return data

def preprocess(data: pd.DataFrame, target: str='category') -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    # one-hot encoding
    non_numeric_features = ['protocol_type', 'service', 'flag']
    for non_numeric_feature in non_numeric_features:
        data = _one_hot_encode(data, non_numeric_feature)
    data = data.drop(columns=non_numeric_features)

    # meta-labelling
    cat_list = pd.read_csv(os.path.join('data','NSL-KDD', 'category_list.csv'))
    cat_list = pd.Series(cat_list['Attack Category'].values, index=cat_list['Label'])
    data['category'] = data['class'].map(cat_list)
    
    # split features and labels
    X: pd.DataFrame = data.drop(columns=['class', 'category'])
    # output_encoding: dict[str, int] = {lbl:enc for enc, lbl in enumerate(np.unique(data['category']))}
    y: pd.Series = data['category'].map(output_encoding[target])

    # feature selection
    selector = SelectKBest(score_func=f_classif, k=50)
    selector.fit(X, y)

    # Get indices of selected features
    feature_mask: np.ndarray = selector.get_support()

    # Get selected feature names
    X = X.loc[:, feature_mask]

    # standardisation
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), index=X.index)

    # split attack types
    subsets: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]] = {}
    for label in y.unique():
        if label == 0:
            continue
        mask: pd.Series = (y == label) | (y == 0)
        X_train, X_test, y_train, y_test = train_test_split(X[mask], y[mask], test_size=0.2, random_state=42, shuffle=True, stratify=y[mask])
        subsets[output_decoding[target][label]] = X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    subsets['mixed'] = X_train, X_test, y_train, y_test
    
    return subsets
