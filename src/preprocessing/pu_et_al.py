'''
convert protocol_type, service and flag to one-hot encoding
F-test for feature selection
normlaisation with StandardScaler (-u/variance)
split into 4 single attack cat subsets and mixed type

DoS
U2R
R2L
probe or scan
mixed

SSC-OCSVM
'''
import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def _one_hot_encode(data: pd.DataFrame, feature: str) -> pd.DataFrame:
    levels: np.ndarray =  data[feature].unique()
    for level in levels:
        data = data.assign(**{feature+'_'+str(level):data[feature] == level})

    return data

def preprocess(data: pd.DataFrame, target: str='category') -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # one-hot encoding
    non_numeric_features = ['protocol_type', 'service', 'flag']
    for non_numeric_feature in non_numeric_features:
        data = _one_hot_encode(data, non_numeric_feature)
    data = data.drop(columns=non_numeric_features)
    levels = ['probe', 'dos', 'r2l', 'u2r']

    # meta-labelling
    cat_list = pd.read_csv(os.path.join('data','NSL-KDD', 'category_list.csv'))
    cat_list = pd.Series(cat_list['Attack Category'].values, index=cat_list['Label'])
    data['category'] = data['class'].map(cat_list)

    # dataset splitting
    subsets: dict[str, pd.DataFrame] = {}
    for l in levels:
        mask: pd.Series = (data['category'] == l) | (data['category'] == 'normal')
        subsets[l] = data[mask]
    subsets['mixed'] = pd.concat([s for s in subsets.values()])
    
    # split features and labels
    X: pd.DataFrame = data.drop(columns=['class', 'category'])
    output_encoding: dict[str, int] = {lbl:enc for enc, lbl in enumerate(np.unique(data['category']))}
    y: pd.Series = data['category'].map(output_encoding)

    # feature selection
    selector = SelectKBest(score_func=f_classif, k=50)
    selector.fit(X, y)

    # Get indices of selected features
    feature_mask: np.ndarray = selector.get_support()

    # Get selected feature names
    X = X.loc[:, feature_mask]
    print(X.columns)

    # standardisation
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test
