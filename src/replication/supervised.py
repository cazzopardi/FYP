from dask import dataframe as dd
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from data.loading import CIC_IDS_2018, load_cic_ids_2018
from preprocessing.supervised import preprocess, reassign_xss
from models.supervised import *

# Replication of the methodology proposed by Karatas et al. in doi.org/10.1109/ACCESS

if __name__ == '__main__':
    print('Loading dataset...')
    dataset: CIC_IDS_2018 = load_cic_ids_2018(n_workers=10)
    
    print('Reassigning XSS category...')
    dataset.data = reassign_xss(dataset.data)

    # stratified sampling to produce data distrbution from the original study
    print('Sampling dataset...')
    value_counts = dataset.data['attack category'].value_counts().compute()
    sampling_strategy = {'Benign': 2856035, 'Bot': 286191, 'Brute Force': 513, 'DoS': 1289544, 'Infiltration': 93063, 'Injection': 53}
    samples: list[dd.DataFrame] = []
    for label, count in sampling_strategy.items():
        label_samples = dataset.data.loc[dataset.data['attack category'] == label].sample(frac=count/value_counts[label])
        samples.append(label_samples)
    
    data: pd.DataFrame = dd.concat(samples).sample(frac=1).compute()  # shuffle and compute
    print(data['attack category'].value_counts())

    print('Preprocessing...')
    X_train, X_test, y_train, y_test = preprocess(data)
    print(np.unique(y_train, return_counts=True)[1])
    print(np.unique(y_test, return_counts=True)[1])
    algorithms = [random_forest, decision_tree, ada_boost, gradient_bosting, k_neighbours, linear_discriminant_analysis]
    for algorithm_function in algorithms:
        model = algorithm_function()
        print('Fitting ', algorithm_function.__name__, '...')
        model.fit(X_train, y_train)
        print("Generating predictions...")
        y_pred = model.predict(X_test)
        print('Results of', algorithm_function.__name__, ':')
        report = classification_report(y_test, y_pred)
        print(report)
