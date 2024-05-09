import pickle

from dask import dataframe as dd
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from data.loading import load_cic_ids_2018
from preprocessing.supervised import preprocess, reassign_xss
from models.supervised import *

# Replication of the methodology proposed by Karatas et al. in doi.org/10.1109/ACCESS

if __name__ == '__main__':
    print('Loading dataset...')
    dataset: dd.DataFrame = load_cic_ids_2018(n_workers=10)
    
    print('Reassigning XSS category...')
    dataset.data = reassign_xss(dataset)

    # stratified sampling to produce data distrbution from the original study
    print('Sampling dataset...')
    value_counts = dataset['attack category'].value_counts().compute()
    sampling_strategy = {'Benign': 2856035, 'Bot': 286191, 'Brute Force': 513, 'DoS': 1289544, 'Infiltration': 93063, 'Injection': 53}
    samples: list[dd.DataFrame] = []
    for label, count in sampling_strategy.items():
        label_samples = dataset.loc[dataset['attack category'] == label].sample(frac=count/value_counts[label], random_state=4325452)
        samples.append(label_samples)
    
    data: pd.DataFrame = dd.concat(samples).sample(frac=1, random_state=4325452).compute()  # shuffle and compute
    print(data['attack category'].value_counts())

    print('Preprocessing...')
    X_train, X_test, y_train, y_test = preprocess(data, smote_amounts={'Brute Force': 286191-513, 'Infiltration': 286191-93063, 'Injection': 286191-53}, keep_timestamp=True)
    pkl = (X_train, X_test, y_train, y_test)
    pickle.dump(pkl, open('temp/smoted_replication.pkl', 'wb'))
    # X_train, X_test, y_train, y_test = pickle.load(open('temp/smoted_replication.pkl', 'rb'))
    algorithms = [gradient_boosting, decision_tree, random_forest, k_neighbours, linear_discriminant_analysis]
    # algorithms = [gradient_boosting]
    for algorithm_function in algorithms:
        model = algorithm_function()
        print('Fitting ', algorithm_function.__name__, '...')
        model.fit(X_train, y_train.to_numpy())
        print("Generating predictions...")
        y_pred = model.predict(X_test)
        pickle.dump(y_pred, open(f'results/replication/{algorithm_function.__name__}_pred.pkl', 'wb'))
        print('Results of', algorithm_function.__name__, ':')
        report = classification_report(y_test.to_numpy(), y_pred)
        # Calculate per-class accuracy
        accs = {}
        for level in np.unique(y_test):
            mask = y_test == level
            acc = accuracy_score(y_test[mask], y_pred[mask])
            accs[level] = acc
        print(pd.Series(accs))
        print(report)
