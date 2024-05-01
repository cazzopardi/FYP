from functools import partial
import os
from multiprocessing import Pool
import pickle

from dask import dataframe as dd
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from data.loading import load_cic_ids_2018
from preprocessing.supervised import preprocess, preprocess_features, preprocess_labels, smote, split
from models.supervised import *
from data.filtered_dataset import FilteredDataset, Level, Mode

def run_experiment(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, label:str, level: Level, mode: Mode):
    algorithms = [gradient_boosting, decision_tree, random_forest]
    for algorithm_function in algorithms:
        model = algorithm_function()
        print('Fitting ', algorithm_function.__name__, '...')
        model.fit(X_train, y_train != 0)
        print("Generating predictions...")
        y_pred = model.predict(X_test).astype('int8')
        pickle.dump(y_pred,open(f'results/Karatas{algorithm_function.__name__}{mode}{level}{label}.pkl', 'wb'))
        f = open('Karatas_results.txt', 'a')
        print(f'Results Algorithm: {algorithm_function.__name__} Mode: {mode}, Level: {level}, Label: {label}')
        print(f'Results Algorithm: {algorithm_function.__name__} Mode: {mode}, Level: {level}, Label: {label}', file=f)

        mask = np.logical_and(y_test != 0, y_pred)  # correct prediction mask
        y_pred[mask] = y_test[mask].to_numpy()  # make prediction appear correct from classification report perspective

        report = classification_report(y_test.to_numpy(), y_pred)
        # Calculate per-class accuracy
        accs = {}
        for attack_label in np.unique(y_test):
            mask = y_test == attack_label
            acc = accuracy_score(y_test[mask], y_pred[mask])
            accs[attack_label] = acc
        print(pd.Series(accs))
        print(report)
        print(pd.Series(accs), file=f)
        print(report, file=f)
        f.close()

if __name__ == '__main__':
    for level in Level:

        X, y, index = pickle.load(open(f'/mnt/d/Calvin/FYP/SMOTE/cicids2018_SMOTE_{level.value}.pkl','rb'))
        print('loaded pickle')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)  # shuffle and split
        
        # remove synthetic samples from testing data
        X_test.drop(index.intersection(X_test.index), inplace=True)
        y_test.drop(index.intersection(y_test.index), inplace=True)
        
        train: pd.DataFrame = pd.concat((X_train, pd.Series(y_train, index=X_train.index, name=level.value)), axis=1)
        print('created training set')
        del y_train
        del X_train
        variants = FilteredDataset(train, level.value, level.value)  # since each iteration only uses one level, the other parameter doesn't matter
        print('created filter object')
        if os.path.exists('done_dict.pkl'):
            done_dict = pickle.load(open('done_dict.pkl', 'rb'))
        else:
            done_dict: dict[tuple[str,Mode,Level],bool] = {}
        for mode in Mode:
            exp_class: list[tuple[pd.Index, str]] = variants.get_variants_index(level, mode)
            # with Pool(4) as p:
            #     f = partial(run_experiment, level=level, mode=mode)
            #     exp_indices, labels = exp_class
            #     params = zip()
            #     p.starmap(run_experiment, exp_class)
            
            # serial
            for exp_index, l in exp_class:
                if (l, mode, level) not in done_dict:
                    print('running experiment')
                    run_experiment(train.loc[exp_index].drop(columns=[level.value]), X_test, train.loc[exp_index,level.value], y_test, l, level, mode)
                    done_dict[l,mode,level] = True
                    pickle.dump(done_dict, open('done_dict.pkl', 'wb'))
        # X_train, X_test, y, label_encodings = pickle.load(open(TEMP_DATA, 'rb'))  # revert SMOTE as it differs between levels
    # os.remove(TEMP_DATA)
