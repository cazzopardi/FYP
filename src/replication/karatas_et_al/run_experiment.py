import pandas as pd
from sklearn.metrics import classification_report

from data.loading import CIC_IDS_2018, load_cic_ids_2018
from preprocessing.karatas_et_al import preprocess
from models.supervised import *

if __name__ == '__main__':
    print('Loading dataset...')
    dataset: CIC_IDS_2018 = load_cic_ids_2018(n_workers=10)
    data: pd.DataFrame = dataset.data.sample(frac=0.2812, random_state=42).compute()

    print('Preprocessing...')
    X_train, X_test, y_train, y_test = preprocess(data)

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
