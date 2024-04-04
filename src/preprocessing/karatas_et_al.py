import sys
sys.path.append('./src')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from data.loading import load_cic_ids_2018, CIC_IDS_2018

TARGET = 'attack category'

# class SMOTE:
#     def __init__(self, k):
#         self.k = k

#     def fit_resample(self, data: pd.DataFrame, amounts: dict[int, int]):
#         """
#         data: original dataset
#         amounts: dict mapping minority class to number of samples to generate
#         """
#         acc: List[pd.DataFrame] = []
#         for minority_class, num_samples in amounts.items():
#             samples: pd.DataFrame = data[data[TARGET] == minority_class]
#             synth: np.ndarray = self._fit(samples, math.ceil(num_samples/len(samples))*100)
#             acc.append(pd.DataFrame(synth, columns=data.columns))
#         resampled_data: pd.DataFrame = pd.concat([data, *acc])
#         return resampled_data.sample(frac=1)

#     def _fit(self, minority_samples: pd.DataFrame, N: int) -> np.ndarray:
#         """
#         T: Number of minority class samples
#         N: Amount of SMOTE (as percentage) Likely need to be integral multiple of 100
#         k: number of nearest neighbours
#         """
#         T: int = len(minority_samples)
#         if N < 100:
#             minority_samples.shuffle()
#             T = (N/100)*T
#             N = 100
#             logging.warn("SMOTE received amount lower than 100% this may influence the shape of the resulting resampled data")
#         N = int(N/100)
#         # numattrs = len(data.columns)  # num attributes
#         self.newindex: int = 0 # tracks number of synthetic samples
#         synthetic: np.ndarray = np.empty((N, len(minority_samples.columns)))  # new samples
#         knn: NearestNeighbors = NearestNeighbors(n_neighbors=self.k)
#         knn.fit(minority_samples)
#         for i in range(T):
#             nnarray: np.ndarray = knn.kneighbors([minority_samples.iloc[i]], return_distance=False)[0]
#             self.populate(minority_samples, synthetic, N, i, nnarray)
#         return synthetic

#     def populate(self, minority_samples, synthetic, N, i, nnarray):
#         """
#         Generates a sample
#         """
#         for _ in range(N):
#             nn: int = random.randint(0, self.k-1)
#             for col_index, col in enumerate(minority_samples.columns):
#                 dif: float = minority_samples[col].iloc[nnarray[nn]] - minority_samples[col].iloc[i]
#                 gap: float = random.random()
#                 synthetic[self.newindex, col_index] = minority_samples[col].iloc[i] + gap*dif
#             self.newindex += 1

def preprocess(data: pd.DataFrame, target: str='attack category') -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    data = data.drop(columns=['Src IP', 'Dst IP', 'Flow ID'])  # remove string columns (TODO: nah mate, if they were those authors trippin')
    data = data.fillna(0)  # change NaN values to 0

    data_numeric = data.select_dtypes(include='number')
    max_vals = data_numeric[~np.isinf(data_numeric)].max()  # replace inf with highest value in column
    replace_dict = {col: {np.inf: max_vals[col]} for col in data_numeric.columns}
    data = data.replace(replace_dict)

    data['Date'] = data['Timestamp'].dt.strftime('%Y%m%d').astype(int)  # split datetime column
    data['Time'] = data['Timestamp'].dt.strftime('%H%M%S').astype(int)
    data = data.drop(columns=['Timestamp'])

    data['Init Fwd Win Byts Neg'] = data['Init Fwd Win Byts'] < 0  # add Negative columns
    data['Init Bwd Win Byts Neg'] = data['Init Bwd Win Byts'] < 0

    # Karatas et al. use a slightly differnet categorisation scheme to this study, hence, we match to replicate
    data.loc[data['Label'] == 'Brute Force -XSS', 'attack category'] = 'Brute Force'

    label_encoding = {'Benign': '0', 'Bot': '1', 'Brute Force': '2', 'DoS': '3', 'Infiltration': '4', 'Injection': '5'}
    data[target] = data[target].replace(label_encoding).astype('int8')  # encode labels

    # split into X and y
    y = data[target].values
    X = data.drop(columns=['Label', 'attack name', 'attack category'])  # remove all label columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)  # shuffle and split

    # TODO: SMOTE

    return X_train, X_test, y_train, y_test
