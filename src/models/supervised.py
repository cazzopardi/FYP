import os

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Replication of the methodology proposed by Karatas et al. in doi.org/10.1109/ACCESS
def random_forest(n_jobs=os.cpu_count()):
    # return RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=2, min_samples_leaf=1, split_criterion=0)
    return RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=2, min_samples_leaf=1, criterion='gini', n_jobs=n_jobs)
def ada_boost():
    return AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R')
def decision_tree():
    return DecisionTreeClassifier(splitter='best', criterion='gini', min_samples_split=2, min_samples_leaf=1)
def k_neighbours():
    return KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski', n_jobs=os.cpu_count())
def gradient_boosting():
    # return GradientBoostingClassifier(loss='log_loss', learning_rate=1, n_estimators=100, max_depth=3, validation_fraction=0.1)  # note: comments in SK Learn indicate log_loss is deviance
    return XGBClassifier(loss='log_loss', learning_rate=1, n_estimators=100, max_depth=3, validation_fraction=0.1, device='cuda')
def linear_discriminant_analysis():
    return LinearDiscriminantAnalysis(solver="svd")
