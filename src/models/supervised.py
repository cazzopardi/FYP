from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from cuml.neighbors.kneighbors_classifier import KNeighborsClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from cuml import RandomForestClassifier

# Replication of the methodology proposed by Karatas et al. in doi.org/10.1109/ACCESS
def random_forest():
    return RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=2, min_samples_leaf=1, split_criterion=0)
def ada_boost():
    return AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R')
def decision_tree():
    return DecisionTreeClassifier(splitter='best', criterion='gini', min_samples_split=2, min_samples_leaf=1)
def k_neighbours(n_jobs=-1):
    return KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski')  # if results differ, figure out how to map more hyperparameters
def gradient_bosting():
    return GradientBoostingClassifier(loss='log_loss', learning_rate=1, n_estimators=100, max_depth=3, validation_fraction=0.1)  # note: comments in SK Learn indicate log_loss is deviance
def linear_discriminant_analysis():
    return LinearDiscriminantAnalysis(solver="svd")
