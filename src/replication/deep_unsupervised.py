import numpy as np
from sklearn.metrics import auc, roc_curve
from preprocessing.deep_unsupervised import preprocess
from models.deep_unsupervised import SAE_OCSVM

# Methodology proposed by Cao et al. in doi.org/10.1109/TCYB.2018.2838668
# Implementation retrieved from https://github.com/vanloicao/SAEDVAE/tree/master

# def display_callback(model: SAE_OCSVM):
#     recon_X = self.decoder.predict(self.encoder.predict(x_test))
#     recon_errors = np.mean(np.square(recon_X - x_test), axis=1)
#     fpr, tpr, thresholds = roc_curve(y_test, -recon_errors)
#     aucc = auc(fpr, tpr)

#     model.clf.fit(z_train)
#     y_pred = clf.decision_function(z_test)
#     FPR, TPR, thresholds = roc_curve(y_test, y_pred)
#     auc_svm = auc(FPR, TPR)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess('data/NSL-KDD/KDDTrain+.txt', 'data/NSL-KDD/KDDTest+.txt')

    model = SAE_OCSVM(0.01, X_train.shape[1], 85, 49, 12, 0.5)  # hyperparameters from original author
    model.train(X_train)
    y_pred = model.predict(X_test)
    FPR, TPR, thresholds = roc_curve(y_test, y_pred)
    auc_svm = auc(FPR, TPR)
    print(auc_svm)


# callbacks
# def AUC_AE(x_test, y_test):    
#     recon_X = self.decoder.predict(self.encoder.predict(x_test))
#     recon_errors = np.mean(np.square(recon_X - x_test), axis=1)
#     fpr, tpr, thresholds = roc_curve(y_test, -recon_errors)
#     return fpr, tpr, auc(fpr, tpr)

#Function to compute The Area Under ROC Curve
# def AUC_SVM(z_train, z_test, y_test):
#     #- Trainning SVM using Z
#     clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
#     clf.fit(z_train)
#     z_pred_test = clf.decision_function(z_test)
    
#     predictions = z_pred_test
    
#     return FPR, TPR, auc_svm