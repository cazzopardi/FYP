import pickle
import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score
from preprocessing.unsupervised_dl import preprocess_nsl_kdd
from models.unsupervised_dl import SAE_OCSVM

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
    X_train, X_test, y_train, y_test = preprocess_nsl_kdd('data/NSL-KDD/KDDTrain+.txt', 'data/NSL-KDD/KDDTest+.txt')

    model = SAE_OCSVM(0.01, X_train.shape[1], 85, 49, 12, 0.5)  # hyperparameters from original author
    model.train(X_train, epochs=420, batch_size=250)
    y_pred = model.infer_similarity(X_test)
    pickle.dump(y_pred, open('results/replication/sae_pred.pkl','wb'))
    FPR, TPR, thresholds = roc_curve(y_test == 0, y_pred)
    auc_svm = auc(FPR, TPR)
    print('Total AUC:',auc_svm)
    for level in np.unique(y_test):
        if level == 0: continue
        mask = np.logical_or(y_test == level, y_test == 0)
        print(f'AUC {level}:',roc_auc_score(y_test[mask] == 0,y_pred[mask]))
# Step 420: Minibatch Loss: 0.0218 - AUC_AE 0.923 - AUC_SVM:0.958 - AUC_CEN:0.959

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