from sklearn.metrics import accuracy_score, precision_score, recall_score

def print_scores(y_test, y_pred):
    # Evaluate the model
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy:', acc)
    precision = precision_score(y_test, y_pred, average='micro')
    print('Precision:', precision)
    recall = recall_score(y_test, y_pred, average='micro')
    print('Recall:', recall)
