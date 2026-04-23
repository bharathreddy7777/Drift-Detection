from sklearn.metrics import accuracy_score

def evaluate_batch(model, X_batch, y_batch):
    y_pred = model.predict(X_batch)
    return accuracy_score(y_batch, y_pred)
