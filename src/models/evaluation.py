from sklearn.metrics import (accuracy_score,confusion_matrix, classification_report)

def get_metrics_for_evaluation(model, X_val, y_val):

    y_pred = model.predict(X_val)


    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "cm": confusion_matrix(y_val, y_pred),
        "classification_report": classification_report(y_val, y_pred)

    }