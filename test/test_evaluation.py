from src.models.evaluation import get_metrics_for_evaluation
from src.models.model_pipeline_builder import build_model_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np 



def test_get_metrics_returns_expected_keys():
    # Use valid, tokenizable text
    X = pd.Series([
        "football match tonight",
        "stock market hits record high"
    ])
    y = pd.Series(["sports", "business"])

    model = build_model_pipeline() 
    model.fit(X, y)

    metrics = get_metrics_for_evaluation(model, X, y)

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "cm" in metrics
    assert "classification_report" in metrics



class DummyPerfectModel:
    """Dummy model that always predicts the true labels (perfect predictions)."""
    def __init__(self, y_to_return):
        self.y_to_return = y_to_return

    def predict(self, X):
        return self.y_to_return


def test_get_metrics_perfect_predictions():
   
    X_val = pd.Series(["text1", "text2", "text3", "text4"])
    y_val = pd.Series([0, 0, 1, 1])


    model = DummyPerfectModel(y_to_return=y_val)

    metrics = get_metrics_for_evaluation(model, X_val, y_val)

  
    assert metrics["accuracy"] == 1.0

    expected_cm = np.array([[2, 0],
                            [0, 2]])
    np.testing.assert_array_equal(metrics["cm"], expected_cm)
