import pytest
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- definition_e3958d6ad76f4e3482f31229adef2418 block START ---
# Placeholder for the actual module import.
# When running tests, replace 'definition_e3958d6ad76f4e3482f31229adef2418' with the actual module name (e.g., 'my_model_utils').
from definition_e3958d6ad76f4e3482f31229adef2418 import generate_model_card_content
# --- definition_e3958d6ad76f4e3482f31229adef2418 block END ---

# Helper function to calculate metrics for expected values, mimicking the internal logic
# This is used in the test file to define the 'expected' metric values, not the actual function under test.
def _calculate_metrics_for_test(true_labels, predicted_labels, prediction_scores):
    if not true_labels:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.0
        }

    # Ensure inputs are numpy arrays for sklearn functions
    true_labels_np = np.array(true_labels, dtype=int)
    predicted_labels_np = np.array(predicted_labels, dtype=int)
    prediction_scores_np = np.array(prediction_scores, dtype=float)

    metrics = {
        "accuracy": accuracy_score(true_labels_np, predicted_labels_np),
        "precision": precision_score(true_labels_np, predicted_labels_np, zero_division=0, pos_label=1),
        "recall": recall_score(true_labels_np, predicted_labels_np, zero_division=0, pos_label=1),
        "f1_score": f1_score(true_labels_np, predicted_labels_np, zero_division=0, pos_label=1),
    }

    # ROC AUC requires probability scores and at least two classes in true labels
    if len(np.unique(true_labels_np)) > 1:
        metrics["roc_auc"] = roc_auc_score(true_labels_np, prediction_scores_np)
    else:
        metrics["roc_auc"] = 0.0 # Or np.nan, depending on specific requirements for single-class ROC AUC

    return metrics

# Test cases for generate_model_card_content
# Each tuple contains: (model_params, true_labels, predicted_labels, prediction_scores, expected_output_or_exception)
test_cases = [
    # Test Case 1: Happy Path - Standard operation with valid data
    (
        {'model_name': 'ChurnPredictor', 'purpose': 'Identify customers at risk of churn', 'version': '1.0'},
        [0, 1, 0, 1, 0, 1],
        [0, 1, 1, 1, 0, 0], # One FP (0->1), one FN (1->0), one correct (1->1), three correct (0->0)
        [0.1, 0.9, 0.6, 0.8, 0.2, 0.3],
        lambda p, t, pr, s: {**p, 'performance_metrics': _calculate_metrics_for_test(t, pr, s)}
    ),
    # Test Case 2: Edge Case - Empty input arrays for labels/scores
    (
        {'model_name': 'EmptyDataModel', 'purpose': 'Demonstration with no data'},
        [], [], [],
        lambda p, t, pr, s: {**p, 'performance_metrics': _calculate_metrics_for_test(t, pr, s)}
    ),
    # Test Case 3: Error Case - Mismatched length of label arrays
    # A robust implementation should raise ValueError if input arrays do not have same length
    (
        {'model_name': 'MismatchedLengthModel'},
        [0, 1, 0],
        [0, 1],
        [0.5, 0.6, 0.7],
        ValueError
    ),
    # Test Case 4: Error Case - Invalid type for model_params (e.g., a list instead of a dict)
    (
        ['model_name', 'InvalidParams'], # model_params as a list
        [0, 1],
        [0, 1],
        [0.5, 0.6],
        TypeError
    ),
    # Test Case 5: Error Case - Non-numeric prediction scores
    # This should raise a ValueError when attempting to convert to float array or during metric calculation
    (
        {'model_name': 'NonNumericScoreModel'},
        [0, 1],
        [0, 1],
        [0.5, 'invalid_score'], # Contains a non-numeric string
        ValueError
    ),
]

@pytest.mark.parametrize(
    "model_params, true_labels, predicted_labels, prediction_scores, expected",
    test_cases
)
def test_generate_model_card_content(model_params, true_labels, predicted_labels, prediction_scores, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        # Expecting an exception
        with pytest.raises(expected):
            generate_model_card_content(model_params, true_labels, predicted_labels, prediction_scores)
    else:
        # Happy path: Function should return a dictionary matching the expected structure and values
        expected_output_dict = expected(model_params, true_labels, predicted_labels, prediction_scores)
        actual_output = generate_model_card_content(model_params, true_labels, predicted_labels, prediction_scores)
        
        # Check that all top-level keys are present in the actual output
        assert actual_output.keys() == expected_output_dict.keys()

        # Check that model_params are correctly included
        for k, v in model_params.items():
            assert actual_output[k] == v
        
        # Check performance_metrics separately, using pytest.approx for float comparisons
        actual_metrics = actual_output.get('performance_metrics', {})
        expected_metrics = expected_output_dict.get('performance_metrics', {})

        assert actual_metrics.keys() == expected_metrics.keys()
        for metric_name, expected_value in expected_metrics.items():
            assert actual_metrics[metric_name] == pytest.approx(expected_value, abs=1e-6)