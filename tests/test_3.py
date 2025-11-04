import pytest
from definition_752d1808c7044019b9b764d909bf33c0 import calculate_model_metrics

@pytest.mark.parametrize(
    "true_labels, predicted_labels, prediction_scores, expected",
    [
        # 1. Standard Case: Balanced dataset with mixed predictions, non-trivial metrics.
        ([0, 1, 0, 1, 0, 1], [0, 1, 1, 1, 0, 0], [0.1, 0.9, 0.6, 0.8, 0.2, 0.3],
         {'accuracy': pytest.approx(4/6), 'precision': pytest.approx(2/3), 'recall': pytest.approx(2/3), 'f1_score': pytest.approx(2/3)}),

        # 2. Edge Case: Perfect Prediction - All labels predicted correctly.
        ([0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
         {'accuracy': pytest.approx(1.0), 'precision': pytest.approx(1.0), 'recall': pytest.approx(1.0), 'f1_score': pytest.approx(1.0)}),

        # 3. Edge Case: All Wrong Prediction - All labels predicted incorrectly.
        ([0, 1, 0, 1], [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2],
         {'accuracy': pytest.approx(0.0), 'precision': pytest.approx(0.0), 'recall': pytest.approx(0.0), 'f1_score': pytest.approx(0.0)}),

        # 4. Edge Case: No Positive Instances in True Labels or Predicted Labels.
        #    Metrics like precision, recall, and f1-score for the positive class (assuming 1)
        #    will be 0.0 when there are no true positive instances or no predicted positive instances.
        ([0, 0, 0, 0], [0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4],
         {'accuracy': pytest.approx(1.0), 'precision': pytest.approx(0.0), 'recall': pytest.approx(0.0), 'f1_score': pytest.approx(0.0)}),

        # 5. Edge Case: Invalid Input Type for 'true_labels'.
        #    Function expects array-like inputs; passing an integer should raise a TypeError.
        #    Assuming other inputs are also invalid types for consistency in this test case.
        (1, 2, 3, TypeError),
    ]
)
def test_calculate_model_metrics(true_labels, predicted_labels, prediction_scores, expected):
    if isinstance(expected, dict):
        result = calculate_model_metrics(true_labels, predicted_labels, prediction_scores)
        assert isinstance(result, dict)
        assert result.keys() == expected.keys()
        for key in expected:
            assert result[key] == expected[key]
    else: # Expected is an exception type
        with pytest.raises(expected):
            calculate_model_metrics(true_labels, predicted_labels, prediction_scores)