import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification # Used for generating synthetic data for tests
from definition_c700aa5efbb24fe7b9632fcc6edb55f3 import simulate_model_predictions

# Helper function to generate synthetic test data
def _generate_test_data(n_samples=100, n_features=5, random_state=42, as_numpy=False):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features, # All features contribute to classification
        n_redundant=0,
        n_classes=2,
        random_state=random_state
    )
    if as_numpy:
        return X, y
    else:
        features_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        labels_series = pd.Series(y, name='label')
        return features_df, labels_series

# --- Test Data Setup ---
# Standard valid data (DataFrame/Series)
features_df, labels_series = _generate_test_data()
# Standard valid data (NumPy arrays)
features_np, labels_np = _generate_test_data(as_numpy=True)

# Data for non-numeric features
non_numeric_features_df = pd.DataFrame({
    'str_feature': ['A', 'B', 'A', 'C', 'B'],
    'num_feature': [1.0, 2.0, 3.0, 4.0, 5.0]
})
non_numeric_labels_series = pd.Series([0, 1, 0, 1, 0])


@pytest.mark.parametrize(
    "features_input, labels_input, model_type, random_state, expected_shape, expected_exception",
    [
        # Test Case 1: Standard functionality with pandas DataFrame and Series
        (features_df, labels_series, 'Logistic Regression', 42, (labels_series.shape[0],), None),

        # Test Case 2: Standard functionality with numpy arrays
        (features_np, labels_np, 'Logistic Regression', 42, (labels_np.shape[0],), None),

        # Test Case 3: Invalid model_type (should raise an error)
        (features_df, labels_series, 'Unsupported Model', 42, None, ValueError),

        # Test Case 4: Empty features and labels (should raise an error during model training)
        (pd.DataFrame(), pd.Series(dtype=int), 'Logistic Regression', 42, None, ValueError),

        # Test Case 5: Features with non-numeric data (scikit-learn models typically require numeric inputs)
        (non_numeric_features_df, non_numeric_labels_series, 'Logistic Regression', 42, None, (TypeError, ValueError)),
    ]
)
def test_simulate_model_predictions(
    features_input, labels_input, model_type, random_state, expected_shape, expected_exception
):
    if expected_exception:
        with pytest.raises(expected_exception):
            simulate_model_predictions(features_input, labels_input, model_type, random_state)
    else:
        predicted_label, prediction_score = simulate_model_predictions(
            features_input, labels_input, model_type, random_state
        )

        # Assert output types
        assert isinstance(predicted_label, np.ndarray)
        assert isinstance(prediction_score, np.ndarray)

        # Assert shapes
        assert predicted_label.shape == expected_shape
        assert prediction_score.shape == expected_shape

        # Assert data types (predicted_label can be int or float, prediction_score should be float)
        assert np.issubdtype(predicted_label.dtype, np.integer) or np.issubdtype(predicted_label.dtype, np.floating)
        assert np.issubdtype(prediction_score.dtype, np.floating)

        # Assert prediction scores are within the valid range [0, 1]
        assert np.all((prediction_score >= 0) & (prediction_score <= 1))

