import pytest
import pandas as pd
from definition_1d33142686b64a65af59899ff45708e8 import generate_synthetic_data

# Define expected base columns that are always added by the augmentation logic
# Based on the description: "categorical features, a timestamp column, and additional metadata"
# The exact names are implementation-dependent, but common sensible names are used here.
EXPECTED_AUGMENTED_COLUMNS = [
    'true_label',
    'timestamp',
    'categorical_feature_A', # Assuming the function adds at least one, likely more, categorical features
    'categorical_feature_B', # Using two for better coverage of "categorical features"
    'data_provenance',       # Assuming 'data_provenance' as a metadata column
    'collection_method'      # Assuming 'collection_method' as another metadata column
]

@pytest.mark.parametrize(
    "n_samples, n_features, n_classes, random_state, expected_rows, expected_features, expected_n_classes_target_range, expected_error_type",
    [
        # Test Case 1: Standard generation - covers expected functionality
        (100, 5, 2, 42, 100, 5, 2, None),
        # Test Case 2: Zero samples - edge case (should return an empty DataFrame with correct columns)
        (0, 5, 2, 42, 0, 5, 2, None),
        # Test Case 3: Single sample, single feature - edge case
        (1, 1, 2, 42, 1, 1, 2, None),
        # Test Case 4: Multiple classes - covers varying 'n_classes' parameter
        (50, 3, 3, 42, 50, 3, 3, None),
        # Test Case 5: Invalid input type for n_samples - error handling edge case
        ("invalid", 5, 2, 42, None, None, None, TypeError),
    ]
)
def test_generate_synthetic_data(
    n_samples, n_features, n_classes, random_state,
    expected_rows, expected_features, expected_n_classes_target_range, expected_error_type
):
    if expected_error_type:
        with pytest.raises(expected_error_type):
            generate_synthetic_data(n_samples, n_features, n_classes, random_state)
    else:
        df = generate_synthetic_data(n_samples, n_features, n_classes, random_state)

        # 1. Assert it returns a pandas DataFrame
        assert isinstance(df, pd.DataFrame)

        # 2. Assert the number of rows (samples)
        assert len(df) == expected_rows

        # Define expected feature columns dynamically (e.g., 'feature_0', 'feature_1', ...)
        expected_feature_columns = [f'feature_{i}' for i in range(expected_features)]
        expected_total_columns = expected_feature_columns + EXPECTED_AUGMENTED_COLUMNS

        # 3. Assert all expected columns are present
        assert set(df.columns) == set(expected_total_columns), \
            f"Expected columns {expected_total_columns}, but got {list(df.columns)}"

        # If the DataFrame is not empty, perform more detailed checks on data types and values
        if expected_rows > 0:
            # Check data types for key columns
            assert pd.api.types.is_numeric_dtype(df['true_label']), "true_label column must be numeric."
            assert pd.api.types.is_datetime64_any_dtype(df['timestamp']), "timestamp column must be datetime."

            # Check that categorical feature columns are object/string/categorical type
            assert pd.api.types.is_object_dtype(df['categorical_feature_A']) or pd.api.types.is_categorical_dtype(df['categorical_feature_A'])
            assert df['categorical_feature_A'].nunique() > 1 if len(df) > 1 else True, "Categorical feature A should ideally have multiple unique values for non-single-row data."
            assert pd.api.types.is_object_dtype(df['categorical_feature_B']) or pd.api.types.is_categorical_dtype(df['categorical_feature_B'])
            assert df['categorical_feature_B'].nunique() > 1 if len(df) > 1 else True, "Categorical feature B should ideally have multiple unique values for non-single-row data."

            # Check target label range (0 to n_classes-1)
            assert df['true_label'].min() >= 0, "true_label should be non-negative."
            assert df['true_label'].max() < expected_n_classes_target_range, \
                f"true_label max should be less than {expected_n_classes_target_range}, got {df['true_label'].max()}."

            # Check that all informative feature columns are numeric
            for col in expected_feature_columns:
                assert pd.api.types.is_numeric_dtype(df[col]), f"Feature column '{col}' must be numeric."

            # Check that metadata columns are non-empty strings/objects (no NaNs or empty strings)
            assert pd.api.types.is_object_dtype(df['data_provenance']), "data_provenance must be object/string type."
            assert not df['data_provenance'].isnull().any(), "data_provenance should not contain NaN values."
            assert not (df['data_provenance'] == '').any(), "data_provenance should not contain empty strings."

            assert pd.api.types.is_object_dtype(df['collection_method']), "collection_method must be object/string type."
            assert not df['collection_method'].isnull().any(), "collection_method should not contain NaN values."
            assert not (df['collection_method'] == '').any(), "collection_method should not contain empty strings."
