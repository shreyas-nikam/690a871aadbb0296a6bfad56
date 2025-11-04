import pytest
import pandas as pd
import numpy as np

# Placeholder for the module import
# DO NOT REPLACE or REMOVE the block below
# definition_d29fb307990544f2ba2670e82e8a5398
from definition_d29fb307990544f2ba2670e82e8a5398 import perform_data_validation
# </your_module>

# --- Test Data Setup ---

# 1. Happy Path: Valid DataFrame, critical fields present, no missing values.
df_happy = pd.DataFrame({
    'feature_1': [1, 2, 3],
    'feature_2': [4.0, 5.0, 6.0],
    'category_col': ['A', 'B', 'C'],
    'true_label': [0, 1, 0]
})
cf_happy = ['feature_1', 'feature_2', 'true_label']
expected_stdout_happy_path = [
    "Summary Statistics for Numeric Columns:",
    "feature_1", "feature_2", "true_label",  # Check for column names in stats output
    "mean", "std", "min", "max"  # Check for standard describe output content
]

# 2. Missing Values: DataFrame with NaN in critical fields.
df_missing_values = df_happy.copy()
df_missing_values.loc[0, 'feature_1'] = np.nan
df_missing_values.loc[1, 'true_label'] = np.nan
cf_missing_values = ['feature_1', 'feature_2', 'true_label']
expected_stdout_missing_values = [
    "Warning: Critical field 'feature_1' has 1 missing value(s).",
    "Warning: Critical field 'true_label' has 1 missing value(s).",
    "Summary Statistics for Numeric Columns:"
]

# 3. Missing Critical Columns: DataFrame missing some columns listed in critical_fields.
df_missing_cols = pd.DataFrame({
    'feature_1': [1, 2, 3],
    'another_col': ['X', 'Y', 'Z']
})
cf_missing_cols = ['feature_1', 'feature_2', 'true_label'] # 'feature_2' and 'true_label' are missing
expected_stdout_missing_cols = [
    "Warning: Critical fields missing from DataFrame:",  # General warning message
    "'feature_2'", "'true_label'",  # Specific missing columns
    "Summary Statistics for Numeric Columns:"  # Should still print stats for existing numeric columns
]

# 4. Empty DataFrame: An empty DataFrame input.
df_empty = pd.DataFrame()
cf_empty = ['feature_1'] # This will be reported as missing
expected_stdout_empty_df = [
    "Warning: Critical fields missing from DataFrame:",
    "'feature_1'",
    "No numeric columns found for summary statistics." # Assumes function handles no numeric columns gracefully
]

# 5. Invalid Input Type: 'dataframe' argument is not a pandas DataFrame.
df_invalid_type = "this is not a DataFrame"
cf_for_invalid_df = ['feature_1'] # A valid list for critical_fields

@pytest.mark.parametrize(
    "dataframe_input, critical_fields_input, expected_exception, expected_stdout_checks",
    [
        (df_happy, cf_happy, None, expected_stdout_happy_path),
        (df_missing_values, cf_missing_values, None, expected_stdout_missing_values),
        (df_missing_cols, cf_missing_cols, None, expected_stdout_missing_cols),
        (df_empty, cf_empty, None, expected_stdout_empty_df),
        (df_invalid_type, cf_for_invalid_df, TypeError, []), # Expecting TypeError, no stdout checks
    ]
)
def test_perform_data_validation(dataframe_input, critical_fields_input, expected_exception, expected_stdout_checks, capsys):
    if expected_exception:
        # If an exception is expected, assert that it is raised.
        with pytest.raises(expected_exception):
            perform_data_validation(dataframe_input, critical_fields_input)
    else:
        # If no exception is expected, execute the function and capture stdout.
        perform_data_validation(dataframe_input, critical_fields_input)
        captured = capsys.readouterr()
        
        # Assert that all expected substrings are present in the captured stdout.
        for substring in expected_stdout_checks:
            assert substring in captured.out, f"Expected '{substring}' not found in stdout for input: {dataframe_input}, {critical_fields_input}"
        
        # Assert that no error messages were printed to stderr (optional but good practice).
        assert not captured.err, f"Stderr was captured for input: {dataframe_input}, {critical_fields_input}: {captured.err}"