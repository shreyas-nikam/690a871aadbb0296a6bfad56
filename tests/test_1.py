import pytest
import pandas as pd
from datetime import datetime
from definition_9f37aeb6d3ee4adea379219ced83644e import add_synthetic_time_series

@pytest.mark.parametrize("dataframe_input, start_date_input, periods_input, expected_outcome", [
    # Test Case 1: Basic functionality - empty DataFrame, positive periods
    (pd.DataFrame(), '2023-01-01', 3, pd.DataFrame({'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])})),
    # Test Case 2: Basic functionality - DataFrame with existing data, matching length
    (pd.DataFrame({'A': [10, 20], 'B': ['x', 'y']}), '2024-05-10', 2, 
     pd.DataFrame({
         'A': [10, 20], 
         'B': ['x', 'y'], 
         'timestamp': pd.to_datetime(['2024-05-10', '2024-05-11'])
     })),
    # Test Case 3: Edge case - periods = 0
    (pd.DataFrame({'A': [10, 20]}), '2024-01-01', 0, pd.DataFrame({'A': [], 'B': [], 'timestamp': []}, index=pd.Index([], dtype='object'))),
    # Test Case 4: Edge case - Invalid start_date format (expect ValueError)
    (pd.DataFrame(), 'invalid-date', 5, ValueError),
    # Test Case 5: Edge case - Non-DataFrame input for 'dataframe' (expect TypeError)
    (None, '2023-01-01', 1, TypeError),
])
def test_add_synthetic_time_series(dataframe_input, start_date_input, periods_input, expected_outcome):
    if isinstance(expected_outcome, type) and issubclass(expected_outcome, Exception):
        with pytest.raises(expected_outcome):
            add_synthetic_time_series(dataframe_input, start_date_input, periods_input)
    else:
        result_df = add_synthetic_time_series(dataframe_input, start_date_input, periods_input)
        
        # Check if the output is a DataFrame
        assert isinstance(result_df, pd.DataFrame)
        
        # Check 'timestamp' column is present and of datetime type
        assert 'timestamp' in result_df.columns
        assert pd.api.types.is_datetime64_any_dtype(result_df['timestamp'])
        
        # Check number of rows
        assert len(result_df) == len(expected_outcome)
        
        # For the periods=0 case, column order might differ and content is empty.
        # Ensure that the column names match and the data types are correct for an empty df.
        if periods_input == 0:
            assert set(result_df.columns) == set(list(dataframe_input.columns) + ['timestamp'])
            assert result_df.empty
            return

        # Check 'timestamp' column values
        pd.testing.assert_series_equal(result_df['timestamp'], expected_outcome['timestamp'], check_names=False)

        # Check other columns are preserved (if applicable)
        for col in dataframe_input.columns:
            if col != 'timestamp': # Assuming new timestamp might overwrite existing one or be added
                pd.testing.assert_series_equal(
                    result_df[col].reset_index(drop=True), 
                    expected_outcome[col].reset_index(drop=True), 
                    check_names=False, 
                    check_dtype=True # Ensure data types are consistent
                )