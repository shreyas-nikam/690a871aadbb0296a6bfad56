import pytest
import pandas as pd
from definition_d57a3a51b0244e34a64d29023525a004 import display_interactive_dataframe

@pytest.mark.parametrize(
    "dataframe_input, title_input, expected_exception",
    [
        # Test case 1: Standard DataFrame with a simple string title (Expected functionality)
        (pd.DataFrame({'col1': [1, 2, 3], 'col2': ['A', 'B', 'C']}), "Test Title", None),
        # Test case 2: Empty DataFrame (Edge case)
        (pd.DataFrame(), "Empty Data", None),
        # Test case 3: DataFrame with various data types (Robustness)
        (pd.DataFrame({
            'IntCol': [1, 2],
            'FloatCol': [1.1, 2.2],
            'StrCol': ['hello', 'world'],
            'BoolCol': [True, False],
            'DateCol': pd.to_datetime(['2023-01-01', '2023-01-02'])
        }), "Mixed Types Data", None),
        # Test case 4: Invalid 'dataframe' type (Non-DataFrame)
        ("not a dataframe", "Valid Title", TypeError),
        # Test case 5: Invalid 'title' type (Non-string)
        (pd.DataFrame({'A': [1, 2]}), 123, TypeError),
    ]
)
def test_display_interactive_dataframe(dataframe_input, title_input, expected_exception):
    """
    Tests the display_interactive_dataframe function for various DataFrame inputs and title types.
    Checks for successful execution (no exception) for valid inputs and
    correct exception raising for invalid inputs.
    """
    if expected_exception:
        # If an exception is expected, assert that it is raised
        with pytest.raises(expected_exception):
            display_interactive_dataframe(dataframe_input, title_input)
    else:
        # For valid inputs, ensure no unexpected exception is raised
        # The function's output is a display action, so we primarily check for successful execution.
        try:
            display_interactive_dataframe(dataframe_input, title_input)
        except Exception as e:
            pytest.fail(f"Function raised an unexpected exception for valid input: {e}")
