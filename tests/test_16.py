import pytest
import pandas as pd
from definition_4568ce5052724244afec60fd698b9fd7 import compile_risk_register

def test_compile_risk_register_standard_case():
    """
    Test with a standard list of valid risk dictionaries.
    Verifies that a pandas DataFrame is returned with correct data and structure.
    """
    risk_entries = [
        {'Risk ID': 1, 'Category': 'Data Quality', 'Description': 'Data outdated', 'Impact Rating': 'High', 'Mitigation Strategy': 'Review data sources'},
        {'Risk ID': 2, 'Category': 'Security', 'Description': 'Vulnerability found', 'Impact Rating': 'Medium', 'Mitigation Strategy': 'Patch system'},
        {'Risk ID': 3, 'Category': 'Performance', 'Description': 'Slow inference', 'Impact Rating': 'Low', 'Mitigation Strategy': 'Optimize code'},
    ]
    expected_df = pd.DataFrame(risk_entries)

    result_df = compile_risk_register(risk_entries)

    assert isinstance(result_df, pd.DataFrame)
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_compile_risk_register_empty_list():
    """
    Test with an empty list of risk entries.
    Verifies that an empty pandas DataFrame (with 0 columns) is returned.
    """
    risk_entries = []
    expected_df = pd.DataFrame([]) # An empty DataFrame with 0 columns by default

    result_df = compile_risk_register(risk_entries)

    assert isinstance(result_df, pd.DataFrame)
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_compile_risk_register_duplicate_risk_id():
    """
    Test with a list containing duplicate 'Risk ID's.
    Verifies that a ValueError is raised because 'Risk ID' must be unique.
    """
    risk_entries = [
        {'Risk ID': 1, 'Category': 'Data Quality', 'Description': 'Data outdated', 'Impact Rating': 'High'},
        {'Risk ID': 2, 'Category': 'Security', 'Description': 'Vulnerability found', 'Impact Rating': 'Medium'},
        {'Risk ID': 1, 'Category': 'Compliance', 'Description': 'Regulation breach', 'Impact Rating': 'High'}, # Duplicate ID
    ]
    with pytest.raises(ValueError, match="Risk ID '1' is not unique"):
        compile_risk_register(risk_entries)

def test_compile_risk_register_missing_risk_id_in_entry():
    """
    Test with a list where one or more risk entries are missing the 'Risk ID' key.
    Verifies that a ValueError is raised as 'Risk ID' is a required field.
    """
    risk_entries = [
        {'Risk ID': 1, 'Category': 'Data Quality', 'Description': 'Data outdated', 'Impact Rating': 'High'},
        {'Category': 'Security', 'Description': 'Vulnerability found', 'Impact Rating': 'Medium'}, # Missing Risk ID
    ]
    with pytest.raises(ValueError, match="All risk entries must contain a 'Risk ID'"):
        compile_risk_register(risk_entries)

@pytest.mark.parametrize("invalid_input", [
    None,              # Not a list
    "not a list",      # Not a list
    123,               # Not a list
    {'Risk ID': 1, 'Category': 'Invalid Structure'}, # Single dictionary, not a list of dictionaries
    [1, 2, 3],         # List of non-dictionary elements
    [{'Risk ID': 1}, "not a dict"] # Mixed list with non-dictionary element
])
def test_compile_risk_register_invalid_input_type_or_content(invalid_input):
    """
    Test with various invalid input types for `risk_entries_list` or invalid content within the list.
    Verifies that either a TypeError or ValueError is raised.
    """
    with pytest.raises((TypeError, ValueError)):
        compile_risk_register(invalid_input)