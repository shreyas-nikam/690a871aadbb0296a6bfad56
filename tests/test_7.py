import pytest
import pandas as pd
from definition_16d5843e1d0b4e1a9163b4c4555ec5c5 import aggregate_risks_by_category_and_impact

@pytest.mark.parametrize("input_df, expected_output, expected_exception", [
    # Test Case 1: Normal functionality with diverse data
    (
        pd.DataFrame([
            {'Risk ID': 1, 'Category': 'Data Quality', 'Description': 'Desc1', 'Impact Rating': 'Medium', 'Mitigation Strategy': 'Strat1'},
            {'Risk ID': 2, 'Category': 'Algorithmic Bias', 'Description': 'Desc2', 'Impact Rating': 'High', 'Mitigation Strategy': 'Strat2'},
            {'Risk ID': 3, 'Category': 'Data Quality', 'Description': 'Desc3', 'Impact Rating': 'Low', 'Mitigation Strategy': 'Strat3'},
            {'Risk ID': 4, 'Category': 'Algorithmic Bias', 'Description': 'Desc4', 'Impact Rating': 'Medium', 'Mitigation Strategy': 'Strat4'},
            {'Risk ID': 5, 'Category': 'Data Quality', 'Description': 'Desc5', 'Impact Rating': 'Medium', 'Mitigation Strategy': 'Strat5'},
        ]),
        pd.DataFrame([
            {'Category': 'Algorithmic Bias', 'Impact Rating': 'High', 'count': 1},
            {'Category': 'Algorithmic Bias', 'Impact Rating': 'Medium', 'count': 1},
            {'Category': 'Data Quality', 'Impact Rating': 'Low', 'count': 1},
            {'Category': 'Data Quality', 'Impact Rating': 'Medium', 'count': 2},
        ]).sort_values(by=['Category', 'Impact Rating']).reset_index(drop=True),
        None
    ),
    # Test Case 2: Edge Case - Empty DataFrame
    (
        pd.DataFrame(columns=['Risk ID', 'Category', 'Description', 'Impact Rating', 'Mitigation Strategy']),
        pd.DataFrame(columns=['Category', 'Impact Rating', 'count']).astype({'count': 'int64'}), # Ensure dtypes match expected for empty agg
        None
    ),
    # Test Case 3: Edge Case - Single risk entry
    (
        pd.DataFrame([
            {'Risk ID': 1, 'Category': 'Data Quality', 'Description': 'Desc1', 'Impact Rating': 'Medium', 'Mitigation Strategy': 'Strat1'},
        ]),
        pd.DataFrame([
            {'Category': 'Data Quality', 'Impact Rating': 'Medium', 'count': 1},
        ]).sort_values(by=['Category', 'Impact Rating']).reset_index(drop=True),
        None
    ),
    # Test Case 4: Error Case - Missing required column ('Category')
    (
        pd.DataFrame([
            {'Risk ID': 1, 'Description': 'Desc1', 'Impact Rating': 'Medium', 'Mitigation Strategy': 'Strat1'},
        ]),
        None,
        KeyError
    ),
    # Test Case 5: Error Case - Non-DataFrame input
    (
        "not a dataframe",
        None,
        TypeError
    ),
])
def test_aggregate_risks_by_category_and_impact(input_df, expected_output, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            aggregate_risks_by_category_and_impact(input_df)
    else:
        actual_output = aggregate_risks_by_category_and_impact(input_df)
        pd.testing.assert_frame_equal(actual_output, expected_output, check_dtype=True)