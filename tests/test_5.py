import pytest
import pandas as pd
from definition_ef82184e7f9f47a28bf247f5326344e5 import generate_data_card_content

@pytest.mark.parametrize(
    "data_params, dataframe, expected_output_checker, expected_exception",
    [
        # Test Case 1: Standard data with mixed types (numeric, categorical, boolean)
        (
            {"dataset_name": "Sample Data", "provenance": "Simulated", "version": "1.0"},
            pd.DataFrame({
                "num_col": [1, 2, 3, 4, 5],
                "cat_col": ["A", "B", "A", "C", "B"],
                "bool_col": [True, False, True, False, True]
            }),
            lambda result: (
                isinstance(result, dict) and
                result.get("dataset_name") == "Sample Data" and
                "feature_statistics" in result and
                isinstance(result["feature_statistics"], dict) and
                "num_col" in result["feature_statistics"] and
                "cat_col" in result["feature_statistics"] and
                "bool_col" in result["feature_statistics"]
            ),
            None,
        ),
        # Test Case 2: Empty DataFrame (with columns defined, expecting zero counts)
        (
            {"dataset_name": "Empty Data", "provenance": "Generated"},
            pd.DataFrame({"num_col": [], "cat_col": [], "date_col": pd.Series(dtype='datetime64[ns]')}),
            lambda result: (
                isinstance(result, dict) and
                result.get("dataset_name") == "Empty Data" and
                "feature_statistics" in result and
                isinstance(result["feature_statistics"], dict) and
                # Check that statistics for empty columns indicate zero count or absence
                result["feature_statistics"].get("num_col", {}).get("count") == 0 and
                result["feature_statistics"].get("cat_col", {}).get("count") == 0
            ),
            None,
        ),
        # Test Case 3: DataFrame with a single row (checking precise statistics)
        (
            {"dataset_name": "Single Row", "provenance": "Manual"},
            pd.DataFrame({"numeric_data": [10], "category_data": ["Single Value"]}),
            lambda result: (
                isinstance(result, dict) and
                result.get("dataset_name") == "Single Row" and
                "feature_statistics" in result and
                isinstance(result["feature_statistics"], dict) and
                result["feature_statistics"].get("numeric_data", {}).get("count") == 1 and
                pytest.approx(result["feature_statistics"].get("numeric_data", {}).get("mean")) == 10.0 and
                result["feature_statistics"].get("category_data", {}).get("unique_count") == 1
            ),
            None,
        ),
        # Test Case 4: DataFrame with only non-numeric columns (ensuring numeric stats are absent)
        (
            {"dataset_name": "Only Categorical", "provenance": "External"},
            pd.DataFrame({"text_col": ["apple", "banana", "orange"], "id_col": ["id1", "id2", "id3"]}),
            lambda result: (
                isinstance(result, dict) and
                result.get("dataset_name") == "Only Categorical" and
                "feature_statistics" in result and
                isinstance(result["feature_statistics"], dict) and
                "text_col" in result["feature_statistics"] and
                "id_col" in result["feature_statistics"] and
                # Ensure 'mean' statistic is not present for non-numeric columns
                result["feature_statistics"]["text_col"].get("mean") is None and
                result["feature_statistics"]["id_col"].get("mean") is None
            ),
            None,
        ),
        # Test Case 5: Invalid dataframe type (e.g., None, string, list instead of DataFrame)
        (
            {"dataset_name": "Invalid DF", "provenance": "Error Test"},
            "this is not a dataframe object", # Invalid type for dataframe
            None,
            TypeError,
        ),
    ]
)
def test_generate_data_card_content(data_params, dataframe, expected_output_checker, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            generate_data_card_content(data_params, dataframe)
    else:
        result = generate_data_card_content(data_params, dataframe)
        assert expected_output_checker(result)