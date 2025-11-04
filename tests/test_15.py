import pytest
from definition_73860cafe90e4668901e423ce77869b8 import add_risk_to_register

# Define a helper function to create a risk dictionary with a given ID
# This assumes the function being tested will generate the ID based on the current length
def _create_risk_dict(risk_id, category, description, impact, mitigation):
    return {
        'Risk ID': risk_id,
        'Category': category,
        'Description': description,
        'Impact Rating': impact,
        'Mitigation Strategy': mitigation
    }

@pytest.mark.parametrize(
    "initial_risks_state, category, description, impact, mitigation, expected_result",
    [
        # Test Case 1: Add a single risk to an empty list.
        # Expected: List with one risk, ID = 1.
        ([],
         "Data Quality", "Synthetic data contains outliers.", "Medium", "Implement robust data cleansing.",
         [_create_risk_dict(1, "Data Quality", "Synthetic data contains outliers.", "Medium", "Implement robust data cleansing.")]),

        # Test Case 2: Add a risk to a list with existing risks.
        # Expected: List with the old risk (ID 1) and new risk (ID 2).
        ([_create_risk_dict(1, "Algorithmic Bias", "Model shows bias.", "High", "Retrain with balanced data.")],
         "Hallucination", "Model generates nonsensical output.", "High", "Implement guardrails.",
         [
             _create_risk_dict(1, "Algorithmic Bias", "Model shows bias.", "High", "Retrain with balanced data."),
             _create_risk_dict(2, "Hallucination", "Model generates nonsensical output.", "High", "Implement guardrails.")
         ]),

        # Test Case 3: Add a risk with empty description and mitigation (edge case).
        # Expected: List with one risk, ID = 1, with empty strings for description/mitigation.
        ([],
         "Governance", "", "Low", "",
         [_create_risk_dict(1, "Governance", "", "Low", "")]),

        # Test Case 4: `current_risks` argument is not a list (error case).
        # Expected: TypeError.
        ("not a list", "Category", "Description", "Impact", "Mitigation", TypeError),

        # Test Case 5: One of the string arguments (e.g., category) is of an incorrect type (error case).
        # Expected: TypeError.
        ([], 123, "Description", "Impact", "Mitigation", TypeError),
    ]
)
def test_add_risk_to_register(initial_risks_state, category, description, impact, mitigation, expected_result):
    if isinstance(expected_result, type) and issubclass(expected_result, Exception):
        # Test for expected exceptions (e.g., TypeError)
        with pytest.raises(expected_result):
            # If initial_risks_state is not a list, pass it directly; otherwise, make a copy if it's a list.
            risks_for_test = initial_risks_state if not isinstance(initial_risks_state, list) else list(initial_risks_state)
            add_risk_to_register(risks_for_test, category, description, impact, mitigation)
    else:
        # Test for successful appending and content verification
        # Always make a copy of the initial state to ensure the test is isolated and doesn't modify the parametrize data
        risks_list = list(initial_risks_state)
        add_risk_to_register(risks_list, category, description, impact, mitigation)
        assert risks_list == expected_result