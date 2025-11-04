import pytest
from definition_4f38f9835f774149a2e3ac26c9770a4f import create_risk_entry

@pytest.mark.parametrize("risk_id, category, description, impact_rating, mitigation_strategy, expected", [
    # Test Case 1: Basic valid input
    (1, 'Data Quality', 'Incomplete data records leading to model bias.', 'Medium', 'Implement data validation checks and imputation.',
     {'Risk ID': 1, 'Category': 'Data Quality', 'Description': 'Incomplete data records leading to model bias.', 'Impact Rating': 'Medium', 'Mitigation Strategy': 'Implement data validation checks and imputation.'}),

    # Test Case 2: Different valid inputs, including 'High' impact and another category
    (2, 'Algorithmic Bias', 'Model exhibits disparate performance across demographic groups.', 'High', 'Conduct fairness audits and re-train with re-balanced data.',
     {'Risk ID': 2, 'Category': 'Algorithmic Bias', 'Description': 'Model exhibits disparate performance across demographic groups.', 'Impact Rating': 'High', 'Mitigation Strategy': 'Conduct fairness audits and re-train with re-balanced data.'}),

    # Test Case 3: Edge case - empty description and mitigation strategy (still valid)
    (3, 'Transparency', '', 'Low', '',
     {'Risk ID': 3, 'Category': 'Transparency', 'Description': '', 'Impact Rating': 'Low', 'Mitigation Strategy': ''}),

    # Test Case 4: Invalid type for risk_id (expected int, got str)
    ('abc', 'Security', 'Unauthorized access to model parameters.', 'High', 'Implement access controls and encryption.', TypeError),

    # Test Case 5: Invalid type for category (expected str, got int)
    (4, 123, 'Misinterpretation of model outputs.', 'Medium', 'Provide clear documentation and training.', TypeError),
])
def test_create_risk_entry(risk_id, category, description, impact_rating, mitigation_strategy, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            create_risk_entry(risk_id, category, description, impact_rating, mitigation_strategy)
    else:
        result = create_risk_entry(risk_id, category, description, impact_rating, mitigation_strategy)
        assert result == expected