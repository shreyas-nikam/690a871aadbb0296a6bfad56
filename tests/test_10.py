import pytest
from unittest.mock import MagicMock, patch
from definition_ec6241aa895d4274be6f3ca1a47d8799 import define_data_characteristics

# Define the expected keys for the output dictionary
EXPECTED_KEYS = [
    'dataset_name',
    'n_samples',
    'n_features',
    'n_categorical_features',
    'data_provenance',
    'collection_method',
    'identified_biases_description',
    'privacy_notes'
]

# Test Case 1: All valid and typical inputs
@patch('ipywidgets.IntText')
@patch('ipywidgets.Textarea')
@patch('ipywidgets.Text')
@patch('IPython.display.display')
def test_define_data_characteristics_typical_inputs(mock_display, mock_text, mock_textarea, mock_inttext):
    # Setup mock widget values
    mock_text.return_value.value = "Synthetic Customer Data"
    mock_inttext.side_effect = [
        MagicMock(value=1000), # n_samples
        MagicMock(value=5),    # n_features
        MagicMock(value=2)     # n_categorical_features
    ]
    mock_textarea.side_effect = [
        MagicMock(value="Internal data generation script"), # data_provenance
        MagicMock(value="Simulated user interactions"),      # collection_method
        MagicMock(value="Potential overrepresentation of male users."), # identified_biases_description
        MagicMock(value="Anonymized; aggregated data."),     # privacy_notes
    ]

    result = define_data_characteristics()

    # Assert that the output is a dictionary
    assert isinstance(result, dict)
    # Assert all expected keys are present
    assert all(key in result for key in EXPECTED_KEYS)
    # Assert values match the mocked inputs
    assert result['dataset_name'] == "Synthetic Customer Data"
    assert result['n_samples'] == 1000
    assert result['n_features'] == 5
    assert result['n_categorical_features'] == 2
    assert result['data_provenance'] == "Internal data generation script"
    assert result['collection_method'] == "Simulated user interactions"
    assert result['identified_biases_description'] == "Potential overrepresentation of male users."
    assert result['privacy_notes'] == "Anonymized; aggregated data."
    # Assert widgets were created and displayed
    assert mock_text.called
    assert mock_inttext.called
    assert mock_textarea.called
    assert mock_display.called

# Test Case 2: Minimal numeric inputs and empty text fields
@patch('ipywidgets.IntText')
@patch('ipywidgets.Textarea')
@patch('ipywidgets.Text')
@patch('IPython.display.display')
def test_define_data_characteristics_minimal_inputs(mock_display, mock_text, mock_textarea, mock_inttext):
    # Setup mock widget values for minimal case
    mock_text.return_value.value = ""
    mock_inttext.side_effect = [
        MagicMock(value=0), # n_samples (boundary)
        MagicMock(value=1), # n_features (minimum positive)
        MagicMock(value=0)  # n_categorical_features (boundary)
    ]
    mock_textarea.side_effect = [
        MagicMock(value=""), # data_provenance
        MagicMock(value=""), # collection_method
        MagicMock(value=""), # identified_biases_description
        MagicMock(value=""), # privacy_notes
    ]

    result = define_data_characteristics()

    assert isinstance(result, dict)
    assert all(key in result for key in EXPECTED_KEYS)
    assert result['dataset_name'] == ""
    assert result['n_samples'] == 0
    assert result['n_features'] == 1
    assert result['n_categorical_features'] == 0
    assert result['data_provenance'] == ""
    assert result['collection_method'] == ""
    assert result['identified_biases_description'] == ""
    assert result['privacy_notes'] == ""
    assert mock_text.called
    assert mock_inttext.called
    assert mock_textarea.called
    assert mock_display.called

# Test Case 3: Mixed inputs - some values, some defaults/empty
@patch('ipywidgets.IntText')
@patch('ipywidgets.Textarea')
@patch('ipywidgets.Text')
@patch('IPython.display.display')
def test_define_data_characteristics_mixed_inputs(mock_display, mock_text, mock_textarea, mock_inttext):
    mock_text.return_value.value = "Experiment Data"
    mock_inttext.side_effect = [
        MagicMock(value=50), # n_samples
        MagicMock(value=10), # n_features
        MagicMock(value=3)   # n_categorical_features
    ]
    mock_textarea.side_effect = [
        MagicMock(value="Open-source Kaggle dataset"),
        MagicMock(value=""), # Empty collection_method
        MagicMock(value="No specific biases identified yet."),
        MagicMock(value=""), # Empty privacy_notes
    ]

    result = define_data_characteristics()

    assert isinstance(result, dict)
    assert result['dataset_name'] == "Experiment Data"
    assert result['n_samples'] == 50
    assert result['collection_method'] == "" # Check the empty one
    assert result['privacy_notes'] == "" # Check another empty one
    assert mock_display.called

# Test Case 4: Verify structure and types of output
@patch('ipywidgets.IntText')
@patch('ipywidgets.Textarea')
@patch('ipywidgets.Text')
@patch('IPython.display.display')
def test_define_data_characteristics_output_structure_and_types(mock_display, mock_text, mock_textarea, mock_inttext):
    # Provide some default values to ensure output is not all empty
    mock_text.return_value.value = "Default Dataset"
    mock_inttext.side_effect = [
        MagicMock(value=100),
        MagicMock(value=3),
        MagicMock(value=1)
    ]
    mock_textarea.side_effect = [
        MagicMock(value="Default provenance"),
        MagicMock(value="Default collection"),
        MagicMock(value="Default biases"),
        MagicMock(value="Default privacy"),
    ]

    result = define_data_characteristics()

    assert isinstance(result, dict)
    assert len(result) == len(EXPECTED_KEYS)
    for key in EXPECTED_KEYS:
        assert key in result
    assert isinstance(result['dataset_name'], str)
    assert isinstance(result['n_samples'], int)
    assert isinstance(result['n_features'], int)
    assert isinstance(result['n_categorical_features'], int)
    assert isinstance(result['data_provenance'], str)
    assert isinstance(result['collection_method'], str)
    assert isinstance(result['identified_biases_description'], str)
    assert isinstance(result['privacy_notes'], str)
    assert mock_display.called

# Test Case 5: Edge numeric inputs - negative and very large
@patch('ipywidgets.IntText')
@patch('ipywidgets.Textarea')
@patch('ipywidgets.Text')
@patch('IPython.display.display')
def test_define_data_characteristics_edge_numeric_inputs(mock_display, mock_text, mock_textarea, mock_inttext):
    mock_text.return_value.value = "Edge Case Data"
    mock_inttext.side_effect = [
        MagicMock(value=-10),  # n_samples (negative, might be constrained by real widget)
        MagicMock(value=-1),   # n_features (negative, might be constrained)
        MagicMock(value=1000000) # n_categorical_features (very large)
    ]
    mock_textarea.side_effect = [
        MagicMock(value=""), # Empty
        MagicMock(value="Automated generation"),
        MagicMock(value="No biases known yet."),
        MagicMock(value="Confidential"),
    ]

    result = define_data_characteristics()

    assert isinstance(result, dict)
    assert result['dataset_name'] == "Edge Case Data"
    assert result['n_samples'] == -10
    assert result['n_features'] == -1
    assert result['n_categorical_features'] == 1000000
    assert mock_display.called
