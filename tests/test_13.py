import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Keep a placeholder definition_2137395c4b6741e7bd60eaf80c5f87ef for the import of the module.
# Keep the `your_module` block as it is. DO NOT REPLACE or REMOVE the block.
from definition_2137395c4b6741e7bd60eaf80c5f87ef import plot_performance_trend

@pytest.fixture
def sample_dataframe():
    """Provides a sample DataFrame with timestamp and metric columns."""
    data = {
        'timestamp_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
        'metric_col': [10, 12, 11, 15],
        'other_col': ['A', 'B', 'A', 'C']
    }
    return pd.DataFrame(data)

@pytest.fixture
def empty_dataframe():
    """Provides an empty DataFrame with expected column names."""
    return pd.DataFrame(columns=['timestamp_col', 'metric_col'])

@pytest.fixture
def dataframe_with_non_numeric_metric():
    """Provides a DataFrame where the metric column contains non-numeric data."""
    data = {
        'timestamp_col': pd.to_datetime(['2023-01-01', '2023-01-02']),
        'metric_col': ['ten', 'twelve'], # Non-numeric
    }
    return pd.DataFrame(data)

def test_plot_performance_trend_display(sample_dataframe, mocker):
    """
    Test case 1: Verifies that a plot is generated and displayed correctly when `save_path` is None.
    Mocks `seaborn.lineplot` and `matplotlib.pyplot` to check function calls.
    """
    mock_sns_lineplot = mocker.patch('definition_2137395c4b6741e7bd60eaf80c5f87ef.seaborn.lineplot')
    mock_plt = mocker.patch('definition_2137395c4b6741e7bd60eaf80c5f87ef.matplotlib.pyplot')
    
    timestamp_col = 'timestamp_col'
    metric_col = 'metric_col'
    title = 'Performance Trend'
    save_path = None

    plot_performance_trend(sample_dataframe, timestamp_col, metric_col, title, save_path)

    # Assert seaborn.lineplot was called with correct data and columns
    mock_sns_lineplot.assert_called_once()
    assert mock_sns_lineplot.call_args.kwargs['x'] == timestamp_col
    assert mock_sns_lineplot.call_args.kwargs['y'] == metric_col
    pd.testing.assert_frame_equal(
        mock_sns_lineplot.call_args.kwargs['data'], 
        sample_dataframe.sort_values(by=timestamp_col) # Function sorts data by timestamp
    )
    
    # Assert plot attributes (title, labels) are set with minimum font size
    mock_plt.title.assert_called_once_with(title, fontsize=mocker.ANY)
    assert mock_plt.title.call_args.kwargs['fontsize'] >= 12

    mock_plt.xlabel.assert_called_once_with(timestamp_col, fontsize=mocker.ANY)
    assert mock_plt.xlabel.call_args.kwargs['fontsize'] >= 12

    mock_plt.ylabel.assert_called_once_with(metric_col, fontsize=mocker.ANY)
    assert mock_plt.ylabel.call_args.kwargs['fontsize'] >= 12
    
    # Assert display and layout calls
    mock_plt.tight_layout.assert_called_once()
    mock_plt.show.assert_called_once()
    mock_plt.savefig.assert_not_called()
    mock_plt.close.assert_not_called() # Plot should not be closed if it's shown

def test_plot_performance_trend_save(sample_dataframe, mocker):
    """
    Test case 2: Verifies that a plot is generated and saved to a file when `save_path` is provided.
    Mocks `seaborn.lineplot` and `matplotlib.pyplot` to check function calls.
    """
    mock_sns_lineplot = mocker.patch('definition_2137395c4b6741e7bd60eaf80c5f87ef.seaborn.lineplot')
    mock_plt = mocker.patch('definition_2137395c4b6741e7bd60eaf80c5f87ef.matplotlib.pyplot')
    
    timestamp_col = 'timestamp_col'
    metric_col = 'metric_col'
    title = 'Performance Trend'
    save_path = 'test_plot.png'

    plot_performance_trend(sample_dataframe, timestamp_col, metric_col, title, save_path)

    # Assert seaborn.lineplot was called with correct data and columns
    mock_sns_lineplot.assert_called_once()
    assert mock_sns_lineplot.call_args.kwargs['x'] == timestamp_col
    assert mock_sns_lineplot.call_args.kwargs['y'] == metric_col
    pd.testing.assert_frame_equal(
        mock_sns_lineplot.call_args.kwargs['data'], 
        sample_dataframe.sort_values(by=timestamp_col)
    )
    
    # Assert plot attributes (title, labels) are set with minimum font size
    mock_plt.title.assert_called_once_with(title, fontsize=mocker.ANY)
    assert mock_plt.title.call_args.kwargs['fontsize'] >= 12

    mock_plt.xlabel.assert_called_once_with(timestamp_col, fontsize=mocker.ANY)
    assert mock_plt.xlabel.call_args.kwargs['fontsize'] >= 12

    mock_plt.ylabel.assert_called_once_with(metric_col, fontsize=mocker.ANY)
    assert mock_plt.ylabel.call_args.kwargs['fontsize'] >= 12
    
    # Assert save and close calls
    mock_plt.tight_layout.assert_called_once()
    mock_plt.savefig.assert_called_once_with(save_path, bbox_inches='tight')
    mock_plt.close.assert_called_once() # Plot should be closed after saving
    mock_plt.show.assert_not_called()

def test_plot_performance_trend_empty_dataframe(empty_dataframe, mocker):
    """
    Test case 3: Verifies that the function handles an empty DataFrame gracefully or raises an expected error.
    It's expected that `seaborn.lineplot` will raise a ValueError when trying to plot from an empty dataset.
    """
    mocker.patch('definition_2137395c4b6741e7bd60eaf80c5f87ef.seaborn.lineplot')
    mocker.patch('definition_2137395c4b6741e7bd60eaf80c5f87ef.matplotlib.pyplot')

    timestamp_col = 'timestamp_col'
    metric_col = 'metric_col'
    title = 'Empty Plot'
    save_path = None

    with pytest.raises(ValueError, match="cannot convert float NaN to integer|No data left to plot|cannot fill rows with NaNs"):
        plot_performance_trend(empty_dataframe, timestamp_col, metric_col, title, save_path)

@pytest.mark.parametrize("missing_col_name", ["timestamp_col", "metric_col"])
def test_plot_performance_trend_missing_column(sample_dataframe, missing_col_name):
    """
    Test case 4: Verifies that a KeyError is raised if the specified timestamp or metric column is missing.
    """
    # Create a DataFrame missing one of the target columns
    dataframe_modified = sample_dataframe.drop(columns=[missing_col_name])
    
    timestamp_col = 'timestamp_col'
    metric_col = 'metric_col'
    title = 'Plot with Missing Column'
    save_path = None

    with pytest.raises(KeyError, match=f"'{missing_col_name}'"):
        plot_performance_trend(dataframe_modified, timestamp_col, metric_col, title, save_path)

def test_plot_performance_trend_non_numeric_metric(dataframe_with_non_numeric_metric, mocker):
    """
    Test case 5: Verifies that a TypeError or ValueError is raised if the metric column is not numeric.
    Plotting libraries require numeric data for the y-axis.
    """
    mocker.patch('definition_2137395c4b6741e7bd60eaf80c5f87ef.seaborn.lineplot')
    mocker.patch('definition_2137395c4b6741e7bd60eaf80c5f87ef.matplotlib.pyplot')

    timestamp_col = 'timestamp_col'
    metric_col = 'metric_col'
    title = 'Plot with Non-Numeric Metric'
    save_path = None

    with pytest.raises(TypeError, match="unsupported operand type|must be numeric"):
        plot_performance_trend(dataframe_with_non_numeric_metric, timestamp_col, metric_col, title, save_path)