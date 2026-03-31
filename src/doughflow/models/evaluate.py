import pandas as pd
import numpy as np 
from typing import Dict, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless/terminal use
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path 

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Parameters:
    -----------
    y_true: np.ndarray
        True values (actual quantities)
    y_pred: np.ndarray
        Predicted values

    Returns:
    --------
    metrics: dict
        Dictionary with metric names and values

    Example:
    --------
    >>> y_true = np.array([50, 55, 60, 52])
    >>> y_pred = np.array([48, 57, 58, 54])
    >>> metrics = calculate_metrics(y_true, y_pred)
    >>> print(metrics)
    {'mae': 2.0, 'rmse': 2.24, 'mape': 3.85, 'r2': 0.75}
    """

    # Mean absolute error
    mae = np.mean(np.abs(y_true - y_pred))

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Mean Absolute Percentage Error (handle zeros)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))  # Avoid division by zero
    
    return {
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'mape': round(mape, 2),
        'r2': round(r2, 4)
    }


def evaluate_by_item(
        y_true: pd.Series,
        y_pred: np.ndarray,
        item_labels: pd.Series
) -> Dict[str, Dict[str,float]]:
    
    """
    Evaluate metrics separately for each item (Danish vs Croissant).
    
    This is important because:
    - Danish has different scale (mean ~30)
    - Croissant has different scale (mean ~54)
    - MAE=5 might be good for Croissant but bad for Danish
    
    Parameters:
    -----------
    y_true : pd.Series
        True values
    y_pred : np.ndarray
        Predicted values
    item_labels : pd.Series
        Item names ('Danish' or 'Croissant') for each sample
        
    Returns:
    --------
    metrics_by_item : dict
        Nested dictionary: {item_name: {metric_name: value}}
        
    Example:
    --------
    >>> metrics = evaluate_by_item(y_true, y_pred, items)
    >>> print(metrics)
    {
        'Croissant': {'mae': 5.2, 'rmse': 7.1, 'mape': 9.6, 'r2': 0.85},
        'Danish': {'mae': 3.1, 'rmse': 4.2, 'mape': 10.4, 'r2': 0.82}
    }
    
    """

    metrics_by_item = {}

    # Get unique items
    unique_items = item_labels.unique()

    for item in unique_items:
        # Filter to just this item 
        mask = item_labels == item
        y_true_item = y_true[mask].values
        y_pred_item = y_pred[mask]

        # calculate metrics for this item 
        metrics_by_item[item] = calculate_metrics(y_true_item, y_pred_item)

    return metrics_by_item

def plot_predictions(
        y_true: pd.Series,
    y_pred: np.ndarray,
    dates: pd.Series,
    item_labels: pd.Series,
    save_path: str = None
) -> None:
    """
    Plot actual vs predicted values over time, separated by item.
    
    Creates a 2-subplot figure:
    - Top: Croissant predictions vs actual
    - Bottom: Danish predictions vs actual
    
    Parameters:
    -----------
    y_true : pd.Series
        True values
    y_pred : np.ndarray
        Predicted values
    dates : pd.Series
        Dates for x-axis
    item_labels : pd.Series
        Item names for each sample
    save_path : str, optional
        If provided, save plot to this path
        
    Example:
    --------
    >>> plot_predictions(y_test, predictions, test_dates, test_items, 
    ...                  save_path='reports/figures/predictions.png')

    """

    # Create ficgure with 2 subplot ( one per item)
    fig, axes = plt.subplots(2,1, figsize = (14,10))

    items = ['Croissant', 'Danish']

    for idx, item in enumerate(items):
        ax = axes[idx]

        # Filter to this item 
        mask = item_labels == item
        item_dates = dates[mask]
        item_true = y_true[mask]
        item_pred = y_pred[mask]

        # Plot actual values
        ax.plot(item_dates, item_true,
                label = 'Actual', marker = 'o', linestyle ='-',
                linewidth=2, markersize=4, alpha = 0.7)
        
        # Plot predicted values
        ax.plot(item_dates, item_pred,
                label='Predicted', marker ='s', linestyle = '--',
                linewidth=2, markersize = 4, alpha = 0.7)
        
        # calculate metrics for title 
        metrics = calculate_metrics(item_true.values, item_pred)

        # labels and title 
        ax.set_xlabel('Date', fontsize = 12)
        ax.set_ylabel('Quantity', fontsize = 12)
        ax.set_title(
            f"{item} - MAE: {metrics['mae']}, RMSE: {metrics['rmse']}, "
            f"MAPE: {metrics['mape']}%, R²: {metrics['r2']}",
            fontsize=14, fontweight='bold'
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha = 0.3)

        # Rotate x-axis labels fro readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation = 45, ha = 'right')

    plt.tight_layout()

    # Save if path provided
    if save_path: 
        Path(save_path).parent.mkdir(parents = True, exist_ok= True)
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        print(f" Plot saved to: {save_path}")

    plt.close()

# Plotting residuals to see where model gets wrong
def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    item_labels: pd.Series,
    save_path: str = None
) -> None: 
    
    """
    Plot residual analysis to understand prediction errors.
    
    Residual = actual - predicted
    - Positive residual: Underestimated (predicted too low)
    - Negative residual: Overestimated (predicted too high)
    - Good model: Residuals randomly scattered around 0
    - Bad model: Patterns in residuals (systematic bias)
    
    Creates 3 plots:
    1. Residuals vs Predicted (check for heteroscedasticity)
    2. Residual histogram (check for normality)
    3. Residuals by item (check for bias per item)
    
    Parameters:
    -----------
    y_true : pd.Series
        True values
    y_pred : np.ndarray
        Predicted values
    item_labels : pd.Series
        Item names
    save_path : str, optional
        Path to save the plot
    """

    residuals = y_true.values - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residuals vs Predicted\n(Should be randomly scattered)', 
                      fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram of residuals
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residual Distribution\n(Should be centered at 0)', 
                      fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Boxplot by item
    df_residuals = pd.DataFrame({
        'residual': residuals,
        'item': item_labels.values
    })
    df_residuals.boxplot(column='residual', by='item', ax=axes[2])
    axes[2].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Item', fontsize=12)
    axes[2].set_ylabel('Residuals', fontsize=12)
    axes[2].set_title('Residuals by Item\n(Should be symmetric around 0)', 
                      fontsize=13, fontweight='bold')
    plt.sca(axes[2])
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Residual plot saved to: {save_path}")
    
    plt.close()

# Print Summary Report Function 
def print_evaluation_summary(
    metrics_overall: Dict[str, float],
    metrics_by_item: Dict[str, Dict[str, float]]
)-> None:
    """
    Print a formatted evaluation summary.
    
    Parameters:
    -----------
    metrics_overall : dict
        Overall metrics across all samples
    metrics_by_item : dict
        Metrics broken down by item
        
    Example Output:
    ---------------
    ========================================
    OVERALL METRICS
    ========================================
    MAE:   5.23
    RMSE:  7.14
    MAPE:  10.12%
    R²:    0.8532
    
    ========================================
    METRICS BY ITEM
    ========================================
    
    Croissant:
      MAE:   5.20
      RMSE:  7.10
      MAPE:  9.60%
      R²:    0.8500
    
    Danish:
      MAE:   3.10
      RMSE:  4.20
      MAPE:  10.40%
      R²:    0.8200
    """

    print("\n" + "="*50)
    print("OVERALL METRICS")
    print("="*50)
    print(f"MAE:   {metrics_overall['mae']}")
    print(f"RMSE:  {metrics_overall['rmse']}")
    print(f"MAPE:  {metrics_overall['mape']}%")
    print(f"R²:    {metrics_overall['r2']}")
    
    print("\n" + "="*50)
    print("METRICS BY ITEM")
    print("="*50)

    for item, metrics in metrics_by_item.items():
        print(f"\n{item}:")
        print(f"  MAE:   {metrics['mae']}")
        print(f"  RMSE:  {metrics['rmse']}")
        print(f"  MAPE:  {metrics['mape']}%")
        print(f"  R²:    {metrics['r2']}")

    print("\n")

