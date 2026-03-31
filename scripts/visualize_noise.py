#!/usr/bin/env python3
"""
Create visualizations for noise and data quality analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Create reports/figures directory if needed
Path('reports/figures').mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
train = pd.read_csv('data/processed/train.csv')
test = pd.read_csv('data/processed/test.csv')
df = pd.concat([train, test], ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print("Creating visualizations...")

# 1. Time series with outliers highlighted
print("  1. Time series with outliers...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

for idx, item in enumerate(['Croissant', 'Danish']):
    item_df = df[df['item'] == item].copy()

    # Calculate outlier bounds
    Q1 = item_df['quantity'].quantile(0.25)
    Q3 = item_df['quantity'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Mark outliers
    item_df['is_outlier'] = (item_df['quantity'] < lower_bound) | (item_df['quantity'] > upper_bound)

    # Plot
    axes[idx].plot(item_df['date'], item_df['quantity'], alpha=0.5, linewidth=1, color='blue', label='Normal')
    outliers = item_df[item_df['is_outlier']]
    axes[idx].scatter(outliers['date'], outliers['quantity'], color='red', s=50, zorder=5, label='Outliers', alpha=0.7)

    # Add horizontal lines for bounds
    axes[idx].axhline(y=upper_bound, color='orange', linestyle='--', alpha=0.5, label=f'Upper bound ({upper_bound:.0f})')
    axes[idx].axhline(y=item_df['quantity'].mean(), color='green', linestyle='-', alpha=0.5, label=f'Mean ({item_df["quantity"].mean():.0f})')

    axes[idx].set_title(f'{item} - Daily Sales with Outliers Highlighted', fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('Quantity')
    axes[idx].legend(loc='upper left')
    axes[idx].grid(True, alpha=0.3)

axes[1].set_xlabel('Date')
plt.tight_layout()
plt.savefig('reports/figures/noise_time_series_outliers.png', dpi=150, bbox_inches='tight')
print("    Saved: reports/figures/noise_time_series_outliers.png")

# 2. Day-of-week variability comparison
print("  2. Day-of-week boxplots...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for idx, item in enumerate(['Croissant', 'Danish']):
    item_df = df[df['item'] == item].copy()
    item_df['day_name'] = item_df['day_of_week'].apply(lambda x: day_names[int(x)])

    # Order by day of week
    day_order = [day_names[int(d)] for d in sorted(item_df['day_of_week'].unique())]

    sns.boxplot(data=item_df, x='day_name', y='quantity', ax=axes[idx], order=day_order, palette='Set2')
    axes[idx].set_title(f'{item} - Variability by Day of Week', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Day of Week')
    axes[idx].set_ylabel('Quantity')
    axes[idx].grid(True, alpha=0.3, axis='y')

    # Add CV values as text
    for pos, day in enumerate(day_order):
        day_num = day_names.index(day)
        day_data = item_df[item_df['day_of_week'] == day_num]['quantity']
        cv = day_data.std() / day_data.mean()
        axes[idx].text(pos, axes[idx].get_ylim()[1] * 0.95, f'CV={cv:.2f}',
                      ha='center', va='top', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/noise_day_of_week_variability.png', dpi=150, bbox_inches='tight')
print("    Saved: reports/figures/noise_day_of_week_variability.png")

# 3. Monthly CV comparison
print("  3. Monthly variability heatmap...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df['month'] = df['date'].dt.month

for idx, item in enumerate(['Croissant', 'Danish']):
    item_df = df[df['item'] == item]

    monthly_stats = item_df.groupby('month')['quantity'].agg(['mean', 'std']).reset_index()
    monthly_stats['cv'] = monthly_stats['std'] / monthly_stats['mean']
    monthly_stats['month_name'] = monthly_stats['month'].apply(lambda x: month_names[int(x)-1])

    # Bar plot
    bars = axes[idx].bar(range(len(monthly_stats)), monthly_stats['cv'], color='steelblue', edgecolor='black', alpha=0.7)
    axes[idx].set_xticks(range(len(monthly_stats)))
    axes[idx].set_xticklabels(monthly_stats['month_name'], rotation=45, ha='right')
    axes[idx].set_title(f'{item} - Coefficient of Variation by Month', fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('Coefficient of Variation (CV)')
    axes[idx].set_xlabel('Month')
    axes[idx].grid(True, alpha=0.3, axis='y')
    axes[idx].axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='High noise threshold (0.4)')
    axes[idx].legend()

    # Color bars based on CV
    for i, bar in enumerate(bars):
        if monthly_stats.iloc[i]['cv'] > 0.5:
            bar.set_color('darkred')
            bar.set_alpha(0.8)

plt.tight_layout()
plt.savefig('reports/figures/noise_monthly_variability.png', dpi=150, bbox_inches='tight')
print("    Saved: reports/figures/noise_monthly_variability.png")

# 4. Distribution comparison
print("  4. Distribution plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for idx, item in enumerate(['Croissant', 'Danish']):
    item_df = df[df['item'] == item]

    # Histogram
    axes[idx, 0].hist(item_df['quantity'], bins=40, edgecolor='black', alpha=0.7, color='steelblue')
    axes[idx, 0].axvline(item_df['quantity'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean ({item_df["quantity"].mean():.1f})')
    axes[idx, 0].axvline(item_df['quantity'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median ({item_df["quantity"].median():.1f})')
    axes[idx, 0].set_title(f'{item} - Sales Distribution', fontsize=12, fontweight='bold')
    axes[idx, 0].set_xlabel('Quantity')
    axes[idx, 0].set_ylabel('Frequency')
    axes[idx, 0].legend()
    axes[idx, 0].grid(True, alpha=0.3)

    # Q-Q plot to check normality
    from scipy import stats
    stats.probplot(item_df['quantity'], dist="norm", plot=axes[idx, 1])
    axes[idx, 1].set_title(f'{item} - Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
    axes[idx, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/noise_distributions.png', dpi=150, bbox_inches='tight')
print("    Saved: reports/figures/noise_distributions.png")

# 5. Consecutive day changes (volatility)
print("  5. Day-to-day volatility...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

for idx, item in enumerate(['Croissant', 'Danish']):
    item_df = df[df['item'] == item].copy().sort_values('date')
    item_df['qty_change'] = item_df['quantity'].diff().abs()
    item_df['pct_change'] = item_df['quantity'].pct_change().abs() * 100

    # Plot absolute changes
    axes[idx].plot(item_df['date'], item_df['qty_change'], alpha=0.6, linewidth=1, color='purple')
    axes[idx].fill_between(item_df['date'], 0, item_df['qty_change'], alpha=0.3, color='purple')

    mean_change = item_df['qty_change'].mean()
    axes[idx].axhline(y=mean_change, color='red', linestyle='--', linewidth=2, label=f'Mean change ({mean_change:.1f})')
    axes[idx].axhline(y=mean_change + 2*item_df['qty_change'].std(), color='orange', linestyle='--',
                     linewidth=1, alpha=0.7, label='Mean + 2σ')

    axes[idx].set_title(f'{item} - Day-to-Day Absolute Change in Sales', fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('Absolute Change')
    axes[idx].legend(loc='upper left')
    axes[idx].grid(True, alpha=0.3)

axes[1].set_xlabel('Date')
plt.tight_layout()
plt.savefig('reports/figures/noise_volatility.png', dpi=150, bbox_inches='tight')
print("    Saved: reports/figures/noise_volatility.png")

# 6. Zero values timeline
print("  6. Zero values analysis...")
fig, axes = plt.subplots(2, 1, figsize=(16, 8))

for idx, item in enumerate(['Croissant', 'Danish']):
    item_df = df[df['item'] == item].copy()
    zeros = item_df[item_df['quantity'] == 0]
    non_zeros = item_df[item_df['quantity'] > 0]

    axes[idx].scatter(non_zeros['date'], non_zeros['quantity'], alpha=0.3, s=20, color='blue', label='Sales > 0')
    axes[idx].scatter(zeros['date'], zeros['quantity'], s=100, color='red', marker='X',
                     label=f'Zero sales ({len(zeros)} days)', zorder=5)

    axes[idx].set_title(f'{item} - Zero Sales Days Highlighted', fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('Quantity')
    axes[idx].legend(loc='upper left')
    axes[idx].grid(True, alpha=0.3)

axes[1].set_xlabel('Date')
plt.tight_layout()
plt.savefig('reports/figures/noise_zero_values.png', dpi=150, bbox_inches='tight')
print("    Saved: reports/figures/noise_zero_values.png")

print("\n✅ All visualizations saved to reports/figures/")
print("\nGenerated files:")
print("  - noise_time_series_outliers.png")
print("  - noise_day_of_week_variability.png")
print("  - noise_monthly_variability.png")
print("  - noise_distributions.png")
print("  - noise_volatility.png")
print("  - noise_zero_values.png")
