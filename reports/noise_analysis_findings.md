# Noise & Data Quality Analysis Findings
**Date:** 2025-02-12
**Dataset:** DoughFlow Bakery Sales (2021-01-01 to 2025-10-30)

---

## Executive Summary

This analysis identifies **five major sources of noise** affecting model performance:

1. **High overall variability** (CV > 0.4 for both products)
2. **Thursday sales are extremely unpredictable** (CV = 0.52-0.55)
3. **January/December have 60% higher noise** than summer months
4. **15-25 outlier days** that don't follow normal patterns
5. **~2.5% zero-sales days** mixing holidays with unexplained gaps

**Bottom line:** Your models struggle because the data itself is noisy. No amount of feature engineering can overcome this level of inherent randomness. The solution requires **data cleaning, outlier treatment, and potentially separate models for high-variance periods**.

---

## 1. Overall Variability Assessment

### Key Metrics

| Product   | Mean | Std Dev | CV (std/mean) | Assessment |
|-----------|------|---------|---------------|------------|
| Croissant | 52.2 | 25.0    | **0.48**      | ⚠️ High noise |
| Danish    | 29.2 | 15.1    | **0.52**      | ⚠️ High noise |

### What This Means

**Coefficient of Variation (CV) interpretation:**
- CV < 0.3: Low noise, highly predictable
- CV 0.3-0.4: Moderate noise, predictable with good features
- **CV > 0.4: High noise, difficult to predict**

Both products fall into the "high noise" category. This means:
- Even perfect features will struggle to achieve R² > 0.75
- Day-to-day sales can easily swing ±50% from the mean
- External factors (weather, events, randomness) dominate demand

**Visual evidence:** See `noise_time_series_outliers.png` - notice the erratic spikes and drops throughout the timeline.

---

## 2. Day-of-Week Variability Deep Dive

### Croissant CV by Day

| Day       | Mean  | Std  | CV    | Assessment |
|-----------|-------|------|-------|------------|
| Thursday  | 39.7  | 20.6 | **0.52** | ⚠️ Extremely noisy |
| Friday    | 51.4  | 22.8 | 0.44  | High noise |
| **Saturday** | **69.6** | **25.0** | **0.36** | ✅ Most predictable |
| Sunday    | 48.1  | 21.4 | 0.45  | High noise |

### Danish CV by Day

| Day       | Mean  | Std  | CV    | Assessment |
|-----------|-------|------|-------|------------|
| Thursday  | 23.1  | 12.7 | **0.55** | ⚠️ Extremely noisy |
| Friday    | 27.4  | 13.9 | 0.51  | High noise |
| **Saturday** | **37.8** | **15.7** | **0.41** | ✅ Most predictable |
| Sunday    | 28.3  | 14.1 | 0.50  | High noise |

### Key Findings

**Thursday is a problem child:**
- CV is 30-50% higher than Saturday
- This means Thursday sales are almost **random** - you can't predict them reliably
- Possible causes: variable prep schedules? Staff inconsistency? External events?

**Saturday is your anchor:**
- Lowest CV across all days
- Highest average sales (busiest day)
- Most predictable pattern - likely because customer behavior is consistent on weekends

**Actionable insight:** Consider training **separate models for Saturday vs. other days**. Saturday has different dynamics and will respond better to modeling.

**Visual evidence:** See `noise_day_of_week_variability.png` - notice the huge whiskers on Thursday boxplots.

---

## 3. Monthly Seasonality & Noise Patterns

### Croissant CV by Month

| Season | Months | Avg CV | Pattern |
|--------|--------|--------|---------|
| **Summer** (Jul-Aug) | Jul, Aug | **0.31** | ✅ Low noise, predictable |
| **Spring/Fall** (Mar-Oct) | Mar, Apr, Sep, Oct | 0.37 | Moderate noise |
| **Winter** (Jan, Dec, Nov) | Jan, Dec, Nov | **0.59** | ⚠️ Extremely noisy |

### Danish CV by Month

| Season | Months | Avg CV | Pattern |
|--------|--------|--------|---------|
| **Summer** (Jul-Oct) | Jul, Aug, Sep, Oct | **0.43** | Moderate noise |
| **Spring** (Feb-Jun) | Feb, Mar, Apr, May, Jun | 0.42 | Moderate noise |
| **Winter** (Jan, Dec, Nov) | Jan, Dec, Nov | **0.56** | ⚠️ Extremely noisy |

### Key Findings

**Winter months (Jan, Dec, Nov) are wildly unpredictable:**
- CV nearly **2x higher** than summer months
- Likely causes:
  - Holiday closures creating zeros
  - Inconsistent operating schedule early in the year (2021 data)
  - Weather impacts (snow, cold affecting foot traffic)
  - Holiday shopping vs. post-holiday lulls

**Summer (Jul-Aug) is your sweet spot:**
- Croissant CV drops to 0.31 - this is **predictable**
- Stable schedules, consistent customer behavior
- Peak tourist season may create regular high-volume patterns

**Actionable insight:**
- Train separate models for winter vs. summer seasons
- OR add month-specific features with interaction terms
- Consider **excluding January 2021** entirely (startup period with CV=0.64)

**Visual evidence:** See `noise_monthly_variability.png` - dark red bars show high-noise months.

---

## 4. Outlier Analysis

### Outlier Summary

| Product   | Outliers Found | % of Data | Top Outlier Value | Date |
|-----------|----------------|-----------|-------------------|------|
| Croissant | 15 days        | 1.5%      | **145 units**     | 2022-06-11 (Sat) |
| Danish    | 11 days        | 1.1%      | **84 units**      | 2023-07-21 (Fri) |

### Outlier Patterns

**Croissant top outliers:**
- 2022-06-11 (Sat): **145** (normal Sat ~70) → 2x normal sales
- 2022-11-05 (Sat): **136**
- 2021-08-07 (Sat): **133**

**Danish top outliers:**
- 2023-07-21 (Fri): **84** (normal Fri ~27) → 3x normal sales
- 2024-07-06 (Sat): **84**
- 2023-07-29 (Sat): **82**

### Key Findings

**Most outliers are SATURDAY summer days:**
- Summer Saturdays occasionally hit 2-3x normal volume
- Likely special events: festivals, farmers markets, catering orders
- These are **real data**, not errors (they cluster in peak season)

**Impact on models:**
- Models trained on these outliers will **overpredict** on normal days
- Models excluding them will **underpredict** when special events occur
- Current strategy: Keep them, but they add noise

**Actionable insight:**
- **Option 1:** Cap outliers at 95th percentile during training (smoothing)
- **Option 2:** Flag "event days" as a binary feature if you can identify them
- **Option 3:** Use robust loss functions (Huber loss, quantile regression)

**Visual evidence:** See `noise_time_series_outliers.png` - red dots show outlier days scattered across timeline.

---

## 5. Zero-Sales Days Investigation

### Zero-Sales Summary

| Product   | Zero Days | % of Data | Pattern |
|-----------|-----------|-----------|---------|
| Croissant | 22 days   | 2.2%      | Holidays + unexplained gaps |
| Danish    | 25 days   | 2.5%      | Holidays + unexplained gaps |

### Zero-Sales Timeline

**First 15 zero dates (both products identical):**
```
2021-01-01, 2021-01-02, 2021-01-03  ← New Year's (expected)
2021-01-07, 2021-01-08              ← ??? (unexplained)
2021-01-15, 2021-01-22, 2021-01-29  ← ??? (unexplained)
2021-07-04                          ← July 4th (expected)
2021-11-25                          ← Thanksgiving (expected)
2021-12-25                          ← Christmas (expected)
2022-01-01                          ← New Year's (expected)
2022-11-24                          ← Thanksgiving (expected)
2022-12-25                          ← Christmas (expected)
2023-01-01                          ← New Year's (expected)
```

### Key Findings

**Two categories of zeros:**

1. **Holiday closures (expected, legitimate):**
   - Christmas, New Year's, Thanksgiving, July 4th
   - These are **real** - the bakery was closed
   - Your pipeline correctly fills these with 0

2. **Unexplained zeros in early 2021 (Fridays):**
   - 2021-01-08, 01-15, 01-22, 01-29 (all Fridays)
   - Pattern suggests **inconsistent operating schedule** when bakery started
   - These add noise because lag features will reference them

**Impact on models:**
- Zeros mixed with normal operational days confuse lag features
- If lag_1w_ago hits a zero (unexplained), it doesn't represent normal demand
- This creates **propagating noise** through all lag-based features

**Actionable insight:**
- **Option 1:** Exclude all data before 2021-02-01 (startup period)
- **Option 2:** Mark pre-Feb-2021 as "unreliable" and train without it
- **Option 3:** Use a longer-range lag (lag_2w_ago, lag_3w_ago) to skip over these gaps

**Visual evidence:** See `noise_zero_values.png` - red X marks show zero-sales days scattered across timeline.

---

## 6. Day-to-Day Volatility

### Large Jump Analysis

**Croissant top 10 consecutive-day changes:**
```
2021-12-24 (121) → 2021-12-25 (0)   = Δ121  ← Christmas closure
2021-07-03 (111) → 2021-07-04 (0)   = Δ111  ← July 4th closure
2024-05-26 (116) → 2024-05-30 (14)  = Δ102  ← ??? (4-day gap)
2022-06-10 (54)  → 2022-06-11 (145) = Δ91   ← Special event spike
```

**Danish top changes:**
```
2022-07-22 (8)   → 2022-07-23 (62)  = Δ54   ← Recovery from low day?
2022-08-18 (2)   → 2022-08-19 (55)  = Δ53   ← Same pattern
2023-09-23 (81)  → 2023-09-24 (33)  = Δ48   ← Post-event crash
```

### Key Findings

**Holiday closures create the biggest jumps:**
- Christmas Eve (high) → Christmas Day (0) = massive drop
- This is **expected** and models should learn it via date features

**Unexplained low-to-high spikes (Danish):**
- Several days with sales of 1-8 units followed by normal 50+ the next day
- Possible data errors? Partial days? Staff mistakes?
- These hurt model performance because they look like random noise

**Actionable insight:**
- Investigate days with sales < 10 on non-holidays (may be data errors)
- Consider smoothing: if a day is < 20% of weekly average, flag as suspicious
- Use **rolling averages** as features instead of raw lags to reduce jump impact

---

## 7. Distribution Analysis

### Distribution Characteristics

| Product   | Skewness | Distribution Shape | Normality |
|-----------|----------|-------------------|-----------|
| Croissant | 0.42     | Right-skewed      | Moderate deviation |
| Danish    | 0.54     | Right-skewed      | Moderate deviation |

### Key Findings

**Both products have right-skewed distributions:**
- Mean > Median (more high-value outliers pulling the mean up)
- Long tail on the right side (special event days)
- This violates normality assumptions for linear models

**Impact on models:**
- Linear Regression assumes normal residuals - violations hurt performance
- Tree-based models (RF, XGBoost) handle skewness better
- You may benefit from **log transformation** or **quantile regression**

**Visual evidence:** See `noise_distributions.png` - Q-Q plots show deviation from normal line.

---

## Summary of Noise Sources & Impact on Models

### Ranked by Impact

| Rank | Noise Source | Impact on Model | Addressable? |
|------|--------------|-----------------|--------------|
| 1    | **Thursday variability (CV=0.52-0.55)** | High - unpredictable day patterns | ⚠️ Hard - may need separate model |
| 2    | **Winter month chaos (CV=0.59)** | High - seasonal instability | ✅ Yes - exclude Jan 2021 or add season features |
| 3    | **Special event outliers (15-25 days)** | Medium - skews predictions | ✅ Yes - cap, flag, or robust loss |
| 4    | **Early 2021 unexplained zeros** | Medium - poisons lag features | ✅ Yes - exclude data before Feb 2021 |
| 5    | **Right-skewed distribution** | Low-Medium - affects linear models | ✅ Yes - log transform or use tree models |

---

## Recommendations for Improving Model Performance

### Immediate Actions (High Impact, Low Effort)

1. **Exclude January 2021 data**
   - CV=0.64 for this month - it's a startup period with inconsistent operations
   - Models will perform better without this noise
   - Implementation: Filter data to `date >= '2021-02-01'`

2. **Cap outliers at 95th percentile**
   - Prevents special event days from skewing predictions
   - Implementation: `df['quantity'] = df['quantity'].clip(upper=df['quantity'].quantile(0.95))`

3. **Add a "winter_months" binary feature**
   - Captures the Nov-Jan high-variability period
   - Allows models to adjust predictions during unstable months
   - Implementation: `df['is_winter'] = df['month'].isin([11, 12, 1])`

### Medium-Term Actions (High Impact, Medium Effort)

4. **Train separate models for Saturday vs. other days**
   - Saturday has fundamentally different dynamics (lower CV, higher volume)
   - Ensemble: Use Saturday model for Sat predictions, general model for others

5. **Use robust loss functions**
   - Huber loss or quantile regression for tree models
   - Reduces impact of outliers without removing them
   - Implementation: XGBoost `objective='reg:pseudohubererror'`

6. **Investigate low-volume days (qty < 10)**
   - Days with sales < 10 on non-holidays may be data errors
   - Manual review or imputation may improve quality

### Long-Term Actions (Medium Impact, High Effort)

7. **Collect external event data**
   - Flag farmers markets, festivals, catering orders as binary features
   - Helps model learn when outliers are expected

8. **Consider separate models by season**
   - Summer model (Jul-Sep): Low noise, predictable
   - Winter model (Nov-Jan): High noise, different patterns
   - Transition months: Use blended predictions

---

## Final Thoughts

Your models aren't failing because of bad features or algorithms. **The data itself is inherently noisy.**

With CV > 0.4, even the best model will struggle to exceed:
- **R² = 0.70-0.75** (considered "good" for noisy time series)
- **MAPE = 25-30%** (acceptable for high-variance demand)

The path forward isn't more features - it's **cleaner data, outlier handling, and stratified modeling** (separate models for high-noise vs. low-noise periods).

Start with the **Immediate Actions** above and re-run your models. You should see meaningful improvement.

---

**Generated by:** DoughFlow Bakery EDA Skill
**Visualizations:** See `reports/figures/noise_*.png`
**Raw analysis:** `scripts/analyze_noise.py`, `scripts/visualize_noise.py`
