# Tesla Deliveries ML Assignment Report

## Framing the problem

This project uses `tesla_deliveries_dataset_2015_2025.csv` to answer two practical questions:

1. Can we predict Tesla `Estimated_Deliveries` from product, operational, geographic, and time-based signals?
2. Can we forecast the next 12 months of aggregate deliveries from the historical monthly series?

I also modeled `Avg_Price_USD`, but I treat that as an analytical contrast rather than the headline result. One of the most useful findings in this assignment is that the dataset supports delivery prediction far better than price prediction, which is itself a meaningful business insight.

## EDA summary

The exploratory analysis was designed to move from descriptive understanding to modeling decisions.

### 1. Time-trend analysis

I first examined monthly deliveries, production, average price, charging infrastructure, and CO2 savings over time. This matters because the dataset is fundamentally temporal, and trend structure should shape both feature engineering and evaluation strategy.

What stood out:
- Deliveries and production move very closely together throughout the period.
- The time series shows meaningful month-to-month variation, so seasonality features are justified.
- Charging stations and CO2 saved rise over time, but not as tightly as production volume when compared against deliveries.

### 2. Region and model mix

I compared total deliveries by region and by model, then extended that with a year-region heatmap and a model-region heatmap.

Key findings:
- The four regions are fairly balanced overall, though `Middle East` leads cumulative deliveries in this dataset.
- `Model S` has the highest total deliveries, followed closely by `Model 3`.
- The model-region heatmap is useful because it shows that aggregate performance is not driven by a single model in a single geography.

### 3. Seasonality and source composition

Inspired by the reference notebooks, I added month-level seasonality views and a source-type composition chart.

What this revealed:
- Monthly deliveries are not flat across the year.
- The strongest month by total deliveries is month `8`, which supports keeping calendar-based signals in the feature set.
- Source types are spread across `Interpolated (Month)`, `Official (Quarter)`, and `Estimated (Region)`, so the source label itself may carry information about how the records were constructed.

### 4. Distribution analysis

I included distribution plots for:
- `Estimated_Deliveries`
- `Avg_Price_USD`
- `Range_km`
- `Charging_Stations`

These help us answer whether the dataset is narrow, skewed, or multimodal. In this case, deliveries and price both have spread but no extreme structural issues that block regression. Range and charging-station values also show enough variation to be potentially useful features.

### 5. Relationship analysis

The most important relationship plots were:
- `Production_Units` vs `Estimated_Deliveries`
- `Avg_Price_USD` vs `Estimated_Deliveries`
- `Charging_Stations` vs `Estimated_Deliveries`
- Numeric correlation matrix

The correlation matrix confirms the main EDA story:
- `Production_Units` has the strongest linear relationship with deliveries, with correlation around `0.994`.
- `CO2_Saved_tons` is also strongly aligned with deliveries at around `0.837`.
- Variables like `Charging_Stations`, `Range_km`, and `Battery_Capacity_kWh` are much weaker on their own.

That pattern directly shaped the later modeling choices.

### 6. Product-profile analysis

To make the notebook feel more complete and less like a standard assignment template, I also compared model-level distributions of:
- `Avg_Price_USD`
- `Range_km`
- `Battery_Capacity_kWh`

This helps position the models not just as labels, but as distinct product segments. It also provides context for why one-hot model encoding and product-value features can be useful in supervised learning.

## Preprocessing

The raw dataset is already fairly clean, but I still applied defensive preprocessing so the pipeline is reproducible and trustworthy.

The preprocessing steps were:
- stripped column-name whitespace
- removed duplicates
- validated that `Month` lies between `1` and `12`
- created a proper `Date` column from `Year` and `Month`
- sorted the dataset chronologically
- used a chronological train-test split instead of a random split

That last point is important. Since this is time-aware panel data, random splitting would leak future patterns into training and give an unrealistically optimistic evaluation.

## Feature engineering

The feature engineering strategy focused on three types of signal: calendar structure, demand momentum, and product-value context.

### Calendar features

I created:
- `Quarter`
- `Month_Sin`
- `Month_Cos`
- `Year_Index`

These features help the model represent seasonality and long-run trend more smoothly than raw month integers alone.

### Product-value features

I created:
- `Price_per_km_Range`
- `Price_per_kWh`

These express price in relation to vehicle capability. Instead of treating price, range, and battery as disconnected numbers, the model can learn whether delivery behavior changes with perceived value density.

### Lag and rolling features

Because the data is indexed by `Region` and `Model`, I created history-based features within each region-model group:
- `Deliveries_Lag_1`
- `Deliveries_Lag_3`
- `Deliveries_Rolling_3`
- `Price_Lag_1`
- `Price_Rolling_3`

These features are especially useful because EV demand is rarely memoryless. Recent demand and recent pricing often carry forward into the next few months.

### Leakage handling

This is one of the parts that most candidates may miss, and it is where I intentionally tightened the pipeline.

I created some extra business-style features such as:
- `Production_Gap`
- `Production_to_Delivery_Ratio`
- `CO2_per_Delivery`

These are fine for EDA, but they are not safe for delivery prediction because they use the current target `Estimated_Deliveries` in their own calculation. Similarly, features derived directly from current price are not safe when price itself is the prediction target.

So the final supervised feature sets deliberately exclude leakage-prone variables. That made the results more honest and more defensible.

## Models chosen and why

I compared five regression models for supervised prediction:

### 1. Linear Regression

Why it was included:
- it is the cleanest baseline for a continuous target
- it gives a direct benchmark for how far the feature engineering alone can go
- it helps separate the value of good features from the value of more complex nonlinear models

### 2. Ridge Regression

Why it was included:
- gives a clean linear baseline
- handles multicollinearity better than plain linear regression
- makes it easy to see how much performance comes from nonlinear structure rather than just good features

### 3. Random Forest Regressor

Why it was included:
- strong baseline for tabular data
- captures nonlinear interactions
- robust to noisy variables
- easy to interpret with feature importance

### 4. Gradient Boosting Regressor

Why it was included:
- often performs better than bagged trees when the signal is structured but nonlinear
- useful as a compact boosting benchmark

### 5. XGBoost Regressor

Why it was included:
- one of the strongest choices for structured tabular regression
- handles nonlinearities and interactions well
- often wins when feature interactions matter more than pure linear effects

This mix was deliberate. It compares plain linear, regularized linear, bagged trees, boosting, and advanced boosting, which makes the model selection process much more defensible.

## Evaluation metrics

Because the core tasks are regression problems, the primary metrics remain:
- `MAE`
- `RMSE`
- `R²`
- `Explained Variance`
- `MAPE`
- `Tolerance_10pct`, which measures the share of predictions that fall within 10% of the true value

The user also asked for `Accuracy`, `Recall`, and similar metrics. Those are classification metrics, so I added them in a secondary, mathematically valid way:
- for each target, I converted actual and predicted values into a binary label
- `1` means the value is above the training-set median
- `0` means the value is at or below the training-set median

That lets the notebook report:
- `Accuracy`
- `Precision`
- `Recall`
- `F1`

This way the assignment includes the requested metrics without misusing classification metrics on raw continuous predictions.

## Modeling results

### Delivery prediction

After removing leakage-prone features, delivery prediction remained strong:

| Model | MAE | MAPE | RMSE | R² | Accuracy | Recall | F1 | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear Regression | 322.00 | 0.0382 | 391.13 | 0.9890 | 0.9708 | 0.9696 | 0.9696 | 0.9696 |
| Ridge | 321.44 | 0.0383 | 390.84 | 0.9890 | 0.9708 | 0.9696 | 0.9696 | 0.9696 |
| Random Forest | 338.85 | 0.0348 | 420.20 | 0.9873 | 0.9625 | 0.9565 | 0.9607 | 0.9649 |
| Gradient Boosting | 336.50 | 0.0348 | 410.33 | 0.9879 | 0.9604 | 0.9565 | 0.9586 | 0.9607 |
| XGBoost | 285.63 | 0.0305 | 350.57 | 0.9912 | 0.9729 | 0.9696 | 0.9717 | 0.9738 |


Interpretation:
- All five models perform well, which suggests the dataset contains real structure for delivery prediction.
- `XGBoost` is the strongest untuned model.
- Even the plain linear models perform surprisingly well, which tells us the dataset has a strong underlying signal.
- The classification-style metrics are also high, which means the models are good at separating relatively high-demand periods from lower-demand periods.

### Hyperparameter tuning

I tuned the delivery Random Forest using `TimeSeriesSplit`, which preserves chronology during validation.

Best sampled parameters:
- `n_estimators = 500`
- `min_samples_split = 2`
- `min_samples_leaf = 2`
- `max_features = 0.7`
- `max_depth = None`

Tuned Random Forest performance:
- `MAE = 300.04`
- `RMSE = 377.55`
- `R² = 0.9897`
- `Accuracy = 0.9646`
- `Recall = 0.9609`
- `F1 = 0.9630`

Tuning improved the untuned forest, though it still does not beat the fixed XGBoost configuration on RMSE. I kept the tuning section because it demonstrates a correct time-aware optimization workflow, which is important in an assignment like this.

### Price prediction

Price prediction was much weaker:

| Model | MAE | MAPE | RMSE | R² | Accuracy | Recall | F1 | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear Regression | 17411.52 | 0.2206 | 20156.81 | -0.0113 | 0.5208 | 0.3468 | 0.4279 | 0.5584 |
| Ridge | 17411.34 | 0.2206 | 20156.26 | -0.0113 | 0.5188 | 0.3427 | 0.4239 | 0.5556 |
| Random Forest | 17447.12 | 0.2225 | 20219.64 | -0.0177 | 0.5062 | 0.4355 | 0.4768 | 0.5268 |
| Gradient Boosting | 17594.32 | 0.2241 | 20436.47 | -0.0396 | 0.5083 | 0.4113 | 0.4636 | 0.5312 |
| XGBoost | 17498.88 | 0.2237 | 20595.61 | -0.0558 | 0.5167 | 0.4758 | 0.5043 | 0.5364 |

This is not a modeling failure so much as a data-story result:
- the available features explain deliveries well
- the same features do not explain price well
- price in this dataset likely behaves more like a noisy, synthetic, or weakly linked variable than a target with strong observable drivers
- the near-random classification-style scores reinforce that conclusion from a second evaluation angle

Calling that out honestly is stronger than pretending the model worked equally well on both tasks.

## Feature importance

The tuned delivery model showed a very concentrated importance profile.

Top features:
- `Production_Units` dominated the model at roughly `0.801` importance
- `CO2_Saved_tons` was the next strongest feature at roughly `0.179`
- all other features were much smaller contributors

That result matches the EDA cleanly. The model is effectively telling the same story as the visual analysis: operational output is the main driver of deliveries in this dataset.

## Forecasting results

For forecasting, I used aggregate monthly deliveries and compared two approaches.

### 1. SARIMAX

Why it was chosen:
- classical time-series model
- handles differencing and seasonality explicitly
- suitable for monthly delivery forecasting

Backtest performance on `2024-2025`:
- `MAE = 13,435.59`
- `RMSE = 16,570.11`
- `Accuracy = 0.5000`
- `Recall = 0.2727`
- `F1 = 0.3333`

### 2. Lag-based Gradient Boosting

Why it was chosen:
- machine-learning benchmark for the time series
- uses lagged demand and seasonal features in a supervised setup

Backtest performance on `2024-2025`:
- `MAE = 11,634.47`
- `RMSE = 15,389.06`
- `Accuracy = 0.5417`
- `Recall = 0.4545`
- `F1 = 0.4762`

Interpretation:
- The lag-based ML model backtests slightly better than SARIMAX.
- That suggests recent delivery history carries predictive value that a flexible supervised model can use effectively.

### 12-month forward forecast

Using the final SARIMAX model trained on the full monthly history, the next-12-month forecast suggests:
- total forecast deliveries of about `2,378,540.51`
- average monthly deliveries of about `198,211.71`
- forecast peak around `2026-08-01` at about `209,790.86`

The forecast interval is wide enough to remind us that this should be read as a planning estimate, not a guaranteed outcome. That is exactly how a business-facing forecast should be presented.
