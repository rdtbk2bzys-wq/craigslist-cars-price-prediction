## About the Project

This project is about predicting used car prices using a Craigslist vehicle dataset.

I worked with the data, cleaned missing and incorrect values, created new features, compared different machine learning models and analyzed the errors made by the models.

The main goal was to build a model that can predict car prices as accurately as possible.

## Dataset

The dataset contains information about used cars listed on Craigslist https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data

Some of the main columns are:

- year
- manufacturer
- model
- condition
- fuel
- odometer
- transmission
- drive
- type
- paint color
- region
- latitude and longitude
- posting date
- price

The target variable is `price`.

## Data Cleaning

The original dataset contained many missing and incorrect values, so I cleaned the data before training the models.

Main steps:

- removed unnecessary columns
- removed cars with unrealistic prices
- replaced negative odometer values with missing values
- filled missing values using related features
- filled missing categorical values with `unknown`
- removed the `description` column because I did not use text processing in this project
- removed the `model` column because it contained many inconsistent and incorrect values

## Feature Engineering

I created a new feature called `age`.

It shows the age of the car when the listing was posted.

```python
age = posting_year - car_year
```

I also checked several possible new features, but not all of them improved the model.

## Preprocessing

Numerical features were scaled using `StandardScaler`.

Categorical features were encoded using:

- `OneHotEncoder`
- `TargetEncoder`

The preprocessing and model were combined into a single `Pipeline`.

This also helped keep the preprocessing consistent between the training and test data.

## Model Comparison

I tested several regression models:

- Linear Regression
- Lasso
- Ridge
- KNN
- Decision Tree
- Random Forest
- Gradient Boosting
- Bagging
- XGBoost

Random Forest gave the best result among the models I tested.

## Error Analysis

After training the model, I looked at the predictions with the largest errors.

I also divided the cars into several price groups to see how the error changes depending on the car price.

For most cars below $50,000, the MAE was around $2,300–3,000.

More expensive cars had much larger errors.

This showed that the model works better for the main part of the dataset, while expensive cars are harder to predict.

## Feature Importance

I used permutation importance to check which features have the biggest effect on the model.

This helped me understand which information is most useful for predicting the price.

## Hyperparameter Tuning

I also tried to improve the Random Forest model using `RandomizedSearchCV`.

The tested parameter combinations did not improve the results.

The tuned model achieved approximately:

- R²: 0.80
- MAE: $2,889

This was worse than the original Random Forest, so I kept the original model as the final one.

## Final Results

The final Random Forest model achieved approximately:

- **R²: 0.83**
- **MAE: $2,400**

The model performs best on the majority of cars in the lower and middle price ranges.

The biggest errors are mostly found among expensive cars and listings with inconsistent data.

## Limitations

The biggest problem with this dataset is the quality of some of the original information.

The `model` column contained many different and sometimes incorrect values. Because of this, I decided not to use it in the final model.

This means that some useful information about the exact car model was lost.

Further improvement would probably require better cleaning or extraction of the car model information rather than only trying more hyperparameter combinations.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Category Encoders
- Matplotlib
- Seaborn
