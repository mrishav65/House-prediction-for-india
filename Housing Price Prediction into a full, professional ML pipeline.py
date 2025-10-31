
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


df = pd.read_csv(r"C:\Users\mrish\Downloads\india_housing_prices.csv")


print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())


df = df.dropna()


X = df.drop("Price_in_Lakhs", axis=1)
y = df["Price_in_Lakhs"]


X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ]),
    "Lasso Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=0.1))
    ]),
    "Random Forest": Pipeline([
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
}


results = []

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    cv_score = cross_val_score(pipeline, X, y, cv=5).mean()

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "CV Score": cv_score
    })


results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)


plt.figure(figsize=(8,6))
sns.barplot(x="R²", y="Model", data=results_df, palette="viridis")
plt.title("Model Performance Comparison (R² Score)")
plt.show()


import joblib
best_model = models["Random Forest"]
best_model.fit(X, y)
joblib.dump(best_model, "best_indian_house_price_model.pkl")
print("Best model saved as 'best_indian_house_price_model.pkl'")

