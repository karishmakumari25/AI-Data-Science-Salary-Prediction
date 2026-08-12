import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



df = pd.read_csv("dataset/ai_ds_job_salaries_2026.csv")

print("Dataset Loaded ✅")
print("Shape:", df.shape)



print("Missing Values:")
print(df.isnull().sum())

print("Duplicate Rows:", df.duplicated().sum())


df = df.drop_duplicates()



X = df.drop("salary_usd", axis=1)
y = df["salary_usd"]



categorical_columns = X.select_dtypes(include=["object"]).columns


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ],
    remainder="passthrough"
)



model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



print("Training Linear Regression Model...")

model.fit(X_train, y_train)

print("Model Training Complete ✅")



y_pred = model.predict(X_test)



mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)



print("\nActual Salary:", y_test.iloc[0])
print("Predicted Salary:", y_pred[0])



joblib.dump(
    model,
    "models/salary_prediction_model.pkl"
)

print("\nModel saved successfully ✅")