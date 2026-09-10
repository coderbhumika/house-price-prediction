import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('House Price Prediction Dataset.csv')

print("Dataset Loaded Successfully!")
print("\nFirst 5 rows: ")
print(df.head())

print("\nDatabase information: ")
print(df.info())

X = df.drop("Price", axis=1)
y = df["Price"]

numerical_features = [
    "Id",
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "YearBuilt"
]

categorical_features = [
    "Location",
    "Condition",
    "Garage"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)

model.fit(X_train, y_train)

print("\nModel trained successfully!")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n----- Model Evaluation -----")

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices:")
print(comparison.head(10))

new_house = pd.DataFrame({
    "Id": [2001],
    "Area": [2500],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "Floors": [2],
    "YearBuilt": [2015],
    "Location": ["Suburban"],
    "Condition": ["Good"],
    "Garage": ["Yes"]
})

predicted_price = model.predict(new_house)

print("\n----- New House Prediction -----")
print("Predicted House Price:", predicted_price[0])