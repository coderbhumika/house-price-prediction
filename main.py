import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("House Price Prediction Dataset.csv")

print(df.head(5))

print(df.info())

print(df.isnull().sum())

print(df.duplicated().sum())


le_location = LabelEncoder()
le_condition = LabelEncoder()
le_garage = LabelEncoder()

df["Location"] = le_location.fit_transform(df["Location"])
df["Condition"] = le_condition.fit_transform(df["Condition"])
df["Garage"] = le_garage.fit_transform(df["Garage"])

print(df.head(5))

x = df.drop("Price", axis=1)
y = df["Price"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)