import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("house_price_regression_dataset.csv")

print(df.head())

df.info()

x = df.drop(columns=["House_Price"])
y = df["House_Price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

print(len(x_train), len(x_test), len(y_train), len(y_test))

linreg = LinearRegression()

import joblib

pipe_fittato = linreg.fit(x_train, y_train)

joblib.dump(pipe_fittato, "linear_model.joblib")