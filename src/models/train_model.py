import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns

import pathlib
import pickle
# pd.set_option('max_columns', 55)


data_dir = pathlib.Path(__file__).parents[2].joinpath(
    'data/raw')
model_dir = pathlib.Path(__file__).parents[2].joinpath(
    'models/')

train_dir = os.path.join(data_dir, "internship_train.csv")
test_dir = os.path.join(data_dir, "internship_hidden_test.csv")

# print(train_dir)

train_data = pd.read_csv(train_dir)
test_data = pd.read_csv(test_dir)


def preprocess(df):
    df = df.copy()
    
    df_drop_corr = df.drop('8', axis=1)
    return df_drop_corr

train_data = preprocess(train_data)
test_data = preprocess(test_data)

X = train_data.iloc[:, 0:-1]
y = train_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123)

dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
y_train_pred = dt.predict(X_train)
print("RMSE in train data:", mean_squared_error(y_train, y_train_pred, squared=False))

# Predictions 
y_test_pred = dt.predict(X_test)
print("RMSE in train data:", mean_squared_error(y_test, y_test_pred, squared=False))

def plot_predicted_vs_true(y_test, y_pred, sort=True):
    """
    Plot the results from a regression model in a plot to compare the prediction vs. acutal values
    
    Args:
        y_test : actual values
        y_pred : model predictions
        sort (bool, optional): Sort the values. Defaults to True.
    """
    # Create canvas
    plt.figure(figsize=(20, 5))
    
    t = pd.DataFrame({"y_pred": y_pred, "y_test": y_test})
    if sort:
        t = t.sort_values(by=["y_test"])

    plt.plot(t["y_test"].to_list(), label="True", marker="o", linestyle="none")
    plt.plot(
        t["y_pred"].to_list(),
        label="Prediction",
        marker="o",
        linestyle="none",
        color="purple",
    )
    plt.ylabel("Value")
    plt.xlabel("Observations")
    plt.title("Predict vs. True")
    plt.legend()
    plt.show()


plot_predicted_vs_true(y_test, y_test_pred)


# save the model to disk
modelname = 'dtree.pkl'
model_path = os.path.join(model_dir, modelname)
pickle.dump(dt, open(model_path, 'wb'))

print(model_path)