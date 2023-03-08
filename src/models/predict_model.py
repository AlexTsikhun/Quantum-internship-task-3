
import os
import pandas as pd
import pathlib
import pickle

data_dir = pathlib.Path(__file__).parents[2].joinpath(
    'data/raw/')
model_dir = pathlib.Path(__file__).parents[2].joinpath(
    'models/')

test_dir = os.path.join(data_dir, "internship_hidden_test.csv")

# print(test_dir)

test_data = pd.read_csv(test_dir)

def preprocess(df):
    df = df.copy()
    
    df_drop_corr = df.drop('8', axis=1)
    return df_drop_corr

test_data = preprocess(test_data)


modelname = 'dtree.pkl'
model_path = os.path.join(model_dir, modelname)
# load the model from disk
loaded_model = pickle.load(open(model_path, 'rb'))

# make predictions
predictions = loaded_model.predict(test_data)

output_name = 'predictions_dtree.csv'
output_path = os.path.join(data_dir, output_name)
final_df = pd.DataFrame(predictions, columns=['Predictions'])
final_df.to_csv(output_path)
print(final_df)