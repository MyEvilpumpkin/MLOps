import pandas as pd


data_df = pd.read_csv('Titanic.csv')

data_df = pd.get_dummies(data_df, columns=['sex'], drop_first=True)

data_df.to_csv('Titanic.csv', index=False)
