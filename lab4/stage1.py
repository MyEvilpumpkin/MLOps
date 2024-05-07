import pandas as pd


data_df = pd.read_csv('Titanic.csv')

data_df = data_df[['pclass', 'sex', 'age', 'target']]

data_df.to_csv('Titanic.csv', index=False)
