import numpy as np
import pandas as pd


data_df = pd.read_csv('Titanic.csv')

data_df['age'] = data_df['age'].fillna(int(np.mean(data_df['age'])))

data_df.to_csv('Titanic.csv', index=False)
