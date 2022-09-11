import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

#Q1
np.__version__
#Q2
df.shape[0]
#or
len(df.index())
#Q3
df['make'].value_counts().head(3)
#Q4
len(df.loc[df['make'] == "audi"]["model"].value_counts())
#Q5
df.isnull().sum() #más o menos
#Q6.
#median
df['engine_cylinders'].median()
#most frequent
df['engine_cylinders'].mode()
result = df['engine_cylinders'].fillna(df['engine_cylinders'].mode().iloc[0])
result.median()
#Q7
lotus=df[['engine_hp','engine_cylinders']][df['make']=="lotus"]
lotus=lotus.drop_duplicates()
X=lotus.to_numpy()
#or X=lotus.__array__()
XTX = X.T @ X
XTXI = np.linalg.inv(XTX)
y = np.array([1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800])

res = XTXI @ X.T @ y
