import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

#not in Notebook
df.columns = df.columns.str.lower().str.replace(' ', '_')
string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

#not in Notebook

#Q1
print(np.__version__)
#Q2
print(df.shape[0])
#or
print(len(df.index))
#Q3
print(df['make'].value_counts().head(3))
#Q4
print(len(df.loc[df['make'] == "audi"]["model"].value_counts()))
#Q5
df.isnull().any()
#Q6.
#median
print(f"Median before changes: {df['engine_cylinders'].median()}")
#most frequent
print(f"Most frequente value (mode): {df['engine_cylinders'].mode()}")
result = df['engine_cylinders'].fillna(df['engine_cylinders'].mode().iloc[0])
print(f"Median after changes: {result.median()}")

#Q7
lotus=df[['engine_hp','engine_cylinders']][df['make']=="lotus"]
lotus=lotus.drop_duplicates()
X=lotus.to_numpy()
#or X=lotus.__array__()
XTX = X.T @ X
XTXI = np.linalg.inv(XTX)
y = np.array([1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800])

res = XTXI @ X.T @ y

print(f"XTXI*X.T*y[0] = {res[0]}")
