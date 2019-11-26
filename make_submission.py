import pandas as pd

df = pd.read_csv("results.txt", usecols=[0], names=['rating'], header=None)

df.insert(0, 'Id', range(0, len(df)))

df.to_csv('submission.csv', index=False)