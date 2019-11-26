import pandas as pd
import sys

def get_dataframes(filePath, yname="rating"):
    dataframe = pd.read_csv(filePath)

    if not yname:
        return dataframe

    y = dataframe[yname]
    x = dataframe.drop(yname, axis=1)

    return x, y

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])

    df = df.drop('Id', axis=1)
    df['Rating'] = 3.5
    df.to_csv('test_ratings_triple.csv', header=False, index=False)

