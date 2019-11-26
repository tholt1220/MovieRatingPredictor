import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    train_df = pd.read_csv("movieratepredictions/train_ratings.csv")
    val_df = pd.read_csv("movieratepredictions/val_ratings.csv")
    val_size = len(val_df.index)
    val_proportion = val_size / (val_size + len(train_df.index))

    combined_df = train_df.append(val_df)
    combined_df.to_csv("combined_ratings.csv", index=False, header=False)

    for i in range(3):
        new_train_df, new_val_df = train_test_split(combined_df, test_size=val_proportion)
        new_train_df.to_csv('train_ratings{}.csv'.format(i), index=False)
        new_val_df.to_csv('val_ratings{}.csv'.format(i), index=False)

