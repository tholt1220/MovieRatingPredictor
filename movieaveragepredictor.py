"""
    movieaveragepredictor
    ~~~~~~~~~~~~~
    Given a movie, predicts that movie's average rating. Predicts global average (3.5) if no previous
    movie data exists

"""

import constants
import numpy as np
import pandas as pd
import util

class MovieAveragePredictor():
    def __init__(self):
        self.df = None

    def fit(self, append_val_data=False, show_validation_results=False):
        """
        append_val_data will use both the training and validation data to make predictions
        """

        X, y = util.get_training_data()
        self.df = pd.concat([X, y], axis=1, sort=False)

        if append_val_data:
            X_val, y_val = util.get_validation_data()
            df_val = pd.concat([X_val, y_val], axis=1, sort=False)
            self.df = self.df.append(df_val)

        # we dont need userId for this classifier
        self.df = self.df.drop("userId", axis=1)
        self.df = self.df.groupby(['movieId'])['rating'].agg(lambda x: x.mean())
        # df = df.set_index("movieId")

        print(len(X))

        # y_pred = X.merge(df, left_on="movieId", right_on="movieId", how="left").fillna(constants.AVG_RATING)
        # y_pred = y_pred.drop("rating")

        if show_validation_results:
            X, y = util.get_validation_data()
            y_pred = np.full(len(X), constants.AVG_RATING)

            X = X.values
            for i, row in enumerate(X):
                if row[1] in self.df.index:
                    y_pred[i] = self.df.loc[row[1]]

                # if i % 1000 == 0:
                #     print("Iteration {}".format(i))

            mse = util.compute_mse(y_pred, y)
            print("Validation mse is {}".format(mse))

    def predict(self):
        test_df = util.get_test_data()
        test_df['rating'] = constants.AVG_RATING
        test_df = test_df.drop('userId', axis=1)

        for index, row in test_df.iterrows():
            if row['movieId'] in self.df.index:
                test_df.at[index, 'rating'] = self.df.loc[row['movieId']]

        test_df = test_df.drop('movieId', axis=1)

        test_df.to_csv('movieaveragepredictor.csv', index=False)

if __name__ == "__main__":
    movieavgpredictor = MovieAveragePredictor()
    movieavgpredictor.fit(append_val_data=True)
    movieavgpredictor.predict()
