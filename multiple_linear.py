import numpy as np


class MultipleLinearRegressor:

    def __init__(self, lr=.05):
        self.lr = lr
        self.n_examples = None
        self.model = None

    def fit(self, X, y):
        self.n_examples = X.shape[0]
        bias = np.ones((self.n_examples, 1))

        # add theta_0 to design matrix
        design_matrix = np.append(bias, X, axis=1)

        self.model = self.normal_eq(design_matrix, y)

    def normal_eq(self, X, y):
        X_transpose_dot_X = np.dot(X.T, X)
        var1 = np.linalg.inv(X_transpose_dot_X)
        var2 = np.dot(X.T, y)
        # var1 should 6x6
        # var2 should 6x1
        theta_vector = np.dot(var1, var2)
        return theta_vector

    def predict(self, X):
        return np.dot(X, self.model)

    def test(self, X, y):
        # R squared score
        theta_0 = np.ones((X.shape[0], 1))
        X = np.append(theta_0, X, axis=1)

        top = np.sum(np.square(self.predict(X) - y))
        bottom = np.sum(np.square(y - np.mean(y)))
        return 1 - (top / bottom)

if __name__ == '__main__':
    import os
    import pandas as pd 

    current_directory = os.getcwd()
    housing_data_path = os.path.join(current_directory, 'regression', 'regression_data', 'USA_Housing.csv')
    housing_dataframe = pd.read_csv(housing_data_path) 
    housing_dataframe.drop('Address', axis=1, inplace=True)
    # print(housing_dataframe.head()) # Shows the first five entries of the dataframe and their corresponding labels

    df_to_array = housing_dataframe.values # each row in array is row accross data frame 
    # print(df_to_array)

    ## Spliting data into training and testing
    np.random.shuffle(df_to_array)
    train, test = df_to_array[:3750, :], df_to_array[3750:, :]

    ## Spliting data into features and labels
    X_test = test[:, :-1]
    X_train = train[:, :-1]
    y_train = train[:, -1]
    y_test = test[:, -1]

    ## Test data
    regressor = MultipleLinearRegressor()
    regressor.fit(X_train, y_train)
    print(regressor.test(X_test, y_test))
