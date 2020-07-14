import numpy as np 
from pandas import read_csv
import os
import time

'''
Now that this is working, I wan't to go through the documentation of np.dot() and figure out the linear algebra
behind how that works and implement in that way. 
'''

class simple_linear_regressor:

    def __init__(self, X, y, i=1000, lr=.05):
        self.features = X
        self.labels = y
        self.iterations = i
        self.weight = 2
        self.bias = 2
        self.learning_rate = lr
        self.data_length = len(self.features)
    
    def fit_model(self):
        for i in range(self.iterations):
            self.update()

    def predict(self, x):
        return (self.weight * x) + self.bias

    def weight_derivitive(self):
        derivitive = -2*self.features*(self.labels - (self.weight*self.features+self.bias))
        return np.sum(derivitive)/self.data_length

    def bias_derivitive(self):
        derivitive = -2*(self.labels - (self.weight*self.features+self.bias))
        return np.sum(derivitive)/self.data_length

    def update(self):
        self.weight = self.weight - self.learning_rate*(self.weight_derivitive())
        self.bias = self.bias - self.learning_rate*(self.bias_derivitive())

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ## opening csv file
    cur_directory = os.getcwd()
    weather_path = os.path.join(cur_directory, 'regression_data', 'weatherHistory.csv')
    weather_data = read_csv(weather_path)
    #print(weather_data[:5])

    ## getting random distribution of desired data
    humitity_X = np.array(weather_data['Humidity'])
    temp_y = np.array(weather_data['Temperature (C)'])

    np.random.seed(seed=1)
    rand_indexs = np.array(np.floor(np.random.rand(150) * len(humitity_X)), 'int')
    #print(rand_indexs, '\n')

    rand_X = humitity_X[rand_indexs]
    #print(rand_X[:3], '\n')

    rand_y = temp_y[rand_indexs]
    #print(rand_y[:3], '\n')

    ## Visualizing the data
    plt.scatter(rand_X, rand_y, color = 'red')
    plt.plot()
    plt.title('Relationship of Humidity and Temperature')
    plt.xlabel('Humidity')
    plt.ylabel('Temperature')


    ## Using the 
    start_time = time.time()
    regressor = simple_linear_regressor(rand_X, rand_y, 1500)
    regressor.weight_derivitive()
    regressor.fit_model()
    total_time = time.time() - start_time
    print(total_time)


    relavant_range = np.arange(.3, 1.1, .1)
    y_vals = relavant_range * regressor.weight + regressor.bias

    plt.plot(relavant_range, y_vals)
    plt.show()
    print(f'{regressor.weight}x + {regressor.bias}')
