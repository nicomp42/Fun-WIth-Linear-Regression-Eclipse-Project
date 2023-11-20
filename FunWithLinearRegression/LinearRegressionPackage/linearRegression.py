# LinearRegression.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def linearRegression():
    X = 30 * np.random.random((20, 1))
    # Just take the default mu and sigma (mean and std dev)
    y = 0.5 * X + 1.0 + np.random.normal(size=X.shape)
    #for i in range(0, len(X)):
    #    print(X[i], ", ", y[i])
    
    model = LinearRegression()
    model.fit(X, y)
    # Coefficient of Determination: best score is 1.
    print(model.score(X, y))

    X_new = np.linspace(0, 30, 100)
    y_new = model.predict(X_new[:, np.newaxis])
    #for i in range(0, len(X_new)):
    #    print(X_new[i], ", ", y_new[i])
    
    fig, axs = plt.subplots(1, 1, figsize=(9, 9))
    fig.suptitle("Linear Regression of random data.\nTraining data in green.\nRegression line in blue.")

    axs.scatter(X, y, color = "g", s = 99)

    axs.plot(X_new, y_new)

    # Clean up the plot and display it
    axs.set_xlabel('x')
    axs.set_ylabel('y')
    
    axs.axis('tight')
    
    plt.show()

    