#pip install -r requirements.txt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
class LocallyWeightedRegression:
    def __init__(self, tau=0.01):
        self.tau = tau

    def kernel(self, query_point, X):
        n = len(X)
        Weight_matrix = np.mat(np.eye(n))
        
        for idx in range(n):
            diff = X[idx] - query_point
            Weight_matrix[idx, idx] = np.exp(-np.dot(diff, diff.T) / (2 * self.tau * self.tau))
        return Weight_matrix

    def predict(self, X, Y, query_point):
        q = np.mat([query_point, 1])
        X_ext = np.hstack((X, np.ones((len(X), 1))))
        W = self.kernel(q, X_ext)
        
        num_rows = min(X.shape[0], Y.shape[0])
        X_ext = X_ext[:num_rows]
        Y = Y[:num_rows]
        
        theta = np.linalg.pinv(X_ext.T * (W * X_ext)) * (X_ext.T * (W * Y))
        pred = np.dot(q, theta)
        return pred

    def fit_and_predict(self, X, Y):
        Y_test = []
        X_test = np.linspace(np.min(X), np.max(X), len(X))
        for x in X_test:
            pred = self.predict(X, Y, x)
            Y_test.append(pred[0, 0])
        return np.array(Y_test)

    def score(self, Y, Y_pred):
        return np.sqrt(np.mean((Y - Y_pred) ** 2))

    def fit_and_show(self, X, Y):
        Y_test = self.fit_and_predict(X, Y)
        
        sns.set_style('whitegrid')
        plt.title(f"The scatter plot for the value of tau = {self.tau:.5f}")
        plt.scatter(X, Y, color='red', label='Original data')
        plt.plot(np.linspace(np.min(X), np.max(X), len(Y_test)), Y_test, color='green', label='Predicted curve')
        plt.legend()
        plt.show()

dfx = pd.read_csv('./weightedX.csv')
dfy = pd.read_csv('./weightedY.csv')

X = dfx.values
Y = dfy.values

X = X.reshape(-1, 1)

num_rows = min(X.shape[0], Y.shape[0])
X = X[:num_rows]
Y = Y[:num_rows]

u = X.mean()
std = X.std()
X = (X - u) / std

tau = 0.2
model = LocallyWeightedRegression(tau)

Y_pred = model.fit_and_predict(X, Y)

model.fit_and_show(X, Y)