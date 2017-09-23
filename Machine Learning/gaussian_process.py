import numpy as np
import matplotlib.pyplot as plt

def RBF(x1, x2):
    return np.exp(-np.linalg.norm(x1 - x2, axis=0)**2/2)

class GaussianProcess:
    def __init__(self, kernel=RBF, sigma=1e-2):
        self.kernel, self.sigma = kernel, sigma
    
    def fit(self, X, y):
        self.X, self.y = X, y.reshape(-1, 1)

    def predict(self, X, samples=1e2):
        step_size = np.std(self.y)/100.0
        dy = np.mean(self.y) + 100*step_size*np.random.randn(X.shape[0], 1)
        samples = int(samples * dy.size)
        X = np.vstack((self.X, X))
        self.K = self.kernel(X.T.reshape(X.shape[1], X.shape[0], 1),
                             X.T.reshape(X.shape[1], 1, X.shape[0])) + self.sigma**2*np.eye(X.shape[0])
        print(np.linalg.det(self.K)), samples
        total, total_sqr = np.zeros(dy.shape), np.zeros(dy.shape)
        E = -np.vstack((self.y, dy)).T.dot(np.linalg.solve(self.K, np.vstack((self.y, dy))))
        for i in range(samples):
            new_dy = dy + step_size*np.random.randn(*dy.shape)
            new_E  = -np.vstack((self.y, new_dy)).T.dot(np.linalg.solve(self.K, np.vstack((self.y, new_dy))))
            if np.exp(new_E - E) > np.random.rand():
                dy, E = new_dy, new_E
            total += dy
            total_sqr += dy**2
        return (total/samples).flatten(), (total_sqr/samples - (total/samples)**2).flatten()

x = np.arange(0, 10, 0.1)
y = np.sin(x) + 0.1*np.random.randn(*x.shape)

gp = GaussianProcess()
gp.fit(x.reshape(-1, 1), y)

y_pred, y_err = gp.predict(x.reshape(-1, 1), samples=1e3)

plt.plot(x, y)
plt.plot(x, y_pred)
plt.fill_between(x, y_pred - np.sqrt(y_err),
                 y_pred + np.sqrt(y_err),
                 alpha=0.2, color='k')
plt.legend(['True', 'Prediction'])
plt.show()




