import numpy as np
from scipy import sparse
from scipy.sparse import linalg as s_linalg

try:
    from itertools import izip as zip
except:
    pass

class EchoStateNetwork:
    def __init__(self, **kwargs):
        self.__dict__ = {
            "hidden_len":1000,
            "density":0.1,
            "reg":1.0e-6,
            "steps_ahead":1,
            "skip_steps":200,
            "decay":0.3,
            "scale":0.8,
            "batch_size":10,
            "spec_rad":0.9,
            "momentum":0.3,
            "learning_rate":1.0e-5
        }
        self.__dict__.update(kwargs)

        self.W, self.U, self.h, self.M, self.g = None, None, None, None, None
        
    def fit(self, X_gen, y_gen, reset=False):
        if self.W is None:
            self.W = sparse.rand(self.hidden_len, self.hidden_len, density=self.density)
            self.W = self.W*np.real(self.spec_rad/max(s_linalg.eigs(self.W)[0]))
        
        if self.h is None or reset:
            self.h = np.random.randn(self.hidden_len, 1)
        
        for i, (x, y) in enumerate(zip(X_gen, y_gen)):
            x = np.hstack(([1], x)) 
            if self.U is None:
                self.U = self.scale*sparse.rand(self.hidden_len, x.size, density=self.density) 
            if self.M is None:
                self.M = np.random.randn(y.size, self.hidden_len)
            if self.g is None:
                self.g = np.zeros((y.size, self.hidden_len))
            
            self.h = (1 - self.decay)*self.h + self.decay*np.tanh(self.W.dot(self.h) + self.U.dot(x.reshape(-1, 1)))
            self.g += self.h.T*(self.M.dot(self.h) - y.reshape(-1, 1)) + self.reg*self.M
            if (i+1) % self.batch_size == 0:
                self.M -= self.learning_rate*self.g
                self.g *= self.momentum
        
    def predict(self, X_gen, reset=False):
        if reset:
            self.h = np.random.randn(self.hidden_len, 1)
        
        for x in X_gen:
            x = np.hstack(([1], x)) 
            self.h = (1 - self.decay)*self.h + self.decay*np.tanh(self.W.dot(self.h) + self.U.dot(x.reshape(-1, 1)))
            yield self.M.dot(self.h)


if __name__ == '__main__':
    f = lambda x: np.sin(x + np.pi/3)
    
    X = np.cos(np.arange(-500, 500, 0.01).reshape(-1, 1))
    y = f(X).reshape(-1, 1)
    esn = EchoStateNetwork()
    esn.fit(X, y)
    X_test = np.cos(np.arange(500, 550, 0.01).reshape(-1, 1))
    y_pred = np.array(list(esn.predict(X_test)))
    
    import matplotlib.pyplot as plt
    plt.plot(y_pred.flatten())
    plt.plot(f(X_test))
    plt.legend(['y_pred', 'y_true'])
    plt.show()


    
