import numpy as np
from scipy import optimize
class Neural_Network(object):
    def __init__(self):   
        
        
        self.layerInput = 2
        
        self.layerHidden1 = 4
        self.layerHidden2 = 5
        self.layerHidden3 = 3

        self.layerOutput = 1

        self.W1 = np.random.randn(self.layerInput,self.layerHidden1)
        self.W2 = np.random.randn(self.layerHidden1,self.layerHidden2)
        self.W3 = np.random.randn(self.layerHidden2,self.layerHidden3)
        self.W4 = np.random.randn(self.layerHidden3,self.layerOutput)
        
    def forward(self, X):
        
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        self.a3 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.W3)
        self.a4 = self.sigmoid(self.z4)
        self.z5 = np.dot(self.a4, self.W4)
        y_hat = self.sigmoid(self.z5)
        return y_hat

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def dsigmoid(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
        
    def relu(self,x):
        return x * (x > 0)
    
    def drelu(self,x):
        return 1. * (x > 0)
        

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta5 = np.multiply(-(y-self.yHat), self.dsigmoid(self.z5))
        dJdW4 = np.dot(self.a4.T, delta5)
        
        delta4 = np.dot(delta5, self.W4.T)*self.dsigmoid(self.z4)
        dJdW3 = np.dot(self.a3.T, delta4)  

        delta3 = np.dot(delta4, self.W3.T)*self.dsigmoid(self.z3)
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.dsigmoid(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        
        return dJdW1, dJdW2, dJdW3, dJdW4

        #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel(), self.W3.ravel(), self.W4.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.layerHidden1 * self.layerInput
        self.W1 = np.reshape(params[W1_start:W1_end], (self.layerInput , self.layerHidden1))
        W2_end = W1_end + self.layerHidden1*self.layerHidden2
        self.W2 = np.reshape(params[W1_end:W2_end], (self.layerHidden1, self.layerHidden2))
        W3_end = W2_end + self.layerHidden2*self.layerHidden3
        self.W3 = np.reshape(params[W2_end:W3_end], (self.layerHidden2, self.layerHidden3))
        W4_end = W3_end + self.layerHidden3*self.layerOutput
        self.W4 = np.reshape(params[W3_end:W4_end], (self.layerHidden3, self.layerOutput))
        
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2, dJdW3, dJdW4 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel(), dJdW3.ravel(), dJdW4.ravel()))
        


class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 1000000000000, 'disp' : True, "gtol":1e-9}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res


