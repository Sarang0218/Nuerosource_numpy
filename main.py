import numpy as np

from nn import Neural_Network, trainer

inp = []
out = []
for i in range(100):
    for j in range(50):
        inp.append([i,j])
        out.append([i+j])


        
X = np.array(inp, dtype=float)
y = np.array(out, dtype=float)

xmax = np.amax(X, axis=0)
ymax = np.amax(y)
X = X/xmax
y = y/ymax

NN = Neural_Network()
cost1 = NN.costFunction(X,y)
dJdW1, dJdW2, dJdW3, dJdW4 = NN.costFunctionPrime(X,y)
T = trainer(NN)
T.train(X,y)

print(NN.costFunction(X,y))

while True:
    inp = input("입력: ")
    nums = inp.split(" ")
    print(ymax*NN.forward( np.array([int(nums[0]),int(nums[1])])/xmax ))