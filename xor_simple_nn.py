import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

parser = argparse.ArgumentParser(description='Xor back propagation')
parser.add_argument('-d', type=int, help='how many layers')
parser.add_argument('-r', type=float, help='your learning rate')

test_x = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])
test_y = np.array([0, 1, 1, 0])

def sigmoid(a):
    return 1/(1+np.exp(-a))

if __name__ == '__main__':
    args = parser.parse_args()
    if args.d and args.d > 3:
        l = args.d
    else:
        l = 3
    
    if args.r:
        lr = args.r
    else:
        lr = 0.2
    
    #initialize
    x = test_x.T
    t = test_y.T
    w = np.random.rand(l-2, x.shape[0], x.shape[0]+1)
    lw = np.random.rand(x.shape[0]+1)

    every = 500
    count = 0
    
    #store x in the first layer
    h = np.full((l-1, x.shape[0]+1, x.shape[1]), 1.)
    y = np.empty(x.shape[1])
    h[0, :-1] = x
    error = []
    
    while True:
        #forward
        for i in range(1, l):
            if i ==l-1:
                y = sigmoid(lw.dot(h[i-1]))
            else:
                h[i, :-1] = sigmoid(w[i-1].dot(h[i-1]))

        error.append(log_loss(t, y))
        if error[-1] < 0.001 or count > 40000:
            break
        #backward
        for i in range(l-1, 0, -1):
            if i == l-1:
                #using sigmoid output and binary entropy, so dE/da = y - t
                g = y - t
            else:
                g = h[i, :-1]*(1-h[i, :-1])*g[:-1]
            
            #get gradient of w and bias
            dw = g.dot(h[i-1].T)
            
            #update w and bias, and pass the gradient to parent layer
            if i == l-1:
                g = np.outer(lw, g)
                lw -= lr*dw
            else:
                g = w[i-1].T.dot(g)
                w[i-1] -= lr*dw
        
        count += 1
        if count % every == 0:
            print "epochs:", count
            
    for pair in zip(test_x, y):
        print "for x =", pair[0], ", y =", pair[1]

    plt.plot(error)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    

    