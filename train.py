
import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt


def gradient_descent(X, y, T): 
  N, d = X.shape
  #w = w_sum = np.ones(d)/d
  w = w_sum = np.zeros(d)

  dim = len(w)
  lhood = np.zeros(T)
  lambda_ = 100/T
  mb = 32

  y_ = y.copy()
  y_[y_ == 0] = -1

  # launch sklearn gradient descend algo

  for t in range(T):
    preds = sigmoid(np.dot(X,w)) - y
    preds = preds.reshape(-1,1)
    grad = np.dot((np.ones(N)/N), np.multiply(np.tile(preds, (1,d)), X) )

    w_avg = w_sum/(t + 1) 
    lhood[t] = np.log(1 + np.exp(np.multiply(-y_, np.dot(X,w_avg)))).mean()
    
    w = w - lambda_*np.ravel(grad)
    #w = w/np.sum(w)
    w_sum = w_sum + w

  w_avg = w_sum/T
  plt.plot(lhood)
  plt.show()
  
  return w_avg

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def ising_learn(Xs, graph_size, num_samples, num_iter, seed):
  s = graph_size
  
  # logistic approach
  A_hat = np.zeros((s,s))

  for i in range(s):
    start_time = time.time()
    print('Gradient Descent: Node number', i, 'out of', s)
    y = (Xs[:,i]+1)/2
    X = np.delete(Xs,i,axis = 1)
    X = np.concatenate((X,np.zeros((num_samples,1))), axis = 1)
    w = gradient_descent(X, y, num_iter)
    #print(s, w.shape)
    w = w[:s-1]#(w[:s-1] - w[s-1:2*s-2])

    A_hat[i, :i] = w[:i]/2
    A_hat[i, i+1:] = w[i:]/2 

    print('Executed Time:', time.time() - start_time)  

  A_gd = A_hat.copy()

  '''
  A_hat = np.zeros((s,s))
  
  for i in range(s):
    #print('Mirror Descent: Node number', i)
    y = (Xs[:,i]+1)/2
    X = np.delete(Xs,i,axis = 1)
    X = np.concatenate((X,-X,np.zeros((num_samples,1))), axis = 1)*W1
    w = mirror_descent(X, y, num_iter, W1)
    w = (w[:s-1] - w[s-1:2*s-2])*W1

    A_hat[i, :i] = w[:i]/2
    A_hat[i, i+1:] = w[i:]/2


  A_md = A_hat.copy()
  '''

  return A_gd

"""# Part 1"""

data1 = pd.read_csv('labeld_specimen_001_new.csv')
X = data1[['is_ccr7', 'is_cd27', 'is_cd28', 'is_cd45ra', 'is_cd57', 'is_cd279', 'is_KLRG1']].values
print(X.shape)
X[X == 0] = -1
num_samples , graph_size = X.shape
num_iter = 10000
seed = 42
A_gd = ising_learn(X, graph_size, num_samples, num_iter, seed)

plt.imshow(A_gd)

np.save('A_lean_001_new.npy', A_gd)


