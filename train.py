import numpy as np
import pandas as pd
import math

def tanghyper(z):
    return (2 / (1 + np.exp(z))) - 1

def train():
   x = ini_swarm(41, 50, 300)
   print(x.shape)        
   #df = pd.DataFrame(x)
   #df.to_csv('ini_swarm.csv')

def fitness(Np, Nh, D, x, C, xe, ye):
    Mse = []
    w2 = []
    for i in range(Np):
        p = x[i]
        w1 = np.reshape(p, (Nh, D))
        H = Activation(xe, w1)
        w2.append(mlp_inv(H, ye, C))
        ze = w2[i] * H
        Mse.append(np.sqrt(mse(ye - ze)))
    return Mse, w2

def Activation(xe, w1):
    return 10

def mse(a):
    return a * 2

def mlp_inv(a, b, c):
    return "hola"

def config_swarm(Np, Nh, D, maxIter):
    x = ini_swarm(Np, Nh, D)
    dim = x.shape[2]
    pBest = np.zeros((Np, dim))
    pFitness = np.ones((1, Np)) * math.inf
    gFitness = math.inf
    wBest = np.zeros((1, Nh))
    alpha = generateAlpha(maxIter)
    return x, pBest, pFitness, gFitness, wBest,alpha

def ini_swarm(Np, Nh, D):
    x = []
    for i in range(Np):
        wh = rand_w(Nh, D)
        x.append(np.reshape(wh, (1, Nh * D)))
    # x2 = np.reshape(np.asarray(x), (Np, Nh * D))
    return np.asarray(x)

def rand_w(Nh, D):
    w = np.random.rand(Nh, D)
    r = np.sqrt(6 / (Nh + D))
    w = w * 2 * r - r 
    return w

def generateAlpha(m):
  max = np.full((m), m)
  for i in range(m):
    max[i] = max[i] - i
  return (0.95-0.2)*max/m + 0.2

train()