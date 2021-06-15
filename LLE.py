# -*- coding: utf-8 -*-
"""ML_LLE_RNE_obj.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18xAtOmWiWOWkQbfuHVko-_8xOijolfCx
"""

def LLE(X,K):
    import numpy as np
    from scipy.sparse import csr_matrix
    X = np.array(X.T)
    N  = X.shape[1]
    D = X.shape[0]

    # print(1,"-->Finding %d nearest neighbors.\n" %K)
    X2 = X**2
    X2 = X2.sum(axis=0)
    distance = np.tile(X2,(N,1))+np.tile(X2.reshape(-1,1),(1,N))-2*(X.T.dot(X))

    index=distance.argsort(axis=0)+1
    neighborhood = index[range(1,K+1),:]
    
    # print('-->Solving for reconstruction weights.')
    if K>D:
        print(1,'   [note: K>D; regularization will be used]')
        tol = 1 * (10**(-3))
    else:
        tol = 0
        
    W = np.zeros([K,N])
    for i in range(N):
        z = X[:,neighborhood[:,i]-1] - np.tile(X[:,i].reshape(-1,1),(1,K))
        C = np.dot(z.T, z)
        C = C+np.identity(K)*tol*np.trace(C)
        W[:,i] = np.dot(np.linalg.pinv(C), np.ones([K,1])).T
        W[:,i] = W[:,i] / np.sum(W[:,i])
    
    res = np.tile(np.array(range(1,N+1)),(K,1))
    
    return csr_matrix((list(W.reshape(1,-1)[0]), (list(res.reshape(1,-1)[0]-1), list(neighborhood.reshape(1,-1)[0]-1))), shape=(N,N))