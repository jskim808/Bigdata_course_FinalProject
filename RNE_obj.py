
def RNE_obj(X, m, numNeighbor):
    import pywt
    
    n,d = [X.shape[i] for i in range(2)]
    W = LLE(X, numNeighbor)
    A = np.dot((np.identity(n)-W.T),X)
    AA = np.dot(A.T,A)
    AAplus = 0.5*(abs(AA)+AA)
    AAsubtract = 0.5*(abs(AA)-AA)
    H = np.random.rand(d,m)
    M = np.dot(A, H)
    Y = np.zeros([n, m])
    mu = 1.1
    gamma = 10
    max_gamma = 10**10
    alpha = 10**3
    iter_num = 50
    iter_numH = 30
    eps = 2.2204*(10**(-16))
    
    AM = np.dot(A.T, M)
    AY = np.dot(A.T,Y)
    AMplus = 0.5*(abs(AM)+AM)
    AMsubtract = 0.5*(abs(AM)-AM)
    AYplus = 0.5*(abs(AY)+AY)
    AYsubtract = 0.5*(abs(AY)-AY)

    obj = np.empty((1,iter_num), dtype=float)

    for iterTotal in range(1,iter_num+1):
        AM = np.dot(A.T, M)
        AY = np.dot(A.T,Y)
        AMplus = 0.5*(abs(AM)+AM)
        AMsubtract = 0.5*(abs(AM)-AM)
        AYplus = 0.5*(abs(AY)+AY)
        AYsubtract = 0.5*(abs(AY)-AY)
        # update H
        for i in range(1,iter_numH+1):
            G1 = np.diag(np.sqrt(1/np.diagonal(np.dot(H.T,H))+eps))
            H = np.dot(H, G1)
            H = np.multiply(H, np.sqrt(np.divide(((alpha*H) + (gamma*AMplus) + (gamma*np.dot(AAsubtract,H)) + AYplus), (alpha*np.dot(H, np.dot(H.T,H)) + gamma*AMsubtract + gamma*np.dot(AAplus, H) + AYsubtract +eps))))
        # update M
        temp = np.dot(A, H) - Y/gamma
        M = pywt.threshold(np.dot(A, H) - Y/gamma, 1/gamma, mode='soft', substitute=0)
        # update Y
        Y = Y+gamma*(M-np.dot(A,H))
        # update gamma
        gamma = min(mu*gamma, max_gamma)
        # obj
        obj[0][iterTotal-1] = sum(abs(np.dot(A,H))).sum()+1/4*alpha*np.linalg.norm(np.dot(H.T,H)-np.identity(m))**2
        if iterTotal % 10 == 0:
            print(f'trainin m = {m}, {iterTotal} epoch are finished')
    tempVector = (np.array(H)**2).sum(axis=1)
    I = np.argsort(tempVector)[::-1][:m]+1
    print(f'training m = {m} are finished')
    return I, obj
