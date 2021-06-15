def ClusterAccMea(T,idx):
    # Step 1: get the index of each cluster of the target index
    k1=len(np.unique(T))
    n1=len(T)
    T_min = np.min(T)
    a__=[]
    for i in range(T_min,T_min+k1):
        temp=np.where(T==i)
        a__.append(temp)
        
    # Step 2: get the index of the learned cluster
    k2 = len(np.unique(idx));n2 = len(idx)
    b__=[]
    idx_min = np.min(idx)
    if n1!=n2:
        print('These two indices do not match!')
    for i in range(idx_min,idx_min+k2):
        temp=np.where(idx==i)
        b__.append(temp)

    # Step 3: compute the cost matrix of these two indices
    disMat = np.zeros((k1,k2))
    for i in range(0,k1):
        for j in range(0,k2):
            disMat[i,j] = len(np.intersect1d(a__[i],b__[j])) # cost matrix 

    # Step 4: munkres algorithm
    m=Munkres()
    indexes = m.compute(-disMat)
    total = 0
    for row, column in indexes:
        value = disMat[row][column]
        total += value
        
    # step 5: compute the cluster accuracy
    Acc = total/n1
    return Acc
