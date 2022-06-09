
import numpy as np

def simulate_AR(π, η, T = 2000, index = False):
    """simulate AR process"""
    
    assert π.shape[0] == π.shape[1],'Matrix π should be a squared matrix!!!!'
    assert np.all(π >= 0), 'All element in π should be positive!!!!'
    assert np.allclose(np.sum(π,axis=1),1 ), 'The sum of state probabilities are not 1!!!!'
    assert π.shape[0] == len(η), 'π and η should have consistente size!!!!'
    assert T>0 & isinstance(T, int) , 'T should be a positive integer!!!'
    
    si = len(η) # size of η
    ind = np.arange(T)
    A = np.linspace(0,1,si)
    ηt = np.zeros(T)
    
    for j in range(T-1):
        if j == 0:
            ind[j] = int(si/2) + 1
  
        πcu   = π[ind[j],:]  # current working row of π
    
        for i in range(len(A)):
            if i == 0:
                A[i] = πcu[i]
            else:
                A[i] = A[i-1] + πcu[i]
                
        x = np.random.uniform()
        for i in range(len(A)):
            c = x <= A[i] if i == 0 else A[i-1] < x <= A[i]
            if c:
                ind[j+1] = i
                break
                
    for t in range(T):
        ηt[t] = η[ind[t]]

    if index:
        return ηt, ind      
    else:
        return ηt

