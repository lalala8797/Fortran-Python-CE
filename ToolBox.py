import numpy as np
def SortOrder(a):
    '''
    Input: a: array like
    
    Output:
            asort: sorted array
            order: new ordering of the array
    '''
    asort = np.sort(a)
    order = []
    for i in range(len(asort)):
        index = np.where(a == asort[i])
        order = np.append(order, index).astype(int)
        
    for j in range(len(order)):
        if j <= len(order) - 1:
            indaux = np.where(order == order[j])[0]
            if len(indaux) != 1 :
                order = np.delete(order, indaux[1:])
    
    return asort, order
