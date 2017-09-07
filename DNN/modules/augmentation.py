

import numpy as np


def mirror3Dimages_leftRight(a):
    '''
    flips with standard axis [entry][x][y][z][feature]
    '''
    return np.flip(a,1)


def mirror3Dimages_topBottom(a):
    '''
    flips with standard axis [entry][x][y][z][feature]
    '''
    return np.flip(a,2)


def rotate3Dimages_Right(a):
    '''
    flips with standard axis [entry][x][y][z][feature]
    '''
    return np.flip(np.swapaxes(a,1,2),2)


def augmentRotationalSymmetry6(a):
    mapmirrlr=mirror3Dimages_leftRight(a)
    mapmirrtb=mirror3Dimages_topBottom(a)
    maprot1=rotate3Dimages_Right(a)
    maprot2=rotate3Dimages_Right(maprot1)
    maprot3=rotate3Dimages_Right(maprot2)
    
    return np.concatenate((a,mapmirrlr,mapmirrtb,maprot1,maprot2,maprot3))

def duplicate6(a):
    return np.concatenate((a,a,a,a,a,a))

def evaluate6(func,arg):
    return np.concatenate((func(arg),func(arg),func(arg),func(arg),func(arg),func(arg)))
    