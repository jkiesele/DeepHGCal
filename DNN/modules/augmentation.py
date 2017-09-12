

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


def augmentRotationalSymmetry8(a):
    
    b=rotate3Dimages_Right(a)
    c=rotate3Dimages_Right(b)
    d=rotate3Dimages_Right(c)
    ma=mirror3Dimages_leftRight(a)
    mb=mirror3Dimages_leftRight(b)
    mc=mirror3Dimages_leftRight(c)
    md=mirror3Dimages_leftRight(d)
    
    
    return np.concatenate((a,b,c,d,ma,mb,mc,md))

def duplicate8(a):
    return np.concatenate((a,a,a,a,a,a,a,a))

def evaluate8(func,arg):
    return np.concatenate((func(arg),func(arg),func(arg),func(arg),func(arg),func(arg),func(arg),func(arg)))
    
    
    