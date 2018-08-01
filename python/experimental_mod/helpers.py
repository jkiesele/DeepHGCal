import numpy as np



def wrap_max(A,B):
    return np.fmod(B + np.fmod(A, B), B)

def wrap_min_max(A):
    return -np.pi + wrap_max(A-np.pi, 2*np.pi)

"""
Computes angle difference between numpy arrays
"""
def delta_angle(A,B):
    return wrap_min_max(A-B)

"""
Returns euclidean distance between two datasets

A is first dataset in form (N,F) where N is number of examples in first dataset, F is number of features
B is second dataset in form (M,F) where M is number of examples in second dataset, F is number of features

Returns:
A matrix of size (N,M) where each element (i,j) denotes euclidean distance between ith entry in first dataset and jth in second dataset

"""
def euclidean_two_datasets(A, B):
    A = np.array(A)
    B = np.array(B)
    return -2*A.dot(B.transpose()) + (np.sum(B*B,axis=1)) + (np.sum(A*A,axis=1))[:,np.newaxis]
