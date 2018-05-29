import numpy as np
import warnings



def euclidean_two_datasets(A, B):
    """
    Returns euclidean distance between two datasets

    A is first dataset in form (N,F) where N is number of examples in first dataset, F is number of features
    B is second dataset in form (M,F) where M is number of examples in second dataset, F is number of features

    Returns:
    A matrix of size (N,M) where each element (i,j) denotes euclidean distance between ith entry in first dataset and jth in second dataset

    """
    A = np.array(A)
    B = np.array(B)
    return (-2*A.dot(B.transpose()) + (np.sum(B*B,axis=1)) + (np.sum(A*A,axis=1))[:,np.newaxis])


A = np.random.randn(100,3)*100-50
B = np.random.randn( 80,3)*100-50
A[80:100,:] = 0
B[60:80,:] = 0
# B = np.copy(A)
np.seterr(all='print')
with open('super.np') as f:
    A = np.reshape(np.fromfile(f, dtype=np.float32), (-1))

np.seterr(all='raise')

A[A>1]=0
A[A<0]=0


A[:] = 0
np.sqrt(A)

for i in range(10000):
    B = np.zeros(10000)
    B[0] = A[i]
    np.sqrt(B)

0/0

for i in range(2000):
    print(i)
    # for j in range(2000):
    temp = np.sqrt(np.array(A[i]))


0/0
A = A.astype(np.float64)

A = A[0:100,:]
B = A.copy()

Z = np.zeros((len(A),len(B)))
print(np.shape(A))

for i in range(len(A)):
    for j in range(len(B)):
        Z[i,j] = np.sqrt((A[i][0] - B[j][0])**2 + (A[i][1] - B[j][1])**2 + (A[i][2] - B[j][2])**2)
#         Z[i, j] = -2*(A[i][0] * B[j][0] + A[i][1] * B[j][1] + A[i][2] * B[j][2])

X = euclidean_two_datasets(A, B)
X = np.sqrt(X)

print(np.shape(X))


print(np.allclose(Z,X, atol=1.e-7))

print(np.sum(A<0))