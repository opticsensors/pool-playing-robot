import numpy as np


def get_file_from_data():
    raise NotImplementedError

def get_row_combinations_of_two_arrays(array1,array2):
    if len(array1.shape)==1:
        array1=array1.reshape(1,2)

    if len(array2.shape)==1:
        array2=array2.reshape(1,2)

    a = np.repeat(array1, array2.shape[0], axis=0)
    b = np.tile(array2, (array1.shape[0],1))
    result = np.hstack([a,b])

    return result

def get_equidistant_points(p1, p2, parts):
    if parts==0:
        points=(p1+p2)/2
    else:
        points_separated=np.linspace(p1[0], p2[0], parts+1),np.linspace(p1[1], p2[1], parts+1)
        points=np.column_stack((points_separated[0],points_separated[1]))
    return points

def angle_between_two_vectors(u,v):
    dot = u[:,0]*v[:,0] + u[:,1]*v[:,1] # equivalent to np.sum(u*v, axis=1)
    cosine_angle = dot / (np.linalg.norm(u, axis=1)* np.linalg.norm(v, axis=1))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def line_intersect(a1, a2, b1, b2):
    T = np.array([[0, -1], [1, 0]])
    da = np.atleast_2d(a2 - a1)
    db = np.atleast_2d(b2 - b1)
    dp = np.atleast_2d(a1 - b1)
    dap = np.dot(da, T)
    denom = np.sum(dap * db, axis=1)
    num = np.sum(dap * dp, axis=1)
    return np.atleast_2d(num / denom).T * db + b1

def get_row_combinations_of_two_arrays(array1,array2):
    if len(array1.shape)==1:
        array1=array1.reshape(1,2)
    if len(array2.shape)==1:
        array2=array2.reshape(1,2)
    a = np.repeat(array1, array2.shape[0], axis=0)
    b = np.tile(array2, (array1.shape[0],1))
    result = np.hstack([a,b])

    return result