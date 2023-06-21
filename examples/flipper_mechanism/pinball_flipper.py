import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

def angle_abc(a,b,c):
    """
    Computes the angle between 3 points 
    (point b is the vertex)

    Parameters
    ----------    
        a,b,c: numpy array of shape (2,)
            x, y coordinates of the point
    Returns
    -------
        angle: numpy float64

    """
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def intersection_two_circles(P0,P1,r0,r1):
    """
    Computes the points of intersection (if any) between two
    circles of radiis r0,r1 and centers P0,P1
    
    """
    # distance from P0 to P1    
    d=np.sqrt((P0[0]-P1[0])**2+(P0[1]-P1[1])**2)
    # distance from P0 to P2 (being P2 the point of intersection between lines P0P1 and X1X2)
    # (X1,X2 are the intersection points that we want to find)
    a=(d**2+r0**2-r1**2)/(2*d)
    # distance from P1 to P2
    b=d-a
    # distance from P2 to X1 = distance from P2 to X2
    h=np.sqrt(r0**2-a**2)
    # convert points to numpy array
    P0_arr = np.array(P0)
    P1_arr= np.array(P1)
    P1P0_arr=P1_arr-P0_arr
    auxiliar_point=(b/d)*P0_arr+(a/d)*P1_arr

    intersec1_x=auxiliar_point[0]+(h/d)*P1P0_arr[1]
    intersec2_x=auxiliar_point[0]-(h/d)*P1P0_arr[1]

    intersec1_y=auxiliar_point[1]-(h/d)*P1P0_arr[0]
    intersec2_y=auxiliar_point[1]+(h/d)*P1P0_arr[0]

    point1=(intersec1_x,intersec1_y)
    point2=(intersec2_x,intersec2_y)

    return point1, point2

# mechanism constants
l1=23
l2=23.6
x_arr=np.arange(16.5,35.5,0.5)
C=(0,0)
d=24
a=51
b=38.16
prev_P=None
list_of_dict=[]
dict_to_save={}

for i,x in enumerate(x_arr):
    A=(-x,d)
    _,B=intersection_two_circles(A,C,l1,l2)    
    norm_BC_vector=math.sqrt(B[0]**2+B[1]**2)
    BC_vector=(-B[0]/norm_BC_vector, -B[1]/norm_BC_vector)
    perpendicular_BC_vector=(-B[1]/norm_BC_vector, B[0]/norm_BC_vector)    
    P=(a*BC_vector[0]+b*perpendicular_BC_vector[0], a*BC_vector[1]+b*perpendicular_BC_vector[1])
    angle=angle_abc(np.array(A),np.array(B),np.array(C))
    dict_to_save['x']= - x
    dict_to_save['Px']=P[0]
    dict_to_save['Py']=P[1]
    dict_to_save['angle']=angle

    if prev_P is not None:
        dict_to_save['displacement_P']=math.sqrt((P[0]-prev_P[0])**2+(P[1]-prev_P[1])**2)
        dict_to_save['displacements_Py']=P[1]-prev_P[1]
        dict_to_save['displacements_Px']=P[0]-prev_P[0]
    else:
        dict_to_save['displacement_P']=0
        dict_to_save['displacements_Py']=0
        dict_to_save['displacements_Px']=0

    list_of_dict.append(dict_to_save.copy())
    prev_P=P

    plt.plot([A[0], B[0]], [A[1], B[1]], 'r', linewidth=1.5)
    plt.plot([C[0], B[0]], [C[1], B[1]], 'b', linewidth=1.5)
    plt.plot([C[0], P[0]], [C[1], P[1]], 'g', linewidth=1.5)
    plt.scatter(A[0], A[1], color='r')
    plt.scatter(B[0], B[1], color='b')
    plt.scatter(C[0], C[1], color='g')
    plt.scatter(P[0], P[1], color='k')
    plt.pause(0.05)
    #plot_config(A,B,C,P)
    
plt.show()
df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
df.to_csv(path_or_buf=f'./data/mechanism.csv', sep=' ',index=False)

max_Py=df['Py'].max()
min_Py=df['Py'].min()
max_Px=df['Px'].max()
max_angle=df['angle'].max()

print('Py stroke', abs(max_Py-min_Py))
print('max Px', max_Px)
print('max angle', max_angle)