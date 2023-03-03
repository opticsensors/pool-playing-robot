import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

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
    return angle

def intersection_circle_line(slope,intercept, r,center):
    """
    Computes the points of intersection (if any) between a line 
    and a circumference given a slope, an intercection, a radii 
    and a center.

    """
    #intersection point between the above line and a cercle of center T and radii 2r
    new_intercept=intercept+slope*center[0]-center[1]
    _A=1+slope**2
    _B=2*slope*new_intercept
    _C=new_intercept**2-(2*r)**2

    x1=(-_B+np.sqrt(_B**2-4*_A*_C))/(2*_A) + center[0]
    x2=(-_B-np.sqrt(_B**2-4*_A*_C))/(2*_A) + center[0]
    y1=slope*x1+intercept
    y2=slope*x2+intercept

    #we need to choose which one is the correct intersection point
    X_calculated1=(x1,y1)
    X_calculated2=(x2,y2)

    return X_calculated1,X_calculated2

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
    P1P0_arr=P1-P0
    auxiliar_point=(b/d)*P0_arr+(a/d)*P1_arr

    intersec1_x=auxiliar_point[0]+(h/d)*P1P0_arr[1]
    intersec2_x=auxiliar_point[0]-(h/d)*P1P0_arr[1]

    intersec1_y=auxiliar_point[1]-(h/d)*P1P0_arr[0]
    intersec2_y=auxiliar_point[1]+(h/d)*P1P0_arr[0]

    X1=(intersec1_x,intersec1_y)
    X2=(intersec2_x,intersec2_y)

    return X1, X2

def generate_random_numbers_inside_circle(center,R,num_points):
    """
    generates random numbers inside a circle of radii=R and center=center

    Parameters
    ----------    
        center: tuple, list of len 2
            x, y coordinates of the center
        R: float
            radii of circumference
        num_points: int
            number of random point generated
    Returns
    -------
        numpy array of shape (num_points, 2)

    """
    theta = np.random.uniform(0,2*np.pi, num_points)
    radius = np.random.uniform(0,R, num_points) ** 0.5
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    x=center[0]+x
    y=center[1]+y

    return np.concatenate((x.reshape(-1,1),y.reshape(-1,1)), axis=1)

def generate_random_numbers_inside_rectangle(W,H,num_points,safety_distance):
    """
    generates random numbers inside a rectangle of left bottom vertex (0,0)
    and right top vertex (W,H). We also consider a safety distance from 
    the sides of the rectangle so there are less collisions.

    Parameters
    ----------    
        W,H: float
            width and height of the rectangle
        num_points: 
            number of random point generated
        safety_distance: float
            We shrink the rectangle a safety distance 
    Returns
    -------
        numpy array of shape (num_points, 2)

    """
    xlist = np.random.uniform(0+safety_distance, W-safety_distance, num_points)
    ylist = np.random.uniform(0+safety_distance, H-safety_distance, num_points)
    real_points=np.concatenate((xlist.reshape(-1,1),ylist.reshape(-1,1)), axis=1)
    return real_points


def geometric_parameters(r,C,T,P):
    """
    Using r, C,T,P we compute the:
        d,b,a distances
        beta, alpha angles
        X point

    Parameters
    ----------    
        r: float
            radii of pool balls
        C,T,P: tuples
            x, y coordinates of rellevant points
    Returns
    -------
        d,b,a,alpha,beta, X: miscellanous
    """

    #virtual point X (see fig 4.1 adelaide university thesis)
    # we parametrize the line PT equation and compute the point 
    # that is 2*r distance from T
    t=1+2*r*(1/np.sqrt((T[0]-P[0])**2+(T[1]-P[1])**2))
    x_x=P[0]+(T[0]-P[0])*t
    y_x=P[1]+(T[1]-P[1])*t
    X=(x_x,y_x)

    # we calculate d and b using T and C points
    d=np.sqrt((T[0]-C[0])**2+(T[1]-C[1])**2)
    b=np.sqrt((T[0]-P[0])**2+(T[1]-P[1])**2)

    #we convert all rellevant points to numpy array:
    C_arr = np.array(C)
    T_arr = np.array(T)
    P_arr = np.array(P)
    X_arr = np.array(X)

    #To compute a and alpha we need to use cos and sin rules
    beta=np.pi-angle_abc(C_arr,T_arr,P_arr)
    #beta=angle_abc(C_arr,T_arr,X_arr)
    #phi=angle_abc(T_arr,C_arr,np.array([1,0]))
    a=np.sqrt(d**2+(2*r)**2-2*d*(2*r)*np.cos(beta))
    alpha=np.arcsin(2*r*np.sin(beta)/a)
    #alpha=angle_abc(X_arr,C_arr,T_arr)

    return d,b,a,alpha, beta, X

#incr_theta=incr_alpha is a random value
#we need to define it in order to analyze the error
def incr_beta(r,d,alpha,beta,incr_alpha):
    """
    Computes the variation of beta for a given variation of alpha
    (it can be vectorized).

    Parameters
    ----------    
        r: float
            radii of pool balls
        d: float
            CT distance
        alpha: float
            angle between XCT
        beta: float
            aangle between XTC or 180 - angle between CTP
        incr_alpha: float or array

    Returns
    -------
        incr_beta: float or array

    """
    A=np.sin(alpha+incr_alpha)/(2*r)#f(incr_alpha)
    B=d**2+(2*r)**2
    C=-2*d*(2*r)
    #cos_x=cos(beta+incr_beta)
    cos_x=(-C*A**2+np.sqrt(C**2*A**4-4*(-1+B*A**2)))/2 
    incr_beta=np.arccos(cos_x)-beta
    return incr_beta

def cue_ball_trajectory(r,C,C_estimated,X_estimated, end_effector='piston'):
    """
    
    """

    if end_effector=='piston':
        #line parallel to C'X' that goes trough C
        slope=(X_estimated[1]-C_estimated[1])/(X_estimated[0]-C_estimated[0])
        intercept=C[1]-slope*C[0]

    elif end_effector=='cue':
        #line C'X'
        slope_estimated=(X_estimated[1]-C_estimated[1])/(X_estimated[0]-C_estimated[0])
        intercept_estimated=C_estimated[1]-slope_estimated*C_estimated[0]
        contact_point1,contact_point2=intersection_circle_line(slope_estimated, intercept_estimated,r,C)

        dist1=np.sqrt((contact_point1[0]-X_estimated[0])**2+(contact_point1[1]-X_estimated[1])**2)
        dist2=np.sqrt((contact_point2[0]-X_estimated[0])**2+(contact_point2[1]-X_estimated[1])**2)

        if dist1<dist2:
            contact_point=contact_point1
        else:
            contact_point=contact_point2

        slope=(C[1]-contact_point[1])/(C[0]-contact_point[0])
        intercept=C[1]-slope*C[0]

    return slope,intercept


def compute_Q(r,C,T,P,C_estimated,X_estimated):


    slope, intercept = cue_ball_trajectory(r,C,C_estimated,X_estimated, end_effector='piston')

    X_calculated1,X_calculated2=intersection_circle_line(slope, intercept,r,T)

    distCX_calculated1=np.sqrt((C[0]-X_calculated1[0])**2+(C[1]-X_calculated1[1])**2)
    distCX_calculated2=np.sqrt((C[0]-X_calculated2[0])**2+(C[1]-X_calculated2[1])**2)

    # we choose the point of intersection that is closest to C
    if distCX_calculated1<distCX_calculated2:
        X_calculated=X_calculated1
    else:
        X_calculated=X_calculated2

    #line X''T (1)
    slope1=(X_calculated[1]-T[1])/(X_calculated[0]-T[0])
    intercept1=X_calculated[1]-slope1*X_calculated[0]

    #line TP (=line XT) (2)
    slope2=(P[1]-T[1])/(P[0]-T[0])
    intercept2=T[1]-slope2*T[0]

    #line perpendicular to TP (3)
    slope3=-1/slope2
    intercept3=P[1]-slope3*P[0]

    #intersection between line X''T and line perpendicular to TP
    Qx=(intercept1-intercept3)/(slope3-slope1)
    Qy=slope1*Qx+intercept1
    Q=(Qx,Qy)
    delta=np.sqrt((Qy-P[1])**2+(Qx-P[0])**2)

    return X_calculated,Q, delta

    
def pockets_inside_region_of_interest(pockets,C,T,d,two_times_r):

    # two_times_r=2*r
    X1,X2=intersection_two_circles(C,T,d,two_times_r)
    M=(np.array(X1)+np.array(X2))/2    

    #now we need to compute the lines X1T and X2T
    #line X1T (1)
    slope1=(X1[1]-T[1])/(X1[0]-T[0])
    intercept1=X1[1]-slope1*X1[0]

    #line X2T (2)
    slope2=(X2[1]-T[1])/(X2[0]-T[0])
    intercept2=X2[1]-slope2*X2[0]

    # M falls in the opposite region of the region of interest
    # 
    if M[1]>slope1*M[0]+intercept1:
        if M[1]>slope2*M[0]+intercept2:
            valid_pockets=(pockets[:,1]<slope1*pockets[:,0]+intercept1) & (pockets[:,1]<slope2*pockets[:,0]+intercept2)
            #(1):<, (2):<
        else:
            valid_pockets=(pockets[:,1]<slope1*pockets[:,0]+intercept1) & (pockets[:,1]>slope2*pockets[:,0]+intercept2)
            #(1):<, (2):>
    else:
        if M[1]>slope2*M[0]+intercept2:
            valid_pockets=(pockets[:,1]>slope1*pockets[:,0]+intercept1) & (pockets[:,1]<slope2*pockets[:,0]+intercept2)
            #(1):>, (2):<
        else: 
            valid_pockets=(pockets[:,1]>slope1*pockets[:,0]+intercept1) & (pockets[:,1]>slope2*pockets[:,0]+intercept2)
            #(1):>, (2):>
    return slope1,slope2,intercept1,intercept2,pockets[valid_pockets]


def draw_ideal_configuration(ax,r,C,T,P,X):
    #plot geometric situation
    ax.add_patch(plt.Circle(C, r, color='b',fill=False, clip_on=False,linewidth=0.5))
    ax.add_patch(plt.Circle(T, r, color='r',fill=False, clip_on=False,linewidth=0.5))
    ax.add_patch(plt.Circle(X, r, color='b',fill=False, clip_on=False,linewidth=0.5))
    ax.add_artist(plt.Circle(P, 0.5, color='b',fill=True, clip_on=False,linewidth=0.5))
    ax.add_artist(plt.Circle(C, 0.1, color='b',fill=True, clip_on=False,linewidth=0.5))
    ax.add_artist(plt.Circle(T, 0.1, color='r',fill=True, clip_on=False,linewidth=0.5))
    ax.add_patch(plt.Circle(T, 2*r, color='r',fill=False, clip_on=False,linewidth=0.5,linestyle='--'))
    ax.add_artist(plt.Circle(X, 0.1, color='b',fill=True, clip_on=False,linewidth=0.5))

    lines = [[C, X], [X, T], [T, P]]
    lc = mc.LineCollection(lines, colors='b', linewidths=0.5)
    ax.add_collection(lc)

    #Use adjustable='box-forced' to make the plot area square-shaped as well.
    ax.set_aspect('equal', adjustable='box')
    
    return ax

def draw_real_configuration(ax,r,C,X_calculated,Q):
    #plot geometric situation
    ax.add_patch(plt.Circle(X_calculated, r, color='orange',fill=False, clip_on=False,linewidth=0.5,linestyle='--'))
    ax.add_artist(plt.Circle(X_calculated, 0.1, color='orange',fill=True, clip_on=False,linewidth=0.5,linestyle='--'))
    ax.add_artist(plt.Circle(Q, 0.5, color='orange',fill=True, clip_on=False,linewidth=0.5))

    lines = [[C, X_calculated], [X_calculated, Q]]
    lc = mc.LineCollection(lines, colors='orange', linewidths=0.5)
    ax.add_collection(lc)

    #Use adjustable='box-forced' to make the plot area square-shaped as well.
    ax.set_aspect('equal', adjustable='box')
    
    return ax

def draw_pool_table_with_pockets(ax,W,H,img,pockets):
    #ax.add_patch(plt.Rectangle((0, 0), W, H,color='k',fill=False, clip_on=False,linewidth=0.5))

    for P in pockets:
        ax.add_artist(plt.Circle(P, 6, color='k',fill=True, clip_on=False,linewidth=0.5,alpha=0.5))

    ax.imshow(img, extent=[0, W, 0, H], cmap='gray')

    #Use adjustable='box-forced' to make the plot area square-shaped as well.
    ax.set_aspect('equal', adjustable='box')

    return ax

def draw_region_of_interest(ax,W,C,T,slope1,slope2,intercept1,intercept2):
    #x = np.arange(0,W,0.5)
    #y1 = slope1*x+intercept1
    #y2 = slope2*x+intercept2

    #ax.plot(x, y1, color='r', linewidth=0.5,alpha=0.25)
    #ax.plot(x, y2,color='r', linewidth=0.5,alpha=0.25)

    # we parametrize the line CT equation and compute the point 
    # that is 1000 distance from T
    t=1000
    far_away_point_x=C[0]+(T[0]-C[0])*t
    far_away_point_y=C[1]+(T[1]-C[1])*t
    far_away_point=(far_away_point_x,far_away_point_y)

    #line CT 
    slope=(C[1]-T[1])/(C[0]-T[0])
    intercept=T[1]-slope*T[0]

    #line perpendicular to CT that contains far_away_point
    slope_perpendicular=-1/slope
    intercept_perpendicular=far_away_point[1]-slope_perpendicular*far_away_point[0]

    #intersection between two lines
    x1=(-intercept1+intercept_perpendicular)/(-slope_perpendicular+slope1)
    x2=(-intercept2+intercept_perpendicular)/(-slope_perpendicular+slope2)
    y1=(slope1*intercept_perpendicular-slope_perpendicular*intercept1)/(-slope_perpendicular+slope1)
    y2=(slope2*intercept_perpendicular-slope_perpendicular*intercept2)/(-slope_perpendicular+slope2)
    
    points = np.array([[T[0],T[1]], [x1,y1], [x2,y2]])

    ax.add_patch(plt.Polygon(points, color='g',fill=True, clip_on=True,alpha=0.2))

    return ax

def draw_point(ax,point):
    ax.add_artist(plt.Circle(point, 0.1, color='orange',fill=True, clip_on=False,linewidth=0.5))

    #Use adjustable='box-forced' to make the plot area square-shaped as well.
    ax.set_aspect('equal', adjustable='box')

    return ax

def draw_specific_configuration(ax,W,H,img,pockets,r,C,T,P,X,C_estimated,T_estimated,X_calculated,Q):
    ax=draw_pool_table_with_pockets(ax,W,H,img,pockets)
    ax=draw_ideal_configuration(ax,r,C,T,P,X)
    ax=draw_real_configuration(ax,r,C,X_calculated,Q)
    ax=draw_point(ax,C_estimated)
    ax=draw_point(ax,T_estimated)
    return ax