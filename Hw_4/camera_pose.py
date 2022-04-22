import numpy as np

from feature import EstimateE_RANSAC

import cv2

def GetCameraPoseFromE(E):
    """
    Find four conﬁgurations of rotation and camera center from E

    Parameters
    ----------
    E : ndarray of shape (3, 3)
        Essential matrix

    Returns
    -------
    R_set : ndarray of shape (4, 3, 3)
        The set of four rotation matrices
    C_set : ndarray of shape (4, 3)
        The set of four camera centers
    """
    
    u,s,v_t=np.linalg.svd(E)
    w_1=np.array([[0,-1,0], [1,0,0], [0,0,1]])
    w_2=np.array([[0,1,0], [-1,0,0], [0,0,1]])

    t_1=u[:,2]
    t_2=-t_1

    R_1=u@w_1@v_t
    #Ensure determinant of 1
    if(np.linalg.det(R_1)<0):
        R_1=-R_1

    R_2=u@w_2@v_t
    #Ensure determinant of 1
    if(np.linalg.det(R_2)<0):
        R_2=-R_2

    R_set=np.array([R_1, R_2, R_1, R_2])
    C_set=np.array([-R_1.T@t_1, -R_2.T@t_1, -R_1.T@t_2, -R_2.T@t_2])

    return R_set, C_set



def Triangulation(P1, P2, track1, track2):
    """
    Use the linear triangulation method to triangulation the point

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X : ndarray of shape (n, 3)
        The set of 3D points
    """
    X=np.zeros((len(track1),3))
    for i in range(len(X)):
        if(np.all(track1[i]==[-1]) or np.all(track2[i]==[-1])):
            X[i]=[-1,-1,-1]
        else:
            A=np.vstack((([[0, -1, track1[i,1]], [1, 0, -track1[i,0]], [-track1[i,1], track1[i,0], 0]]@P1),
                ([[0, -1, track2[i,1]], [1, 0, -track2[i,0]], [-track2[i,1], track2[i,0], 0]]@P2)))
            #Get null space
            u,s,v_t=np.linalg.svd(A)
            #Normalize by last entry to make it [X 1]
            if(abs(v_t[-1,3])>0):
                X[i]=v_t[-1,:3]/v_t[-1,3]
            #Prevent divide by zero
            else:
                X[i]=v_t[-1,:3]/(v_t[-1,3]+1e-8)

    return X



def EvaluateCheirality(P1, P2, X):
    """
    Evaluate the cheirality condition for the 3D points

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    X : ndarray of shape (n, 3)
        Set of 3D points

    Returns
    -------
    valid_index : ndarray of shape (n,)
        The binary vector indicating the cheirality condition, i.e., the entry 
        is 1 if the point is in front of both cameras, and 0 otherwise
    """

    valid_index=np.zeros(len(X))
    R1=P1[:,:3]
    C1=-R1.T@P1[:,3]
    R2=P2[:,:3]
    C2=-R2.T@P2[:,3]

    for i in range(len(X)):
        if(np.all(X[i]==[-1])):
            continue
        if((np.dot(R1[2],(X[i]-C1)))>0 and (np.dot(R2[2],(X[i]-C2)))>0):
            valid_index[i]=1

    #Conver to a boolean matrix
    valid_index=np.array(valid_index, dtype=bool)

    return valid_index



def EstimateCameraPose(track1, track2):
    """
    Return the best pose conﬁguration

    Parameters
    ----------
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    X : ndarray of shape (F, 3)
        The set of reconstructed 3D points
    """
    #Remove the -1s from the tracks before ransac to avoid issues
    keep=[]
    for k in range(track1.shape[0]):
        if((np.any(track1[k]!=-1) and np.any(track2[k]!=-1))):
            keep.append(k)

    clean_t1=track1[keep,:]
    clean_t2=track2[keep,:]
    
    E, inliers=EstimateE_RANSAC(clean_t1, clean_t2, 1000, 1e-3)
    

    R_set, C_set=GetCameraPoseFromE(E)
    R=None
    C=None
    X=None
    best=0
    for i in range(len(R_set)):
        #Build P1
        P1=np.hstack((np.eye(3),np.zeros((3,1))))

        #Build P2
        P2=R_set[i]@np.hstack((np.eye(3), -np.reshape(C_set[i], (3,1))))
        
        #Triangulate points using each configuration
        X_est=Triangulation(P1, P2, track1, track2)

        #Evaluate cheirality
        valid_index=EvaluateCheirality(P1, P2, X_est)

        #Update X_est
        X_est[valid_index==False]=[-1,-1,-1]

        #Update with best orientation
        if(np.count_nonzero(valid_index)>best):
            R=R_set[i]
            C=C_set[i]
            X=X_est

    return R, C, X