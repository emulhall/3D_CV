import numpy as np
import random

from utils import Rotation2Quaternion
from utils import Quaternion2Rotation


def PnP(X, x):
    """
    Implement the linear perspective-n-point algorithm

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    """
    
    #Sample 6 random points
    indices=random.sample(range(0,len(x)), 6)
    sample_X=X[indices,:]
    sample_x=x[indices,:]

    #Build A
    A=[]
    for i in range(len(sample_X)):
        A.append([sample_X[i,0], sample_X[i,1], sample_X[i,2], 1, 0, 0, 0, 0, -sample_x[i,0]*sample_X[i,0], -sample_x[i,0]*sample_X[i,1],-sample_x[i,0]*sample_X[i,2],-sample_x[i,0]])
        A.append([0, 0, 0, 0, sample_X[i,0], sample_X[i,1], sample_X[i,2], 1, -sample_x[i,1]*sample_X[i,0], -sample_x[i,1]*sample_X[i,1],-sample_x[i,1]*sample_X[i,2],-sample_x[i,1]])

    A=np.array(A)

    #Get the null space of A using svd
    u, s, v_t=np.linalg.svd(A)
    p=v_t[-1]
    p=np.reshape(p, (3,4))

    R=p[:,:3]

    #Ensure orthogonality of R
    u_r, s_r, v_t_r = np.linalg.svd(R)
    R=u_r@v_t_r

    t=p[:,3]/s_r[0]

    #Ensure determinant of 1
    if(np.linalg.det(R)<0):
        R=-R
        t=-t

    C=-R.T@t

    return R, C



def PnP_RANSAC(X, x, ransac_n_iter, ransac_thr):
    """
    Estimate pose using PnP with RANSAC

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    inlier : ndarray of shape (n,)
        The indicator of inliers, i.e., the entry is 1 if the point is a inlier,
        and 0 otherwise
    """
    R=None
    C=None
    inlier=np.zeros(len(X))
    for n in range(ransac_n_iter):
        #Get R and C estimate
        R_est, C_est = PnP(X,x)

        #Build the projection matrix
        p=R_est@np.hstack((np.eye(3),-np.reshape(C_est, (3,1))))

        #Evaluate each point to check for inliers
        inlier_est=np.zeros(len(X))
        for i in range(X.shape[0]):
            #Must satisfy Cheirality
            if(np.dot(R_est[2,:], (X[i]-C_est))>0):
                #Calculate projection
                X_4=np.vstack((np.reshape(X[i], (3,1)),[1]))
                u_est=p@X_4
                #Normalize
                u_est=u_est/u_est[-1]
                #Calculate error
                err=np.linalg.norm(u_est[:2].flatten()-x[i])
                #Must be below the threshold
                if(err<ransac_thr):
                    inlier_est[i]=1

        #Get best R and C
        if(np.count_nonzero(inlier_est)>np.count_nonzero(inlier)):
            R=R_est
            C=C_est
            inlier=inlier_est

    return R, C, inlier



def ComputePoseJacobian(p, X):
    """
    Compute the pose Jacobian

    Parameters
    ----------
    p : ndarray of shape (7,)
        Camera pose made of camera center and quaternion
    X : ndarray of shape (3,)
        3D point

    Returns
    -------
    dfdp : ndarray of shape (2, 7)
        The pose Jacobian
    """
    C=p[:3]
    q=p[3:]
    R=Quaternion2Rotation(q)

    u,v,w=R@(X-C)

    drdq=np.array([[0, 0, -4*q[2], -4*q[3]], 
    [-2*q[3], 2*q[2], 2*q[1], -2*q[0]],
    [2*q[2], 2*q[3], 2*q[0], 2*q[1]],
    [2*q[3], 2*q[2], 2*q[1], 2*q[0]],
    [0, -4*q[1], 0, -4*q[3]],
    [-2*q[1], -2*q[0], 2*q[3], 2*q[2]],
    [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
    [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
    [0, -4*q[1], -4*q[2], 0]])

    x_c=X-C

    ddr=np.array([[x_c[0], x_c[1], x_c[2], 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, x_c[0], x_c[1], x_c[2], 0, 0, 0], 
        [0,0,0,0,0,0,x_c[0], x_c[1], x_c[2]]])

    ddq=ddr@drdq

    ddc=-R
    

    ddp=np.hstack((ddc, ddq))
    
    r_1=(w*ddp[0]-u*ddp[2])/(w**2)
    r_2=(w*ddp[1]-v*ddp[2])/(w**2)

    dfdp=np.vstack((r_1, r_2))

    return dfdp



def PnP_nl(R, C, X, x):
    """
    Update the pose using the pose Jacobian

    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix refined by PnP
    c : ndarray of shape (3,)
        Camera center refined by PnP
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R_refined : ndarray of shape (3, 3)
        The rotation matrix refined by nonlinear optimization
    C_refined : ndarray of shape (3,)
        The camera center refined by nonlinear optimization
    """
    
    #Initialize the pose matrix
    q=Rotation2Quaternion(R)
    p=np.vstack((np.reshape(C,(-1,1)),np.reshape(q, (-1,1))))
    p=np.reshape(p, (7))
    nIters=50

    for j in range(nIters):
        C=p[:3]
        q=p[3:]
        R=Quaternion2Rotation(q)

        dfdpT_dfdp_sum=np.zeros((7,7))
        dfdpT_bi_fi_sum=np.zeros((7,1))
        for i in range(len(X)):
            #Invalid points shouldn't contribute to the error
            if(np.all(X[i]==-1) or np.all(x[i]==-1)):
                continue
            else:
                #Compute pose Jacobian
                dfdp_i=ComputePoseJacobian(p,X[i])

                #compute f(p)
                f_p_i=R@np.hstack((np.eye(3), np.reshape(-C, (3,1))))@np.vstack((np.reshape(X[i], (3,1)), [1]))
                f_p_i=f_p_i/f_p_i[-1]
                f_p_i=f_p_i[:2].flatten()
            
                dfdpT_dfdp_sum=dfdpT_dfdp_sum+(dfdp_i.T@dfdp_i)
                dfdpT_bi_fi_sum=dfdpT_bi_fi_sum+dfdp_i.T@(np.reshape(x[i], (-1,1))-np.reshape(f_p_i, (-1,1)))

        delta_p=np.linalg.inv(dfdpT_dfdp_sum+2*np.eye(7))@dfdpT_bi_fi_sum

        p=p+np.reshape(delta_p, (7))

        #Normalize by scale
        p[3:]=p[3:]/np.linalg.norm(p[3:])

    R_refined=Quaternion2Rotation(p[3:])
    C_refined=p[:3]

    return R_refined, C_refined