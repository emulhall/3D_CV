import numpy as np
from scipy.optimize import least_squares

from utils import Rotation2Quaternion
from utils import Quaternion2Rotation


def FindMissingReconstruction(X, track_i):
    """
    Find the points that will be newly added

    Parameters
    ----------
    X : ndarray of shape (F, 3)
        3D points
    track_i : ndarray of shape (F, 2)
        2D points of the newly registered image

    Returns
    -------
    new_point : ndarray of shape (F,)
        The indicator of new points that are valid for the new image and are 
        not reconstructed yet
    """
    
    new_point=np.zeros(len(X))
    for f in range(len(new_point)):
        if((np.all(X[f,:]==-1) and np.any(track_i[f,:]!=-1))):
            new_point[f]=1

    return new_point



def Triangulation_nl(X, P1, P2, x1, x2):
    """
    Refine the triangulated points

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        3D points
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    x1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    x2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X_new : ndarray of shape (n, 3)
        The set of refined 3D points
    """
    R1=P1[:,:3]
    R2=P2[:,:3]
    q1=Rotation2Quaternion(R1)
    q2=Rotation2Quaternion(R2)
    C1=-R1.T@P1[:,3]
    C2=-R2.T@P2[:,3]
    p1=np.vstack((np.reshape(C1,(-1,1)),np.reshape(q1, (-1,1))))
    p2=np.vstack((np.reshape(C2,(-1,1)),np.reshape(q2, (-1,1))))
    p1=np.reshape(p1, (7))
    p2=np.reshape(p2, (7))

    nIters=100
    for n in range(nIters):
        #Calculate P1's contribution to delta_X
        delta_X=np.zeros(X.shape)
        for i in range(len(X)):
            #Invalid points should not contribute to delta_X
            if(np.all(X[i]==-1)):
                continue
            else:
                #Compute the point jacobian
                dfdX_1=ComputePointJacobian(X[i], p1)
                dfdX_2=ComputePointJacobian(X[i], p2)

                #compute f(X)
                f_p_i_1=R1@np.hstack((np.eye(3), np.reshape(-C1, (3,1))))@np.vstack((np.reshape(X[i], (3,1)), [1]))
                f_p_i_1=f_p_i_1/f_p_i_1[-1]
                f_p_i_1=f_p_i_1[:2].flatten()

                f_p_i_2=R2@np.hstack((np.eye(3), np.reshape(-C2, (3,1))))@np.vstack((np.reshape(X[i], (3,1)), [1]))
                f_p_i_2=f_p_i_2/f_p_i_2[-1]
                f_p_i_2=f_p_i_2[:2].flatten()

                #Compute delta_x
                delta_X[i]=delta_X[i]+np.reshape((np.linalg.inv(dfdX_1.T@dfdX_1 + 2*np.eye(3))@dfdX_1.T@(np.reshape(x1[i], (-1,1))-np.reshape(f_p_i_1, (-1,1)))), (3))
                delta_X[i]=delta_X[i]+np.reshape((np.linalg.inv(dfdX_2.T@dfdX_2 + 2*np.eye(3))@dfdX_2.T@(np.reshape(x2[i], (-1,1))-np.reshape(f_p_i_2, (-1,1)))), (3))
        

        X=X+delta_X

    X_new=X
    return X_new



def ComputePointJacobian(X, p):
    """
    Compute the point Jacobian

    Parameters
    ----------
    X : ndarray of shape (3,)
        3D point
    p : ndarray of shape (7,)
        Camera pose made of camera center and quaternion

    Returns
    -------
    dfdX : ndarray of shape (2, 3)
        The point Jacobian
    """
    
    ddx=Quaternion2Rotation(p[3:])
    R=Quaternion2Rotation(p[3:])
    C=p[:3]
    u,v,w=R@(X-C)

    r_1=(w*ddx[0]-u*ddx[2])/(w**2)
    r_2=(w*ddx[1]-v*ddx[2])/(w**2)

    dfdX=np.vstack((r_1,r_2))

    return dfdX



def SetupBundleAdjustment(P, X, track):
    """
    Setup bundle adjustment

    Parameters
    ----------
    P : ndarray of shape (K, 3, 4)
        Set of reconstructed camera poses
    X : ndarray of shape (J, 3)
        Set of reconstructed 3D points
    track : ndarray of shape (K, J, 2)
        Tracks for the reconstructed cameras

    Returns
    -------
    z : ndarray of shape (7K+3J,)
        The optimization variable that is made of all camera poses and 3D points
    b : ndarray of shape (2M,)
        The 2D points in track, where M is the number of 2D visible points
    S : ndarray of shape (2M, 7K+3J)
        The sparse indicator matrix that indicates the locations of Jacobian computation
    camera_index : ndarray of shape (M,)
        The index of camera for each measurement
    point_index : ndarray of shape (M,)
        The index of 3D point for each measurement
    """
    
    #Get R and t from P
    R=P[:,:,:3]
    t=P[:,:,3]

    #Build z
    z=np.zeros((7*P.shape[0]+3*X.shape[0]))
    for i in range(P.shape[0]):
        C=-R[i].T@t[i]
        q=Rotation2Quaternion(R[i])
        z[i*7:(i+1)*7]=(np.vstack((np.reshape(C, (-1,1)), np.reshape(q, (-1,1))))).flatten()

    #Append X to the end of z
    z[7*P.shape[0]:]=X.flatten()

    #Get valid 2D points
    valid=np.logical_and(track[:,:,0]!=-1, track[:,:,1]!=-1)
    indices=np.argwhere(valid)
    #Build valid camera_index
    camera_index=indices[:,0]
    K=len(np.unique(camera_index))
    #Build valid point_index
    point_index=indices[:,1]

    #Build b
    b=track[valid].flatten()

    #Build S
    S=np.zeros((b.shape[0], z.shape[0]))
    for m in range(len(indices)):
        k=camera_index[m]
        j=point_index[m]

        #Update valid poses - do not want to update first two poses
        if(k>1):
            S[2*m:2*(m+1), 7*k:7*(k+1)]=1

        #Update valid points
        S[2*m:2*(m+1), 7*K+3*j:7*K+3*(j+1)]=1


    return z, b, S, camera_index, point_index
    


def MeasureReprojection(z, b, n_cameras, n_points, camera_index, point_index):
    """
    Evaluate the reprojection error

    Parameters
    ----------
    z : ndarray of shape (7K+3J,)
        Optimization variable
    b : ndarray of shape (2M,)
        2D measured points
    n_cameras : int
        Number of cameras
    n_points : int
        Number of 3D points
    camera_index : ndarray of shape (M,)
        Index of camera for each measurement
    point_index : ndarray of shape (M,)
        Index of 3D point for each measurement

    Returns
    -------
    err : ndarray of shape (2M,)
        The reprojection error
    """
    err=np.zeros(len(b))
    for m in range(len(camera_index)):
        k=camera_index[m]
        j=point_index[m]

        p=z[7*k:7*(k+1)]
        C=p[:3]
        q=p[3:]
        q=q/np.linalg.norm(q)
        R=Quaternion2Rotation(q)

        X=z[7*n_cameras+3*j:7*n_cameras+3*(j+1)]

        f_p=R@np.hstack((np.eye(3), np.reshape(-C, (3,1))))@np.vstack((np.reshape(X, (3,1)), [1]))
        f_p=f_p/f_p[-1]
        f_p=f_p[:2].flatten()

        err[2*m:2*(m+1)]=f_p-b[2*m:2*(m+1)]

    return err



def UpdatePosePoint(z, n_cameras, n_points):
    """
    Update the poses and 3D points

    Parameters
    ----------
    z : ndarray of shape (7K+3J,)
        Optimization variable
    n_cameras : int
        Number of cameras
    n_points : int
        Number of 3D points

    Returns
    -------
    P_new : ndarray of shape (K, 3, 4)
        The set of refined camera poses
    X_new : ndarray of shape (J, 3)
        The set of refined 3D points
    """
    
    P=np.reshape(z[0:7*n_cameras], (n_cameras,7))
    X_new=np.reshape(z[7*n_cameras:], (n_points,3))

    P_new=np.zeros((n_cameras, 3, 4))

    for k in range(n_cameras):
        C=P[k,:3]
        q=P[k,3:]
        q=q/np.linalg.norm(q)
        R=Quaternion2Rotation(q)
        P_new[k]=R@np.hstack((np.eye(3), -np.reshape(C, (3,1))))

    return P_new, X_new



def RunBundleAdjustment(P, X, track):
    """
    Run bundle adjustment

    Parameters
    ----------
    P : ndarray of shape (K, 3, 4)
        Set of reconstructed camera poses
    X : ndarray of shape (J, 3)
        Set of reconstructed 3D points
    track : ndarray of shape (K, J, 2)
        Tracks for the reconstructed cameras

    Returns
    -------
    P_new : ndarray of shape (K, 3, 4)
        The set of refined camera poses
    X_new : ndarray of shape (J, 3)
        The set of refined 3D points
    """
    
    z, b, S, camera_index, point_index = SetupBundleAdjustment(P, X, track)

    #Run sparse bundle adjustment
    n_cameras=len(np.unique(camera_index))
    n_points=len(np.unique(point_index))

    res=least_squares(MeasureReprojection, z, jac_sparsity=S, verbose=2,x_scale='jac',
        ftol=1e-4, method='trf', args=(b, n_cameras, n_points, camera_index,point_index))

    #Update poses and points
    P_new, X_new = UpdatePosePoint(res.x, n_cameras, n_points)

    return P_new, X_new