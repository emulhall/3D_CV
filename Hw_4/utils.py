import numpy as np



def Rotation2Quaternion(R):
    """
    Convert a rotation matrix to quaternion
    
    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix

    Returns
    -------
    q : ndarray of shape (4,)
        The unit quaternion (w, x, y, z)
    """

    tr=np.trace(R)
    
    q=np.zeros(4)
    if(tr>0):
        q[0]=np.math.sqrt((float(1)+R[0][0]+R[1][1]+R[2][2]))/2
        q[1]=(R[2][1]-R[1][2])/(4*q[0])
        q[2]=(R[0][2]-R[2][0])/(4*q[0])
        q[3]=(R[1][0]-R[0][1])/(4*q[0])
    elif((R[0][0]>R[1][1]) and (R[0][0]>R[2][2])):
        q[1]=np.math.sqrt((float(1)+R[0][0]-R[1][1]-R[2][2]))/2
        q[0]=(R[2][1]-R[1][2])/(4*q[1])
        q[3]=(R[0][2]+R[2][0])/(4*q[1])
        q[2]=(R[1][0]+R[0][1])/(4*q[1])
    elif((R[1][1]>R[2][2])):
        q[2]=np.math.sqrt((float(1)-R[0][0]+R[1][1]-R[2][2]))/2
        q[3]=(R[2][1]+R[1][2])/(4*q[2])
        q[0]=(R[0][2]-R[2][0])/(4*q[2])
        q[1]=(R[1][0]+R[0][1])/(4*q[2])
    else:
        q[3]=np.math.sqrt((float(1)-R[0][0]-R[1][1]+R[2][2]))/2
        q[2]=(R[2][1]+R[1][2])/(4*q[3])
        q[1]=(R[0][2]+R[2][0])/(4*q[3])
        q[0]=(R[1][0]-R[0][1])/(4*q[3])



    return q



def Quaternion2Rotation(q):
    """
    Convert a quaternion to rotation matrix
    
    Parameters
    ----------
    q : ndarray of shape (4,)
        Unit quaternion (w, x, y, z)

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    """
    
    #We're going to build it row by row to keep things a bit clearer
    R=[]
    R.append([((q[0]**2)+(q[1]**2)-0.5), (q[1]*q[2]-q[0]*q[3]), (q[0]*q[2]+q[1]*q[3])])
    R.append([(q[0]*q[3]+q[1]*q[2]), (q[0]**2+q[2]**2-0.5), (q[2]*q[3]-q[0]*q[1])])
    R.append([(q[1]*q[3]-q[0]*q[2]), (q[0]*q[1]+q[2]*q[3]), (q[0]**2+q[3]**2-0.5)])

    R=np.reshape(np.array(R), (3,3))

    return 2*R