import os
import cv2
import numpy as np
from pylsd import lsd
import random
import math
from matplotlib import pyplot as plt



def FindVP(lines, K, ransac_thr, ransac_iter):
    """
    Find the vanishing point
    
    Parameters
    ----------
    lines : ndarray of shape (N_l, 4)
        Set of line segments where each row contains the coordinates of two 
        points (x1, y1, x2, y2)
    K : ndarray of shape (3, 3)
        Camera intrinsic parameters
    ransac_thr : float
        Error threshold for RANSAC
    ransac_iter : int
        Number of RANSAC iterations

    Returns
    -------
    vp : ndarray of shape (2,)
        The vanishing point
    inlier : ndarray of shape (N_i,)
        The index set of line segment inliers
    """

    vp=np.zeros((2))
    inlier=[]

    for n in range(ransac_iter):
        #Sample two random lines
        indices=random.sample(range(0, len(lines)), 2)
        l1=lines[indices[0]]
        l2=lines[indices[1]]

        #Pull out the four points that make up these lines and convert to the metric space
        u1=np.linalg.inv(K)@[l1[0], l1[1], 1]
        u2=np.linalg.inv(K)@[l1[2], l1[3], 1]
        u3=np.linalg.inv(K)@[l2[0], l2[1], 1]
        u4=np.linalg.inv(K)@[l2[2], l2[3], 1]

        #Use the points to calculate the plane normals
        line_1=np.cross(u1, u2)
        line_2=np.cross(u3, u4)

        #Take the cross product of these two lines to get the vanishing point estimate
        v=np.cross(line_1, line_2)

        #Check for paralell lines to avoid dividing by zero
        if(v[2]<1e-15):
            continue

        #Normalize
        v=v/v[2]

        #Move back to the pixel space by multiplying by K
        v=K@v

        #Move back to the 2D so that we return the right shape
        vp_est=np.array([v[0], v[1]])

        #Now that we have our estimated v, we need to test it for inliers
        inlier_est=[]
        for l in range(len(lines)):
            x1=lines[l][0]
            y1=lines[l][1]
            x2=lines[l][2]
            y2=lines[l][3]

            #Calculate distance
            num=abs(((x2-x1)*(y1-v[1]))-((x1-v[0])*(y2-y1)))
            den=math.sqrt((x2-x1)**2+(y2-y1)**2)
            d=num/den

            if(d<ransac_thr):
                inlier_est.append(l)

        if(len(inlier_est)>len(inlier)):
            inlier=inlier_est
            vp=vp_est


    return vp, np.array(inlier)


def ClusterLines(lines):
    """
    Cluster lines into two sets

    Parameters
    ----------
    lines : ndarray of shape (N_l - N_i, 4)
        Set of line segments excluding the inliers from the ﬁrst vanishing 
        point detection

    Returns
    -------
    lines_x : ndarray of shape (N_x, 4)
        The set of line segments for horizontal direction
    lines_y : ndarray of shape (N_y, 4)
        The set of line segments for vertical direction
    """

    lines_x=[]
    lines_y=[]

    for l in range(len(lines)):
        #We don't care about the direction of the slope, so take the absolute value 
        #Calculate the change in the x and y directions
        change_y=abs(lines[l][3]-lines[l][1])
        change_x=abs(lines[l][2]-lines[l][0])

        #We can't divide by 0, so check for undefined slopes which characterize vertical lines
        if(change_x<1e-10):
            lines_y.append(lines[l])
        else:
            #Calculate slope
            slope=change_y/change_x

            #Determine whether the line is horizontal or vertical based on a tested threshold
            if(slope<0.8):
                lines_x.append(lines[l])
            else:
                lines_y.append(lines[l])




    return np.array(lines_x), np.array(lines_y)


def CalibrateCamera(vp_x, vp_y, vp_z):
    """
    Calibrate intrinsic parameters

    Parameters
    ----------
    vp_x : ndarray of shape (2,)
        Vanishing point in x-direction
    vp_y : ndarray of shape (2,)
        Vanishing point in y-direction
    vp_z : ndarray of shape (2,)
        Vanishing point in z-direction

    Returns
    -------
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters
    """

    #Construct the A matrix
    u1=vp_x[0]
    u2=vp_y[0]
    u3=vp_z[0]

    v1=vp_x[1]
    v2=vp_y[1]
    v3=vp_z[1]
    A=np.array([[(u1*u2+v1*v2), u1+u2, v1+v2, 1], [(u3*u2+v3*v2), u3+u2, v3+v2, 1], [(u1*u3+v1*v3), u1+u3, v1+v3, 1]])

    #We solve for the null space of A to get our b matrix because Ab=0
    u, s, vh= np.linalg.svd(A)
    b=vh[-1]
    px=-b[1]/b[0]
    py=-b[2]/b[0]
    f=math.sqrt((b[3]/b[0])-(px**2+py**2))

    K=np.array([[f, 0, px], [0, f, py], [0, 0, 1]])

    return K


def GetRectificationH(K, vp_x, vp_y, vp_z):
    """
    Find a homography for rectification
    
    Parameters
    ----------
    K : ndarray of shape (3, 3)
        Camera intrinsic parameters
    vp_x : ndarray of shape (2,)
        Vanishing point in x-direction
    vp_y : ndarray of shape (2,)
        Vanishing point in y-direction
    vp_z : ndarray of shape (2,)
        Vanishing point in z-direction

    Returns
    -------
    H_rect : ndarray of shape (3, 3)
        The rectiﬁcation homography induced by pure rotation
    """
    #Move our vanishing points back to the metric space for calculations
    vp_x_3d=np.linalg.inv(K)@[vp_x[0], vp_x[1], 1]
    vp_y_3d=np.linalg.inv(K)@[vp_y[0], vp_y[1], 1]
    vp_z_3d=np.linalg.inv(K)@[vp_z[0], vp_z[1], 1]

    #We need the unit vectors for x and y 
    vp_x_3d=vp_x_3d/np.linalg.norm(vp_x_3d)
    vp_y_3d=vp_y_3d/np.linalg.norm(vp_y_3d)
    vp_z_3d=vp_z_3d/np.linalg.norm(vp_z_3d)

    #We stack these unit vectors to become our rotation matrix
    R=np.vstack((vp_x_3d, vp_y_3d))
    R=np.vstack((R, vp_z_3d))

    #Let's just do a double check on the determinant of our rotation matrix
    if(np.linalg.det(R)<0):
        R=-R

    H_rect=K@R@np.linalg.inv(K)

    return H_rect


def ImageWarping(im, H):
    """
    Warp image by the homography

    Parameters
    ----------
    im : ndarray of shape (h, w, 3)
        Input image
    H : ndarray of shape (3, 3)
        Homography

    Returns
    -------
    im_warped : ndarray of shape (h, w, 3)
        The warped image
    """

    im_warped=np.zeros(np.shape(im), dtype=np.uint8)
    for v in range(im_warped.shape[0]):
        for u in range(im_warped.shape[1]):
            #Calculate the forward mapping
            x2=[u, v, 1]
            x1=np.linalg.inv(H)@x2

            #normalize and take the floor
            x1=x1//x1[2]

            #Check if it is a valid point
            if(x1[0]<im.shape[1] and x1[0]>=0 and x1[1]<im.shape[0] and x1[1]>=0):
                im_warped[v][u]=im[math.floor(x1[1])][math.floor(x1[0])]
            else:
                im_warped[v][u]=[0,0,0]

    return im_warped


def ConstructBox(K, vp_x, vp_y, vp_z, W, a, d_near, d_far):
    """
    Construct a 3D box to approximate the scene geometry
    
    Parameters
    ----------
    K : ndarray of shape (3, 3)
        Camera intrinsic parameters
    vp_x : ndarray of shape (2,)
        Vanishing point in x-direction
    vp_y : ndarray of shape (2,)
        Vanishing point in y-direction
    vp_z : ndarray of shape (2,)
        Vanishing point in z-direction
    W : float
        Width of the box
    a : float
        Aspect ratio
    d_near : float
        Depth of the front plane
    d_far : float
        Depth of the back plane

    Returns
    -------
    U11, U12, U21, U22, V11, V12, V21, V22 : ndarray of shape (3,)
        The 8 corners of the box
    """

    #Move our vanishing points back to the metric space for calculations
    x_3d=np.linalg.inv(K)@[vp_x[0], vp_x[1], 1]
    y_3d=np.linalg.inv(K)@[vp_y[0], vp_y[1], 1]
    z=np.linalg.inv(K)@[vp_z[0], vp_z[1], 1]

    #Calculate the near and far Zs
    Z_rear = d_far*z
    Z_near = d_near*z

    #We need the unit vectors for x and y 
    #Because of error we use the Gram-Schmidt procedure to guarantee orthonormality
    z_norm=z/np.linalg.norm(z)
    x_norm=x_3d/np.linalg.norm(x_3d)
    y_norm=y_3d/np.linalg.norm(y_3d)
    x=x_norm-((np.dot(x_norm, z_norm)/np.dot(z_norm, z_norm))*z_norm)
    y=y_norm-((np.dot(y_norm, z_norm)/np.dot(z_norm, z_norm))*z_norm)-((np.dot(y_norm, x)/np.dot(x, x))*x)

    #Get the height from the width and aspect ratio
    H=W/a

    #Get the far coordinates
    U11=Z_rear+(W/2)*x+(H/2)*y
    U21=Z_rear+(W/2)*x-(H/2)*y
    U12=Z_rear-(W/2)*x+(H/2)*y
    U22=Z_rear-(W/2)*x-(H/2)*y


    #Get the near coordinates
    V11=Z_near+(W/2)*x+(H/2)*y
    V21=Z_near+(W/2)*x-(H/2)*y
    V12=Z_near-(W/2)*x+(H/2)*y
    V22=Z_near-(W/2)*x-(H/2)*y

    return U11, U12, U21, U22, V11, V12, V21, V22



def InterpolateCameraPose(R1, C1, R2, C2, w):
    """
    Interpolate the camera pose
    
    Parameters
    ----------
    R1 : ndarray of shape (3, 3)
        Camera rotation matrix of camera 1
    C1 : ndarray of shape (3,)
        Camera optical center of camera 1
    R2 : ndarray of shape (3, 3)
        Camera rotation matrix of camera 2
    C2 : ndarray of shape (3,)
        Camera optical center of camera 2
    w : float
        Weight between two poses

    Returns
    -------
    Ri : ndarray of shape (3, 3)
        The interpolated camera rotation matrix
    Ci : ndarray of shape (3,)
        The interpolated camera optical center
    """
    #Linear interpolation for the camera optical centers (C)
    Ci=(1-w)*C1+w*C2

    #Spherical linear interpolation for the camera rotation (R)
    #Convert to quaternion
    q1=Rotation2Quaternion(R1)
    q2=Rotation2Quaternion(R2)

    #Quaternion interpolation
    omega=np.arccos(np.dot(q1, q2))
    den=np.sin(omega)
    #Prevent divide by 0
    if(den<1e-20):
        den=1e-20
    p=(q1*np.sin((1-w)*omega)+q2*np.sin(w*omega))/den

    #Convert back to rotation matrix
    Ri=Quaternion2Rotation(p)

    return Ri, Ci


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

    q=np.zeros(4)
    #First we compute the trace
    tr=np.trace(R)

    #From the trace and diagonal of the matrix we can compute the quaternion values
    q[0]=math.sqrt((tr+1)/4)
    q[1]=math.sqrt((R[0][0]/2)+((1-tr)/4))
    q[2]=math.sqrt((R[1][1]/2)+((1-tr)/4))
    q[3]=math.sqrt((R[2][2]/2)+((1-tr)/4))


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

    return 2*np.array(R)


def GetPlaneHomography(p11, p12, p21, K, R, C, vx, vy):
    """
    Interpolate the camera pose
    
    Parameters
    ----------
    p11 : ndarray of shape (3,)
        Top-left corner
    p12 : ndarray of shape (3,)
        Top-right corner
    p21 : ndarray of shape (3,)
        Bottom-left corner
    K : ndarray of shape (3, 3)
        Camera intrinsic parameters
    R : ndarray of shape (3, 3)
        Camera rotation matrix
    C : ndarray of shape (3,)
        Camera optical center
    vx : ndarray of shape (h, w)
        All x coordinates in the image
    vy : ndarray of shape (h, w)
        All y coordinates in the image

    Returns
    -------
    H : ndarray of shape (3, 3)
        The homography that maps the rectiﬁed image to the canvas
    visibility_mask : ndarray of shape (h, w)
        The binary mask indicating membership to the plane constructed by p11, 
        p12, and p21
    """
    visibility_mask=np.ones(np.shape(vx))

    c=p11
    B1=(p12-c)/np.linalg.norm((p12-c))
    B2=(p21-c)/np.linalg.norm((p21-c))


    #Stack B1, B2 and c to form a matrix
    temp1 = np.column_stack((B1, B2))
    temp1 = np.column_stack((temp1, c))


    #Calculate first homography
    H_hat = K@temp1

    #Compute and stack 
    temp2 = np.column_stack((Ri@B1, Ri@B2))
    temp2 = np.column_stack((temp2, Ri@c-Ri@Ci))

    #Calculate the second homography
    H_tilda = K@temp2

    #Calculate our composite H
    H=H_tilda@np.linalg.inv(H_hat)

    #Calculate our max mu values
    mu_1_max = np.linalg.norm((c-p12))
    mu_2_max = np.linalg.norm((c-p21))

    for v in range(vx.shape[0]):
        for u in range(vx.shape[1]):
            mu=np.linalg.inv(H_tilda)@[u,v,1]

            if(mu[2]<0):
                visibility_mask[v][u]=0

            #Prevent divide by 0 and normalize
            if(mu[2]<1e-20):
                mu=mu/1e-20
            else:
                mu=mu/mu[2]

            if((mu[0]>=mu_1_max) or (mu[1]>=mu_2_max) or (mu[0]<=0) or (mu[1]<=0)):
                visibility_mask[v][u]=0



    return H, visibility_mask



if __name__ == '__main__':

    # Load the input image and detect the line segments
    im = cv2.imread('airport.jpg')
    im_h = im.shape[0]
    im_w = im.shape[1]
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lines = lsd(im_gray)

	# Approximate K
    f = 300
    K_apprx = np.asarray([
        [f, 0, im_w/2],
        [0, f, im_h/2],
        [0, 0, 1]
    ])

	#####################################################################
    # Compute the major z-directional vanishing point and its line segments using approximate K
    vp_z, inliers_z=FindVP(lines, K_apprx, 10, 1000)

	#####################################################################
    # Cluster the rest of line segments into two major directions and compute the x- and y-directional vanishing points using approximate K
    outliers=[]
    for i in range(len(lines)):
        if(i not in inliers_z):
            outliers.append(lines[i])
    lines_x, lines_y=ClusterLines(outliers)

    vp_x, inliers_x=FindVP(lines_x, K_apprx, 2, 8000)
    vp_y, inliers_y=FindVP(lines_y, K_apprx, 12, 6000)


	#####################################################################
    # Calibrate K 
    K=CalibrateCamera(vp_x, vp_y, vp_z)

	#####################################################################
    # Compute the rectiﬁcation homography
    H=GetRectificationH(K, vp_x, vp_y, vp_z)

	#####################################################################
    # Rectify the input image and vanishing points
    im_warped=ImageWarping(im, H)

    vp_z_warped=H@[vp_z[0], vp_z[1], 1]
    vp_z_warped=vp_z_warped//vp_z_warped[2]
    vp_z=[int(vp_z_warped[0]), int(vp_z_warped[1])]

    vp_x_warped=H@[vp_x[0], vp_x[1], 1]
    vp_x_warped=vp_x_warped//vp_x_warped[2]
    vp_x=[int(vp_x_warped[0]), int(vp_x_warped[1])]

    vp_y_warped=H@[vp_y[0], vp_y[1], 1]
    vp_y_warped=vp_y_warped//vp_y_warped[2]
    vp_y=[int(vp_y_warped[0]), int(vp_y_warped[1])]

	#####################################################################
    # Construct 3D representation of the scene using a box model
    W = 1
    aspect_ratio = 2.5
    near_depth = 0.4
    far_depth = 4

    U11, U12, U21, U22, V11, V12, V21, V22 = ConstructBox(K, vp_x, vp_y, vp_z, W, aspect_ratio, near_depth, far_depth)
    
	#####################################################################
    # The sequence of camera poses
    R_list = []
    C_list = []
    # Camera pose 1
    R_list.append(np.eye(3))
    C_list.append(np.zeros((3,)))
    # Camera pose 2
    R_list.append(np.asarray([
        [np.cos(np.pi/12), 0, -np.sin(np.pi/12)],
        [0, 1, 0],
        [np.sin(np.pi/12), 0, np.cos(np.pi/12)]
    ]))
    C_list.append(np.asarray([0, 0, 0.5]))
    # Camera pose 3
    R_list.append(np.asarray([
        [np.cos(np.pi/4), 0, -np.sin(np.pi/4)],
        [0, 1, 0],
        [np.sin(np.pi/4), 0, np.cos(np.pi/4)]
    ]))
    C_list.append(np.asarray([-0.1, 0, 0.4]))
    # Camera pose 4
    R_list.append(np.asarray([
        [np.cos(-np.pi/4), 0, -np.sin(-np.pi/4)],
        [0, 1, 0],
        [np.sin(-np.pi/4), 0, np.cos(-np.pi/4)]
    ]))
    C_list.append(np.asarray([0.2, 0.1, 0.6]))


	#####################################################################
    # Render images from the interpolated virtual camera poses
    w_list=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    vx=[]
    vy=[]

    #Construct vx
    for i in range(im_h):
        vx.append(np.arange(0, im_w))
    vx=np.asarray(vx)

    #Construct vy
    vy=np.ones((im_h, im_w))
    for i in range(vy.shape[0]):
        vy[i]=vy[i]*i


    #For the number of transitions
    for i in range(len(R_list)-1):
        R1=R_list[i]
        C1=C_list[i]
        R2=R_list[i+1]
        C2=C_list[i+1]

        #For the number of time instants
        for w in w_list:
            Ri, Ci = InterpolateCameraPose(R1, C1, R2, C2, w)

            #Compute homography and mask for all 5 walls
            H_back, mask_back = GetPlaneHomography(U21, U22, U11, K, Ri, Ci, vx, vy)
            H_left, mask_left = GetPlaneHomography(V21, U21, V11, K, Ri, Ci, vx, vy)
            H_right, mask_right = GetPlaneHomography(U22, V22, U12, K, Ri, Ci, vx, vy)
            H_top, mask_top = GetPlaneHomography(V21, V22, U21, K, Ri, Ci, vx, vy)
            H_bottom, mask_bottom = GetPlaneHomography(U11, U12, V11, K, Ri, Ci, vx, vy)

            canvas = np.zeros((im_h, im_w, 3), dtype=np.uint8)
            for v in range(canvas.shape[0]):
                for u in range(canvas.shape[1]):
                    if mask_back[v][u]==1:
                        x2=[u, v, 1]
                        x1=np.linalg.inv(H_back)@x2
                        x1=x1//x1[2]
                        if(x1[0]<im_warped.shape[1] and x1[0]>=0 and x1[1]<im_warped.shape[0] and x1[1]>=0):
                            canvas[v][u]=im_warped[math.floor(x1[1])][math.floor(x1[0])]
                    elif mask_left[v][u]==1:
                        x2=[u, v, 1]
                        x1=np.linalg.inv(H_back)@x2
                        x1=x1//x1[2]
                        if(x1[0]<im_warped.shape[1] and x1[0]>=0 and x1[1]<im_warped.shape[0] and x1[1]>=0):
                            canvas[v][u]=im_warped[math.floor(x1[1])][math.floor(x1[0])]
                    elif mask_right[v][u]==1:
                        x2=[u, v, 1]
                        x1=np.linalg.inv(H_back)@x2
                        x1=x1//x1[2]
                        if(x1[0]<im_warped.shape[1] and x1[0]>=0 and x1[1]<im_warped.shape[0] and x1[1]>=0):
                            canvas[v][u]=im_warped[math.floor(x1[1])][math.floor(x1[0])]
                    elif mask_top[v][u]==1:
                        x2=[u, v, 1]
                        x1=np.linalg.inv(H_back)@x2
                        x1=x1//x1[2]
                        if(x1[0]<im_warped.shape[1] and x1[0]>=0 and x1[1]<im_warped.shape[0] and x1[1]>=0):
                            canvas[v][u]=im_warped[math.floor(x1[1])][math.floor(x1[0])]
                    elif mask_bottom[v][u]==1:
                        x2=[u, v, 1]
                        x1=np.linalg.inv(H_back)@x2
                        x1=x1//x1[2]
                        if(x1[0]<im_warped.shape[1] and x1[0]>=0 and x1[1]<im_warped.shape[0] and x1[1]>=0):
                            canvas[v][u]=im_warped[math.floor(x1[1])][math.floor(x1[0])]
                    else:
                        continue
            cv2.imwrite('output_{}.png'.format(i+w), canvas)


