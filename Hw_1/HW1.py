import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random
import math


def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    """
    
    x1=[]
    x2=[]
    image_2_to_1=[]
    #Ratio test using Nearest Neighbors training on image 2 and querying image 1
    nbrs=NearestNeighbors(n_neighbors=2).fit(des2)
    distances, indices=nbrs.kneighbors(des1)
    for i in range(indices.shape[0]):
        #Get our matches for this index and their corresponding distances
        matches=indices[i]
        dists=distances[i]

        #According to documentation for NearestNeighbors, result points are not necessarily in order of distance
        #Let's order these as d1 and d2
        d1=0
        d2=0
        d1_index=0

        if(dists[0]<dists[1]):
            d1=dists[0]
            d2=dists[1]
            d1_index=0
        else:
            d1=dists[1]
            d2=dists[0]
            d1_index=1

        #Lowe's ratio test
        #Ensure that the two points are discriminant enough according to Lowe's paper Distinctice Image Features from Scale-Invariant Keypoints
        if(d1<d2*0.7):
            image_2_to_1.append((i,matches[d1_index]))


    #To ensure bi-directional consistency we have to do it the other way around as well
    #Might've been good to put this small part in its own method if methods weren't already pre-defined
    nbrs=NearestNeighbors(n_neighbors=2).fit(des1)
    distances, indices=nbrs.kneighbors(des2)
    image_1_to_2=[]
    for i in range(indices.shape[0]):
        #Get our matches for this index and their corresponding distances
        matches=indices[i]
        dists=distances[i]

        #According to documentation for NearestNeighbors, result points are not necessarily in order of distance
        #Let's order these as d1 and d2
        d1=0
        d2=0
        d1_index=0

        if(dists[0]<dists[1]):
            d1=dists[0]
            d2=dists[1]
            d1_index=0
        else:
            d1=dists[1]
            d2=dists[0]
            d1_index=1

        #Lowe's ratio test
        #Ensure that the two points are discriminant enough according to Lowe's paper Distinctice Image Features from Scale-Invariant Keypoints
        if(d1<d2*0.7):
            image_1_to_2.append((matches[d1_index],i))

    #Let's compare and only keep those that are in both to ensure bi-directional consistency
    for i in range(len(image_2_to_1)):
        if(image_2_to_1[i] in image_1_to_2):
            x1.append(loc1[image_2_to_1[i][0]])
            x2.append(loc2[image_2_to_1[i][1]])


    return np.array(x1), np.array(x2)


def EstimateH(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the homography between images using RANSAC
    
    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    H : ndarray of shape (3, 3)
        The estimated homography
    inlier : ndarray of shape (k,)
        The inlier indices
    """
    H=None
    inlier=[]
    #Iterate through ransac algorithm given number of times
    for n in range(ransac_n_iter):
        sample_x1=[]
        sample_x2=[]
        #Sample 4 random points
        indices=random.sample(range(0, len(x1)), 4)
        for indx in indices:
            sample_x1.append(x1[indx])
            sample_x2.append(x2[indx])

        #From these 4 random points, let's build our homography matrix
        A=[]
        b=[]
        for i in range(len(sample_x1)):
            u=sample_x1[i][0]
            u_prime=sample_x2[i][0]
            v=sample_x1[i][1]
            v_prime=sample_x2[i][1]

            b.append(u_prime)
            b.append(v_prime)

            A.append([u, v, 1, 0, 0, 0, -u*u_prime, -v*u_prime])
            A.append([0, 0, 0, u, v, 1, -u*v_prime, -v*v_prime])

        A=np.array(A)
        b=np.array(b)

        #Ax=b OR x=(A^TA)^-1A^Tb
        a_t_a=np.transpose(A)@A
        #Added a small amount of noise along the diagonal to avoid singularity issues
        a_t_a=a_t_a+np.eye(A.shape[1])*1e-6
        x=np.linalg.inv(a_t_a)@np.transpose(A)
        x=x@b
        x=np.append(x,1)
        #Reshape to 3x3
        H_est=np.reshape(x, (3, 3))

        #Now, let's go through the points to determine which of our keypoints are inliers of our estimated homography matrix
        inlier_est=[]
        for i in range(len(x1)):
            p1=[x1[i][0], x1[i][1], 1]
            p2=[x2[i][0], x2[i][1], 1]

            #Estimate point 2 by multiplying our estimated H by point 1
            est=H_est@(p1)
            #normalize by the scale
            est=est/est[2]
            #Calculate the error as the L2 norm
            err=np.linalg.norm(p2-est)

            #If the error is less than the threshold, append this point to our list of inliers
            if(err<ransac_thr):
                inlier_est.append(i)

        #If the length of our inliers is longer than our current best this becomes our new best        
        if len(inlier_est)>len(inlier):
            inlier=inlier_est
            H=H_est




    return H, np.array(inlier)


def EstimateR(H, K):
    """
    Compute the relative rotation matrix
    
    Parameters
    ----------
    H : ndarray of shape (3, 3)
        The estimated homography
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters

    Returns
    -------
    R : ndarray of shape (3, 3)
        The relative rotation matrix from image 1 to image 2
    """
    #R = lambda K^-1HK where lambda is some scale
    R_scaled=((np.linalg.inv(K)@H)@K)
    #Ensure orthogonal matrix
    u, s, v=np.linalg.svd(R_scaled)
    R=u@v

    #The determinant of a rotation matrix must be 1
    if(np.linalg.det(R)<0):
        return -R
    else:
        return R


def ConstructCylindricalCoord(Wc, Hc, K):
    """
    Generate 3D points on the cylindrical surface
    
    Parameters
    ----------
    Wc : int
        The width of the canvas
    Hc : int
        The height of the canvas
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters of the source images

    Returns
    -------
    p : ndarray of shape (Hc, Hc, 3)
        The 3D points corresponding to all pixels in the canvas
    """
    #Get focal length
    f=K[0][0]
    p=np.zeros((Hc, Wc, 3))
    for h in range(p.shape[0]):
        for w in range(p.shape[1]):
            #The x coordinate is f sin(phi) where phi=w(2pi/Wc)
            p[h][w][0]=f*np.sin(w*(2*math.pi/Wc))
            #The y coordinate is h-Hc/2
            p[h][w][1]=h-(Hc/2)
            #The z coordinate is f cos(phi) where phi=w(2pi/Wc)
            p[h][w][2]=f*np.cos(w*(2*math.pi/Wc))
    return p


def Projection(p, K, R, W, H):
    """
    Project the 3D points to the camera plane
    
    Parameters
    ----------
    p : ndarray of shape (Hc, Wc, 3)
        A set of 3D points that correspond to every pixel in the canvas image
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters
    R : ndarray of shape (3, 3)
        The rotation matrix
    W : int
        The width of the source image
    H : int
        The height of the source image

    Returns
    -------
    u : ndarray of shape (Hc, Wc, 2)
        The 2D projection of the 3D points
    mask : ndarray of shape (Hc, Wc)
        The corresponding binary mask indicating valid pixels
    """
    u=np.zeros((p.shape[0], p.shape[1], 2))
    mask=np.ones((p.shape[0],p.shape[1]))

    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            #Calculate our 2D projection of the 3D points
            x=K@R@p[i][j]
            #Normalize our x and y by the scale, which is z of our projection
            u[i][j][0]=x[0]/x[2]
            u[i][j][1]=x[1]/x[2]

            #Check to see if the projected point is beyond the image boundary
            if(u[i][j][1]>H or u[i][j][0]>W or u[i][j][0]<0 or u[i][j][1]<0):
                mask[i][j]=0

            #Check to see if the 3D point is behind the camera
            if (x[2]<0):
                mask[i][j]=0
    
    return u, mask


def WarpImage2Canvas(image_i, u, mask_i):
    """
    Warp the image to the cylindrical canvas
    
    Parameters
    ----------
    image_i : ndarray of shape (H, W, 3)
        The i-th image with width W and height H
    u : ndarray of shape (Hc, Wc, 2)
        The mapped 2D pixel locations in the source image for pixel transport
    mask_i : ndarray of shape (Hc, Wc)
        The valid pixel indicator

    Returns
    -------
    canvas_i : ndarray of shape (Hc, Wc, 3)
        the canvas image generated by the i-th source image
    """
    #Follow the inverse warping steps
    canvas_i=np.zeros((u.shape[0],u.shape[1],3))
    for i in range(canvas_i.shape[0]):
        for j in range(canvas_i.shape[1]):
            #Get the x and y values of our inverse projection
            x,y=u[i][j]
            #Check our mask values
            if(mask_i[i][j]==1):
                #Add it to our canvas if it is a valid point
                #We have to be careful about making sure we put x and y in the right order
                #Up until this point we've been referring to points as [x,y], but our image is a 2D matrix, where
                #the points are (y,x)
                canvas_i[i][j]=image_i[int(y)][int(x)]
            else:
                canvas_i[i][j]=[0,0,0]

    return canvas_i



def UpdateCanvas(canvas, canvas_i, mask_i):
    """
    Update the canvas with the new warped image
    
    Parameters
    ----------
    canvas : ndarray of shape (Hc, Wc, 3)
        The previously generated canvas
    canvas_i : ndarray of shape (Hc, Wc, 3)
        The i-th canvas
    mask_i : ndarray of shape (Hc, Wc)
        The mask of the valid pixels on the i-th canvas

    Returns
    -------
    canvas : ndarray of shape (Hc, Wc, 3)
        The updated canvas image
    """

    #If our points are valid, then copy them over to our canvas
    for i in range(canvas.shape[0]):
        for j in range(canvas.shape[1]):
            if(mask_i[i][j]==1):
                canvas[i][j]=canvas_i[i][j]
    return canvas

if __name__ == '__main__':
    ransac_n_iter = 500
    ransac_thr = 3
    K = np.asarray([
        [320, 0, 480],
        [0, 320, 270],
        [0, 0, 1]
    ])

    # Read all images
    im_list = []
    for i in range(1, 9):
        im_file = '{}.jpg'.format(i)
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_list.append(im)

    rot_list = []
    rot_list.append(np.eye(3))
    for i in range(len(im_list) - 1):
        # Load consecutive images I_i and I_{i+1}
        I_1=im_list[i]
        I_2=im_list[i+1]
		
        # Extract SIFT features
        sift=cv2.xfeatures2d.SIFT_create()
        kp1, des1=sift.detectAndCompute(I_1,None)
        kp2, des2=sift.detectAndCompute(I_2,None)
        loc1=[kp1[index].pt for index in range(len(kp1))]
        loc2=[kp2[index].pt for index in range(len(kp2))]
		
        # Find the matches between two images (x1 <--> x2)
        x1, x2 = MatchSIFT(loc1, des1, loc2, des2)

        # Estimate the homography between images using RANSAC
        H, inlier = EstimateH(x1, x2, ransac_n_iter, ransac_thr)

        # Compute the relative rotation matrix R
        R = EstimateR(H, K)
        #We premultiply our matrix by the last element in the rotation matrix list to get the rotation matrix from 0 to i+1
        R_new=R@rot_list[-1]
		
        rot_list.append(R_new)

    Him = im_list[0].shape[0]
    Wim = im_list[0].shape[1]
    
    Hc = Him
    Wc = len(im_list) * Wim // 2
	
    canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    p = ConstructCylindricalCoord(Wc, Hc, K)

    fig = plt.figure('HW1')
    plt.axis('off')
    plt.ion()
    plt.show()
    for i, (im_i, rot_i) in enumerate(zip(im_list, rot_list)):
        # Project the 3D points to the i-th camera plane
        u, mask_i = Projection(p, K, rot_i, Wim, Him)
        # Warp the image to the cylindrical canvas
        canvas_i = WarpImage2Canvas(im_i, u, mask_i)
        # Update the canvas with the new warped image
        canvas = UpdateCanvas(canvas, canvas_i, mask_i)
        plt.imshow(canvas)
        plt.savefig('output_{}.png'.format(i+1), dpi=600, bbox_inches = 'tight', pad_inches = 0)