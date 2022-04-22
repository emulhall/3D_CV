import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random



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
    x1 : ndarray of shape (m, 2)
        The matched keypoint locations in image 1
    x2 : ndarray of shape (m, 2)
        The matched keypoint locations in image 2
    ind1 : ndarray of shape (m,)
        The indices of x1 in loc1
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
    ind1=[]
    for i in range(len(image_2_to_1)):
        if(image_2_to_1[i] in image_1_to_2):
            x1.append(loc1[image_2_to_1[i][0]])
            x2.append(loc2[image_2_to_1[i][1]])
            ind1.append(image_2_to_1[i][0])


    return np.array(x1), np.array(x2), np.array(ind1)



def EstimateE(x1, x2):
    """
    Estimate the essential matrix, which is a rank 2 matrix with singular values
    (1, 1, 0)

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    """
    
    A=[]
    for i in range(len(x1)):
        A.append([x1[i,0]*x2[i,0], x1[i,1]*x2[i,0], x2[i,0], x1[i,0]*x2[i,1], x1[i,1]*x2[i,1], x2[i,1], x1[i,0], x1[i,1], 1])

    #Get the null space of A, which is equal to the last row of V^T
    u_initial, s_initial, v_t_initial = np.linalg.svd(np.array(A))
    E_messy=np.reshape(v_t_initial[-1], (3,3))

    #Clean up to ensure rank 2 and s=[1,1,0]
    u, s_messy, v_t = np.linalg.svd(E_messy)
    s_clean = [1,1,0]
    E=(u*s_clean)@v_t
    return E



def EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the essential matrix robustly using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    inlier : ndarray of shape (k,)
        The inlier indices
    """
    
    E=None
    inlier=[]

    #Iterate through ransac algorithm given number of times
    for n in range(ransac_n_iter):

        #Sample 8 random points
        indices=random.sample(range(0,len(x1)), 8)
        sample_x1=x1[indices]
        sample_x2=x2[indices]

        E_est = EstimateE(sample_x1, sample_x2)

        #Find inliers given the estimated E using v^TFu=0
        #Rearrange to add a column of ones
        u=x1.T
        u=np.vstack((u, np.ones(u.shape[1])))
        v=np.hstack((x2, np.ones((len(x2), 1))))

        #Multiply Eu
        Eu=E_est@u
        Eu=Eu.T

        #Multiply by v^T
        vtEu=v*Eu
        vtEu=np.sum(vtEu, axis=1)

        v_norm=np.linalg.norm(v, axis=1)

        Eu_norm=np.linalg.norm(Eu, axis=1)

        normalization=v_norm*Eu_norm

        score=np.abs(vtEu/normalization)


        #Find indices below threshold
        inlier_est=np.argwhere(score<ransac_thr)

        #Check to see if this E is a better estiamte based on the number of inliers
        if len(inlier_est)>len(inlier):
            inlier=np.reshape(inlier_est, (len(inlier_est)))
            E=E_est


    return E, inlier



def BuildFeatureTrack(Im, K):
    """
    Build feature track

    Parameters
    ----------
    Im : ndarray of shape (N, H, W, 3)
        Set of N images with height H and width W
    K : ndarray of shape (3, 3)
        Intrinsic parameters

    Returns
    -------
    track : ndarray of shape (N, F, 2)
        The feature tensor, where F is the number of total features
    """
    des_list=[]
    loc_list=[]

    for i in range(len(Im)):
        #Extract SIFT features
        sift=cv2.xfeatures2d.SIFT_create()
        kp, des=sift.detectAndCompute(Im[i], None)
        loc=[kp[index].pt for index in range(len(kp))]
        des_list.append(des)
        loc_list.append(loc)

    track = []
    for i in range(len(Im)-1):
        #Initialize track_i with some very large number for F
        track_i=np.ones((Im.shape[0], len(des_list[i]), 2))*(-1.0)
        for j in range(i+1, len(Im)):
            #Match SIFT features
            x1, x2, ind1 = MatchSIFT(loc_list[i], des_list[i], loc_list[j], des_list[j])

            #Normalize coordinates
            x1_3D=x1.T
            x1_3D=np.vstack((x1_3D, np.ones(x1_3D.shape[1])))
            x1_3D = np.linalg.inv(K)@x1_3D
            x1_3D=x1_3D.T
            x1_3D=x1_3D[:,:2]

            x2_3D=x2.T
            x2_3D=np.vstack((x2_3D, np.ones(x2_3D.shape[1])))
            x2_3D = np.linalg.inv(K)@x2_3D
            x2_3D=x2_3D.T
            x2_3D=x2_3D[:,:2]

            #Find inlier matches using the essential matrix
            E, inliers = EstimateE_RANSAC(x1_3D, x2_3D, 1000, 1e-2)

            #Update track_i
            track_i[i,ind1[inliers],:]=x1_3D[inliers]
            track_i[j,ind1[inliers],:]=x2_3D[inliers]

        #Remove features in track_i that have not been matched in any image
        keep=[]
        for j in range(track_i.shape[1]):
            for k in range(i+1,track_i.shape[0]):
                if(track_i[k,j,0]!=-1 or track_i[k,j,1]!=-1):
                    keep.append(j)
                    break
        track_i=track_i[:,keep,:]

        #track=track U track_i
        if(i==0):
            track=track_i
        else:
            track=np.concatenate((track, track_i), axis=1)

    return track