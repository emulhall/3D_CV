import os
from PIL import Image
from utils import *
from rays import *
from tsdf import *
import numpy as np
import time


def ProcessDepthImage(file_name, depth_factor):
    """
    Process Depth Image

    Parameters
    ----------
    filename : string
        input depth file
    depth_factor : float
        normalized depth value

    Returns
    -------
    depth_img : ndarray of shape (480, 640)
        filtered depth image
    """
    depth_img = Image.open(file_name).convert('F')
    depth_img = np.array(depth_img) / depth_factor
    scale = np.max(depth_img)
    d_ = depth_img / scale
    d_ = cv2.bilateralFilter(d_, 5, 3, 0.01)
    depth_img = d_ * scale
    return depth_img


def Get3D(depth, K):
    """
        Inverse Projection - create point cloud from depth image

        Parameters
        ----------
        depth : ndarray of shape (H, W)
            filtered depth image
        K : ndarray of shape (3, 3)
            Intrinsic parameters
        Returns
        -------
        point : ndarray of shape (3, H, W)
            Point cloud from depth image
        normal : ndarray of shape (3, H, W)
            Surface normal
    """

    #Build up our 3D points
    vv,uu = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing='ij')
    u=np.vstack((uu.flatten(), vv.flatten()))
    u=np.vstack((u,np.ones(u.shape[1])))

    #Move to 3D by multiplying by inverse camera intrinsics
    u=np.linalg.inv(K)@u
    u=np.reshape(u, (3,depth.shape[0],depth.shape[1]))
    point=depth*u

    #Calculate offsets
    v_offset = np.arange(1,depth.shape[0]+1)
    u_offset = np.arange(1,depth.shape[1]+1)

    #Reflection padding for the cases out of image bounds
    v_offset[-1]=v_offset[-3]
    u_offset[-1]=u_offset[-3]

    vv_offset, uu_offset = np.meshgrid(v_offset, u_offset, indexing='ij')

    normal=Normalize(np.cross((point[:,vv,uu_offset]-point[:,vv,uu]),(point[:,vv_offset,uu]-point[:,vv,uu]), axis=0),dim=0)

    return point, normal


def CreateTSDF(depth, T, voxel_param, K):
    """
        CreateTSDF : VoxelParams class' member function
            Compute distance of each voxel w.r.t a camera

        Parameters
        ----------
        depth : ndarray of shape (H, W)
            Filtered depth image
        T : ndarray of shape (4, 4)
            Transformation that brings camera to world coordinate
        voxel_param : an instance of voxel parameter VoxelParams
        K : ndarray of shape (3, 3)
                Intrinsic parameters
        Returns
        -------
        tsdf : TSDF
            An instance of TSDF with value computed as projective TSDF
    """
    #Convert cell locations from world coordinate system to camera coordinate system
    vw=np.vstack((np.reshape(voxel_param.voxel_x.flatten(), (1,-1)), np.reshape(voxel_param.voxel_y.flatten(), (1,-1)), np.reshape(voxel_param.voxel_z.flatten(), (1,-1)), np.ones((1,len(voxel_param.voxel_x.flatten())))))
    vc=np.linalg.inv(T)@vw
    vc=vc[:3,:]/vc[3,:]

    #Build up our 3D points by projecting onto the camera plane
    u=K@vc
    u=u/u[2]
    coords=np.floor(u[:2])

    #Move to 3D
    u=np.linalg.inv(K)@u

    #Mask out pixels outside of image boundaries
    mask1=np.logical_and(coords[0,:]<depth.shape[1], coords[1,:]<depth.shape[0])
    mask2=np.logical_and(coords[0,:]>=0, coords[1,:]>=0)
    mask3=np.logical_and(mask1, mask2)
    mask=np.vstack((mask3,mask3))

    #For now we'll just set invalid pixels to 0, but don't worry we'll take care of that later
    masked=np.where(mask,coords,0)
    masked=masked.astype(int)
    pc=depth[masked[1,:],masked[0,:]]*u

    SDF=np.linalg.norm(pc, axis=0)-np.linalg.norm(vc,axis=0)
    #Now let's mask out those pixels that were invalid earlier by making them far larger than the truncation threshold so they get masked out in the next step
    SDF=np.where(mask3,SDF,voxel_param.trunc_thr*1e6)
    SDF=np.reshape(SDF, (voxel_param.voxel_x.shape))


    tsdf=TSDF(voxel_param=voxel_param,sdf=SDF,valid_voxel=np.reshape(mask3, (voxel_param.voxel_x.shape)))
    return tsdf


def ComputeTSDFNormal(point, tsdf, voxel_param):
    """
        ComputeTSDFNormal : Compute surface normal from tsdf


        Parameters
        ----------
        point : ndarray of shape (3, H, W)
            Point cloud predicted by casting rays to tsdf
        voxel_param : an instance of voxel parameter VoxelParams
        tsdf : an instance of TSDF

        Returns
        -------
        normal : ndarray of shape (3, H, W)
            Surface normal at each 3D point indicated by 'point' variable

        Note
        -------
        You can use scipy.ndimage.map_coordinates to interpolate ndarray
    """

    #Flatten our point cloud for calculations
    point_flat=np.reshape(point, (3,-1))

    #Calculate offsets
    delta=0.01
    x_offset=point_flat+delta*np.reshape([1,0,0], (3,1))
    y_offset=point_flat+delta*np.reshape([0,1,0], (3,1))
    z_offset=point_flat+delta*np.reshape([0,0,1], (3,1))

    #Move to voxel space
    point_flat=(point_flat-np.reshape(voxel_param.voxel_origin, (3,1)))/voxel_param.vox_size
    x_offset=(x_offset-np.reshape(voxel_param.voxel_origin, (3,1)))/voxel_param.vox_size
    y_offset=(y_offset-np.reshape(voxel_param.voxel_origin, (3,1)))/voxel_param.vox_size
    z_offset=(z_offset-np.reshape(voxel_param.voxel_origin, (3,1)))/voxel_param.vox_size

    #Swap x and y because map coordinates expects (x,y,z)
    point_flat=np.vstack((np.reshape(point_flat[1,:], (1,-1)), np.reshape(point_flat[0,:], (1,-1)), np.reshape(point_flat[2,:],(1,-1))))
    x_offset=np.vstack((np.reshape(x_offset[1,:], (1,-1)), np.reshape(x_offset[0,:], (1,-1)), np.reshape(x_offset[2,:],(1,-1))))
    y_offset=np.vstack((np.reshape(y_offset[1,:], (1,-1)), np.reshape(y_offset[0,:], (1,-1)), np.reshape(y_offset[2,:],(1,-1))))
    z_offset=np.vstack((np.reshape(z_offset[1,:], (1,-1)), np.reshape(z_offset[0,:], (1,-1)), np.reshape(z_offset[2,:],(1,-1))))

    #Interpolate
    fp=scipy.ndimage.map_coordinates(tsdf.value,point_flat)

    #Calculate normals
    nx=scipy.ndimage.map_coordinates(tsdf.value,x_offset)-fp
    ny=scipy.ndimage.map_coordinates(tsdf.value,y_offset)-fp
    nz=scipy.ndimage.map_coordinates(tsdf.value,z_offset)-fp

    #Stack and reshape
    normal=Normalize(np.vstack((nx,ny,nz)),dim=0)
    normal=np.reshape(normal, (3,point.shape[1],point.shape[2]))

    return normal


def FindCorrespondence(T, point_pred, normal_pred, point, normal, valid_rays, K, e_p, e_n):
    """
    Find Correspondence between current tsdf and input image's depth/normal

    Parameters
    ----------
    T : ndarray of shape (4, 4)
        Transformation of camera to world coordinate
    point_pred : ndarray of shape (3, H, W)
        Point cloud from ray casting the tsdf
    normal_pred : ndarray of shape (3, H, W)
        Surface normal from tsdf
    point : ndarray of shape (3, H, W)
        Point cloud extracted from depth image
    normal : ndarray of shape (3, H, W)
        Surface normal extracted from depth image
    valid_rays : ndarray of shape (H, W)
        Valid ray casting pixels
    K : ndarray of shape (3, 3)
        Intrinsic parameters
    e_p : float
        Threshold on distance error
    e_n : float
        Threshold on cosine angular error
    Returns
    -------
    Correspondence point of 4 variables
    p_pred, n_pred, p, n : ndarray of shape (3, m)
        Inlier point_pred, normal_pred, point, normal

    """

    #Calculate the projection of the predicted points
    u_hat=K@T[:3,:3].T@(np.reshape(point_pred, (3,-1))-np.reshape(T[:3,3], (3,1)))

    #Normalize
    coords=u_hat[:2]/u_hat[2]

    #Determine points outside of the image boundaries and greater than 0
    mask=np.logical_and(np.logical_and(coords[0,:]<point_pred.shape[2], coords[1,:]<point_pred.shape[1]), np.logical_and(coords[0,:]>=0, coords[1,:]>=0))

    #Combine valid rays and mask out points behind the camera
    mask=np.logical_and(mask, np.logical_and(u_hat[2,:]>0, valid_rays.flatten()))

    coords=coords[:,mask]

    #map_coordinates expects (v,u)
    vu=np.vstack((np.reshape(coords[1,:], (1,-1)), np.reshape(coords[0,:], (1,-1))))

    #Interpolate point and normal using u_hat coordinates
    p_c_x=np.reshape(scipy.ndimage.map_coordinates(point[0], vu), (1,-1))
    p_c_y=np.reshape(scipy.ndimage.map_coordinates(point[1], vu), (1,-1))
    p_c_z=np.reshape(scipy.ndimage.map_coordinates(point[2], vu), (1,-1))

    n_c_x=np.reshape(scipy.ndimage.map_coordinates(normal[0], vu), (1,-1))
    n_c_y=np.reshape(scipy.ndimage.map_coordinates(normal[1], vu), (1,-1))
    n_c_z=np.reshape(scipy.ndimage.map_coordinates(normal[2], vu), (1,-1))

    #Stack x,y,z to form p_c and n_c
    p_c=np.vstack((p_c_x, p_c_y, p_c_z))
    n_c=np.vstack((n_c_x, n_c_y, n_c_z))

    #Rotate and translate to convert to world
    p=(T[:3,:3]@p_c)+np.reshape(T[:3,3], (3,1))
    n=T[:3,:3]@n_c

    point_pred=np.reshape(point_pred, (3,-1))
    normal_pred=np.reshape(normal_pred, (3,-1))

    p_pred=point_pred[:,mask]
    n_pred=normal_pred[:,mask]

    #Filter out the ones that are too far away and with different surface normals
    mask=np.logical_and(np.linalg.norm(p_pred-p,axis=0)<e_p,(np.sum(n_pred*n,axis=0))>e_n)

    #Final filtering
    p_pred=p_pred[:,mask]
    n_pred=n_pred[:,mask]
    p=p[:,mask]
    n=n[:,mask]

    return p_pred, n_pred, p, n


def SolveForPose(p_pred, n_pred, p):
    """
        Solve For Incremental Update Pose

        Parameters
        ----------
        p_pred : ndarray of shape (3, -1)
            Inlier tsdf point
        n_pred : ndarray of shape (3, -1)
            Inlier tsdf surface normal
        p : ndarray of shape (3, -1)
            Inlier depth image's point
        Returns
        -------
        deltaT : ndarray of shape (4, 4)
            Incremental updated pose matrix
    """
    b=np.sum(n_pred*(p_pred-p),axis=0)
    skew=Vec2Skew(p.T)
    A=[]
    for i in range(skew.shape[0]):
    	A.append(np.reshape(np.reshape(n_pred[:,i], (1,3))@np.concatenate((skew[i,:,:], np.eye(3)), axis=1), (-1)))

    A=np.array(A)

    #Ax=b OR x=(A^TA)^-1A^Tb
    x=np.linalg.inv(A.T@A)@A.T@b

    deltaT=np.array([[1, x[2], -x[1], x[3]], 
    	[-x[2], 1, x[0], x[4]],
    	[x[1], -x[0], 1, x[5]],
    	[0,0,0,1]])

    if(np.linalg.det(deltaT[:3,:3])<0):
    	deltaT=-deltaT

    return deltaT


def FuseTSDF(tsdf, tsdf_new):
    """
        FuseTSDF : Fusing 2 tsdfs

        Parameters
        ----------
        tsdf, tsdf_new : TSDFs
        Returns
        -------
        tsdf : TSDF
            Fused of tsdf and tsdf_new
    """
    weight_norm=(tsdf.weight+tsdf_new.weight)
    tsdf.value[weight_norm>0]=(tsdf.weight[weight_norm>0]*tsdf.value[weight_norm>0]+tsdf_new.weight[weight_norm>0]*tsdf_new.value[weight_norm>0])/weight_norm[weight_norm>0]
    tsdf.weight=tsdf.weight+tsdf_new.weight

    return tsdf


if __name__ == '__main__':
    DEPTH_FOLDER = 'depth_images'
    OUTPUT_FOLDER = 'results'
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    voxel_param = VoxelParams(3, 256)
    fx = 525.0
    fy = 525.0
    cx = 319.5
    cy = 239.5
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1]])
    depth_factor = 5000.
    n_iters = 3
    e_p = voxel_param.vox_size * 10.0
    e_n = np.cos(np.pi / 3.0)

    T_cur = np.eye(4)
    depth_file_list = open(os.path.join(DEPTH_FOLDER, 'filelist.list'), 'r').read().split('\n')
    depth_img = ProcessDepthImage(os.path.join(DEPTH_FOLDER, depth_file_list[0]), depth_factor)
    tsdf = CreateTSDF(depth_img, T_cur, voxel_param, K)
    SaveTSDFtoMesh('%s/mesh_initial.ply' % OUTPUT_FOLDER, tsdf)


    rays = ImageRays(K, voxel_param, depth_img.shape)
    for i_frame in range(1, len(depth_file_list)-1):
        print('process frame ', i_frame)

        point_pred, valid_rays = rays.cast(T_cur, voxel_param, tsdf)
        SavePointDepth('%s/pd_%02d.ply' % (OUTPUT_FOLDER, i_frame), point_pred, valid_rays)

        normal_pred = -ComputeTSDFNormal(point_pred, tsdf, voxel_param)
        SavePointNormal('%s/pn_%02d.ply' % (OUTPUT_FOLDER, i_frame), point_pred, normal_pred, valid_rays)

        depth_img = ProcessDepthImage(os.path.join(DEPTH_FOLDER, depth_file_list[i_frame]), depth_factor)
        point, normal = Get3D(depth_img, K)

        for i in range(n_iters):
            p_pred, n_pred, p, n = FindCorrespondence(T_cur, point_pred, normal_pred,
                                                      point, normal, valid_rays, K, e_p, e_n)

            deltaT = SolveForPose(p_pred, n_pred, p)

            # Update pose
            T_cur = deltaT @ T_cur
            u, s, vh = np.linalg.svd(T_cur[:3, :3])
            R = u @ vh
            R *= np.linalg.det(R)
            T_cur[:3, :3] = R


        tsdf_new = CreateTSDF(depth_img, T_cur, voxel_param, K)
        tsdf = FuseTSDF(tsdf, tsdf_new)
        SaveTSDFtoMesh('%s/mesh_%02d.ply' % (OUTPUT_FOLDER, i_frame), tsdf)

    SaveTSDFtoMesh('%s/mesh_final.ply' % OUTPUT_FOLDER, tsdf, viz=True)



