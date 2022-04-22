import scipy.io
import scipy.ndimage
from tsdf import *
from utils import *
import time


class ImageRays:
    def __init__(self, K, voxel_param=VoxelParams(3, 256), im_size=np.array([480, 640])):
        """
            ImageRays : collection of geometric parameters of rays in an image

            Parameters
            ----------
            K : ndarray of shape (3, 3)
                Intrinsic parameters
            voxel_param : an instance of voxel parameter VoxelParams
            im_size: image size

            Class variables
            -------
            im_size : ndarray of value [H, W]
            rays_d : ndarray of shape (3, H, W)
                Direction of each pixel ray in an image with intrinsic K and size [H, W]
            lambda_step : ndarray (-1, )
                Depth of casted point along each ray direction
        """
        self.im_size = im_size
        h, w = im_size
        xx, yy = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
        uv1 = np.linalg.inv(K) @ np.reshape(np.concatenate((xx, yy, np.ones_like(xx)), axis=0), (3, h * w))
        self.rays_d = uv1 / np.linalg.norm(uv1, axis=0, keepdims=True)
        self.lambda_step = np.arange(voxel_param.vox_size, voxel_param.phy_len, voxel_param.vox_size)


    def cast(self, T, voxel_param, tsdf):
        """
            cast : ImageRays class' member function
                Collection of geometric parameters of rays in an image

            Parameters
            ----------
            T : ndarray of shape (4, 4)
                Transformation that brings camera to world coordinate
            voxel_param : an instance of voxel parameter VoxelParams
            tsdf : an instance of TSDF

            Returns
            -------
            point_pred : ndarray of shape (3, H, W)
                Point cloud from casting ray to tsdf
            valid : ndarray of shape (H, W)
                Mask to indicate which points are properly casted
        """
        r=np.tile(np.reshape(T[:3,-1], (1,3,1)), [255,1,1])+np.tensordot(self.lambda_step,(T[:3,:3]@self.rays_d),axes=0) #255, 3, H*W
        r=np.transpose(r,axes=(1,0,2))

        #Move to voxel space
        r_vox=(np.reshape(r,(3,-1))-np.reshape(voxel_param.voxel_origin, (3,1)))/voxel_param.vox_size
        r_vox=np.reshape(r_vox, (3,255,-1))

        #Swap x and y for map_coordinates function
        r_vox=np.vstack((np.reshape(r_vox[1,:],(1,255,-1)),np.reshape(r_vox[0,:],(1,255,-1)),np.reshape(r_vox[2,:], (1,255,-1))))


        fr=scipy.ndimage.map_coordinates(tsdf.value,r_vox)
        fr_1=fr[1:,:]
        fr=fr[:-1,:]
        r_1=r_vox[:,1:,:]
        r_vox=r_vox[:,:-1,:]

        s=np.argwhere(fr*fr_1<0)
        #Remove duplicates
        unique, indices=np.unique(s[:,1], return_index=True)
        s=s[indices]

        fr_1=fr_1[s[:,0],s[:,1]]
        fr=fr[s[:,0],s[:,1]]
        r_1=r_1[:,s[:,0],s[:,1]]
        r_vox=r_vox[:,s[:,0],s[:,1]]

        point_pred=np.zeros((3,r.shape[2]))
        point_pred[:,s[:,1]]=-fr/(fr_1-fr)*(r_1-r_vox)+r_vox
        #Swap x and y back after map_coordinates function
        point_pred=np.vstack((np.reshape(point_pred[1,:],(1,-1)),np.reshape(point_pred[0,:], (1,-1)), np.reshape(point_pred[2,:], (1,-1))))
        point_pred=point_pred*voxel_param.vox_size+np.reshape(voxel_param.voxel_origin, (3,1))
        valid=np.zeros((r.shape[2]))
        valid[s[:,1]]=1
        point_pred=np.reshape(point_pred, (3,480,640))
        valid=np.reshape(valid,(480,640))

        return point_pred, valid.astype('bool')

