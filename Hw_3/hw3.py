import argparse
import logging
from datetime import datetime
import pickle
import numpy as np
import os
import sys
import torch
import torch.autograd
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import cv2


from dataset import *
from network import *


def ParseCmdLineArguments():
    parser = argparse.ArgumentParser(description='CSCI 5563 HW3')
    parser.add_argument('--dataset_pickle_file', type=str,
                        default='./tiny_scannet.pkl')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1.e-4)
    parser.add_argument('--save', type=str, default='gdrive/MyDrive/CSCI5563_Hw3/')

    return parser.parse_args()


def ComputeDepthError(depth_pred, depth_gt):
    #Generate the mask of the ground truth depth
    M=torch.where(depth_gt>0,1.0,0.0)

    #Get the difference between the ground truth and the prediction
    diff=depth_gt-depth_pred

    #Apply the mask and take absolute value so that we can sum over to get the L1 norm
    masked_diff=torch.abs(M*diff)

    #Calculate the cardinality, which is the number of valid depth points, which is also the sum of the mask
    cardinality=torch.sum(M)

    #Take the L1 norm and multiply by 1/cardinality of mask
    L_d = (1/cardinality.item())*torch.sum(masked_diff)

    return L_d


def ComputeNormalError(depth_pred, K, normal_gt, depth_gt):
    #Generate the mask of the ground truth depth
    M=torch.where(depth_gt>0,1.0,0.0)

    #Calculate the cardinality, which is the number of valid depth points, which is also the sum of the mask
    cardinality=torch.sum(M)

    ##Calculate the 3D point
    #Get the indices
    vv, uu = torch.meshgrid(torch.arange(depth_pred.shape[2]), torch.arange(depth_pred.shape[3]))
    coord=torch.vstack((uu.flatten(), vv.flatten()))
    coord=torch.vstack((coord, torch.ones(coord.shape[1])))

    #Move to 3D by multiplying by inverse camera intrinsics
    coord_3d=torch.linalg.inv(torch.from_numpy(K))@coord.double()

    #Reshape
    coord_3d=torch.reshape(coord_3d, (1,3,depth_pred.shape[2], depth_pred.shape[3]))

    #Make into batches
    coord_batch=torch.repeat_interleave(coord_3d, depth_pred.shape[0], dim=0)

    #Calculate X by doing elementwise multiplication of depth 
    X=depth_pred*coord_batch.cuda(non_blocking=True)

    #Calculate offsets
    v_offset = torch.arange(1,depth_pred.shape[2]+1)
    u_offset = torch.arange(1,depth_pred.shape[3]+1)

    #Reflection padding for the cases out of image bounds
    v_offset[-1]=v_offset[-3]
    u_offset[-1]=u_offset[-3]

    vv_offset, uu_offset = torch.meshgrid(v_offset, u_offset)

    #Calculate the numerator
    num=torch.cross((X[:,:,vv,uu_offset]-X[:,:,vv,uu]),(X[:,:,vv_offset,uu]-X[:,:,vv,uu]),dim=1)

    #Calculate the denominator
    den=torch.linalg.norm(num,dim=1, keepdim=True)
    #Prevent divide by zero
    den[den==0]=1e-15

    n_hat=num/den

    #Compute the absolute value of n_hat transpose times the ground truth, which is the same as multiplying elementwise and summing over dimension 1
    temp=n_hat*normal_gt
    err=torch.abs(torch.sum(temp, dim=1, keepdim=True))
    #Subtract err from 1
    pre_masked=torch.ones(err.shape).cuda(non_blocking=True)-err

    #Sum over dimensions and divide by cardinality of the mask
    L_c=(1/cardinality.item())*torch.sum(M*pre_masked)

    return L_c

if __name__ == '__main__':
    args = ParseCmdLineArguments()

    train_dataset = TinyScanNetDataset(usage='train',
                                       dataset_pickle_file=args.dataset_pickle_file)

    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=args.batch_size,
                                  num_workers=4)

    val_dataset = TinyScanNetDataset(usage='val',
                                     dataset_pickle_file=args.dataset_pickle_file)

    val_dataloader = DataLoader(val_dataset, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=2)


    #depth_net = SimpleDepthNet().cuda()
    depth_net = ExtendedFPNDepth().cuda()
    optimizer = torch.optim.Adam(depth_net.parameters(), lr=args.learning_rate)

    K = np.array([[288.935, 0, 159.938], [0, 290.129, 119.938], [0.0, 0.0, 1.0]])

    for epoch in range(100):

        ## Train
        L_err=[]
        L_d_err=[]
        L_c_err=[]
        depth_net.train()
        for train_idx, train_data in enumerate(train_dataloader):
            # Retrieve a batch of data
            image = train_data['image'].cuda(non_blocking=True)
            depth_gt = train_data['depth'].cuda(non_blocking=True)
            normal_gt = train_data['normal'].cuda(non_blocking=True)

            # Evaluate loss and backpropagate to update the weights
            depth_pred=depth_net(image)
            L_d=ComputeDepthError(depth_pred, depth_gt)
            L_d_err.append(L_d.item())
            L_c=ComputeNormalError(depth_pred, K, normal_gt, depth_gt)
            L_c_err.append(L_c.item())
            L=L_d+0.5*L_c
            L_err.append(L)
            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            print('epoch: %d, iter: %d, L_train: %2.2f (L_d_train: %2.2f, L_c_train: %2.2f)' % (epoch, train_idx, L, L_d.item(), L_c.item()))
        # Saving network's weights
        path = os.path.join(args.save, 'Models/trained_model-%05d.ckpt' % epoch)
        plot_path = os.path.join(args.save, 'Results/training_error-%05d.png' % epoch)
        torch.save(depth_net.state_dict(), path)
        plt.plot(L_err, label="L")
        plt.plot(L_c_err, label="L_c")
        plt.plot(L_d_err, label="L_d")
        plt.xlabel("Training iterations")
        plt.ylabel("Error")
        plt.title("Epoch " + str(epoch))
        plt.legend()
        plt.savefig(plot_path)
        plt.clf()

        depth_net.eval()
        with torch.no_grad():
            # Evaluate on validation data and visualize
            for val_idx, val_data in enumerate(val_dataloader):
                image_val=val_data['image'].cuda(non_blocking=True)
                depth_gt_val = val_data['depth'].cuda(non_blocking=True)
                normal_gt_val = val_data['normal'].cuda(non_blocking=True)
                depth_pred_val=depth_net(image_val)
                L_d_val=ComputeDepthError(depth_pred_val, depth_gt_val)
                L_c_val=ComputeNormalError(depth_pred_val, K, normal_gt_val, depth_gt_val)
                L_val=L_d_val+L_c_val
                print('epoch: %d, iter: %d, L_val: %2.2f (L_d_val: %2.2f, L_c_val: %2.2f)' % (epoch, val_idx, L_val, L_d_val.item(), L_c_val.item()))

                #Save images to visualize
                gt_path = os.path.join(args.save, 'Results/val_gt_epoch_%05d_iter_%05d.png' % (epoch,val_idx))
                color_path = os.path.join(args.save, 'Results/val_color_epoch_%05d_iter_%05d.png' % (epoch,val_idx))
                pred_path = os.path.join(args.save, 'Results/val_pred_epoch_%05d_iter_%05d.png' % (epoch,val_idx))

                #On our first epoch let's save the ground truth depth map and images for easy access
                if(epoch==0):
                    #Save ground truth for easier access
                    gt_to_save=depth_gt_val[0].cpu().numpy()
                    gt_to_save=np.reshape(gt_to_save,(240,320))
                    plt.imsave(gt_path, gt_to_save, cmap='magma')

                    #Save color image for easier access
                    color_to_save=image_val[0].cpu()
                    color_to_save=color_to_save.permute(1,2,0)
                    color_to_save=color_to_save.numpy()
                    plt.imsave(color_path, color_to_save)

                #Save prediction depth map
                pred_to_save = depth_pred_val[0].cpu().numpy()
                pred_to_save = np.reshape(pred_to_save, (240, 320))
                plt.imsave(pred_path, pred_to_save, cmap='magma')


