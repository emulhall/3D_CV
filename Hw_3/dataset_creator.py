import argparse
import os
import pickle
import fnmatch
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

def ParseCmdLineArguments():
    parser = argparse.ArgumentParser(description='CSCI5563 HW3')
    parser.add_argument('--data_path', type=str, default='gdrive/Shareddrives/CSCI5563_data/tiny_scannet_data/',
                        help='The path to the tiny scannet data.')
    parser.add_argument('--output_path', type=str, default='gdrive/MyDrive/CSCI5563_HW3/',
                        help='The path to save the tiny_scannet.pkl file.')
    return parser.parse_args()



def create_split(ROOT_DIR, split_name):
    split_folder = os.path.join(ROOT_DIR, split_name)

    final_split = [[], [], [], []]
    num_frames = 0
    color_filelist = fnmatch.filter(os.listdir(split_folder), 'color_*')
    for img_idx in range(len(color_filelist)):
        color_path = os.path.join(split_folder, color_filelist[img_idx])
        normal_path = color_path.replace('color', 'normal')
        depth_path = color_path.replace('color', 'depth')
        final_split[0].append(color_path)
        final_split[1].append(depth_path)
        final_split[2].append(normal_path)   
        num_frames += 1

    print('Number of frames in the %s split: %d' % (split_name, num_frames))

    return final_split


def main():
    args = ParseCmdLineArguments()
    final_dict = {'train': create_split(args.data_path, 'train')}
    with open('TUM.pkl', 'wb') as f:
        pickle.dump(final_dict, f)

if __name__ == "__main__":
    main()
