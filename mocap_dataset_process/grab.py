
import pandas as pd
import os
import pickle
import numpy as np
from utils.rotation_conversions import *
from smplx import SMPLX
from utils.face_z_align_util import joint_idx, face_z_transform
import re
from tqdm import tqdm
import argparse



def compute_canonical_transform(global_orient):
    rotation_matrix = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=global_orient.dtype)
    global_orient_matrix = axis_angle_to_matrix(global_orient)
    global_orient_matrix = torch.matmul(rotation_matrix, global_orient_matrix)
    global_orient = matrix_to_axis_angle(global_orient_matrix)
    return global_orient

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            if "zip" in f: # jaron-modify: 把除了.npy结尾的文件过滤掉，例如.zip
                continue
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path



def transform_motions(data):
    ex_fps = 30
    fps = 120
    down_sample = int(fps / ex_fps)

    frame_number = data['body']['params']['transl'].shape[0]
    


    fId = 0 # frame id of the mocap sequence
    pose_seq = []

    

    for fId in range(0, frame_number, down_sample):
        pose_root = data['body']['params']['global_orient'][fId:fId+1]
        pose_root = compute_canonical_transform(torch.from_numpy(pose_root)).detach().cpu().numpy()
        pose_body = data['body']['params']['body_pose'][fId:fId+1]


        pose_left_hand = data['lhand']['params']['fullpose'][fId:fId+1]
        pose_right_hand = data['rhand']['params']['fullpose'][fId:fId+1]


        pose_jaw = data['body']['params']['jaw_pose'][fId:fId+1]

        ####grab expression is 10-dim, so we use zeros

        pose_expression =  np.zeros((1, 50))
        pose_face_shape = np.zeros((1, 100))

        pose_trans = data['body']['params']['transl'][fId:fId+1]

        pose_body_shape = np.zeros((1, 10))
        pose = np.concatenate((pose_root, pose_body, pose_left_hand, pose_right_hand, pose_jaw, pose_expression, pose_face_shape, pose_trans, pose_body_shape), axis=1)
        pose_seq.append(pose)

    pose_seq = np.concatenate(pose_seq, axis=0)

    return pose_seq

def create_parent_dir(target_t2m_joints_case_path):
    target_t2m_joints_parent_directory = os.path.dirname(target_t2m_joints_case_path)

    if not os.path.exists(target_t2m_joints_parent_directory):
        os.makedirs(target_t2m_joints_parent_directory)


def process_text(text):
    result = re.sub(r'_\d+', '', text)
    result = result.replace('_', ' ')
    result = re.sub(r'\s+', ' ', result)
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./GRAB", help="the directory that stores the downloaded data.")

if __name__ == '__main__':
    opt = parser.parse_args()
    data_dir = opt.data_dir

    for case_path in tqdm(findAllFile(data_dir)):
        data = np.load(case_path, allow_pickle=True)
        data = {k: data[k].item() for k in data.files}

        data = transform_motions(data)

        text = os.path.basename(case_path).replace('.npz', '')
        text = process_text(text)

        output_motion_path = case_path.replace('GRAB', 'GRAB_motion').replace('.npz', '.npy')
        output_text_path = case_path.replace('GRAB', 'GRAB_text').replace('.npz', '.txt')

        create_parent_dir(output_motion_path)
        create_parent_dir(output_text_path)
        np.save(output_motion_path, data)
        with open(output_text_path, 'w') as f:
            f.write(text)

        


# python grab.py --data_dir "D:\\jarondu\\Datasets\\motion_X_two\\GRAB"
