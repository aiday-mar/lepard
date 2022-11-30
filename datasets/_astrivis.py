import os, sys
[sys.path.append(i) for i in ['.', '..']]
import numpy as np
import open3d as o3d
import h5py
from torch.utils.data import Dataset
HMN_intrin = np.array( [443, 256, 443, 250 ])
cam_intrin = np.array( [443, 256, 443, 250 ])

class _Astrivis(Dataset):

    def __init__(self, config, split):
        super(_Astrivis, self).__init__()

        assert split in ['train','val','test']
        print('split : ', split)
        self.split = split
        self.matches = {}
        self.number_matches = 0
        self.n_files_per_folder = 0
        self.config = config
        self.folders = np.linspace(0, 225, num=160)
        n_files_per_folder_found = False
        
        path = ''
        if self.split == 'train':
            path = '/home/aiday.kyzy/dataset/Synthetic/PartialDeformedData/TrainingData/'
        elif self.split == 'val':
            path = '/home/aiday.kyzy/dataset/Synthetic/PartialDeformedData/ValidationData/'
        elif self.split == 'test':
            path = '/home/aiday.kyzy/dataset/Synthetic/PartialDeformedData/TestingData/'
        
        self.path = path
        for folder in os.listdir(self.path):
            self.matches[folder] = []
            for filename in os.listdir(self.path + folder + '/matches'):
                self.matches[folder].append(filename)
                self.number_matches += 1
                if not n_files_per_folder_found:
                    self.n_files_per_folder += 1
            
            n_files_per_folder_found = True
    
        print('number of files per folder: ', self.n_files_per_folder)
        print('number of matches : ', self.number_matches)
        print('number of folders : ', len(self.matches))
        
    def __len__(self):
        return self.number_matches


    def __getitem__(self, index):

        folder_number = index // self.n_files_per_folder
        idx_inside_folder = index % self.n_files_per_folder
        print(self.folders[folder_number])
        folder_string = 'model' + str(round(self.folders[folder_number])).zfill(3)
        files_array = self.matches[folder_string]
        filename = files_array[idx_inside_folder]
                
        file_pointers = filename[:-4]
        file_pointers = file_pointers.split('_')
        
        src_pcd_file = file_pointers[0] + '_' + file_pointers[2] + '.ply'
        tgt_pcd_file = file_pointers[1] + '_' + file_pointers[3] + '.ply'
        
        src_pcd = o3d.io.read_point_cloud(self.path + folder_string + '/transformed/' + src_pcd_file)
        src_pcd = np.array(src_pcd.points)
        tgt_pcd = o3d.io.read_point_cloud(self.path + folder_string + '/transformed/' + tgt_pcd_file)
        tgt_pcd = np.array(tgt_pcd.points)
        
        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        
        matches = np.load(self.path + folder_string + '/matches/' + filename)
        correspondences = np.array(matches['matches'])
        indices_src = correspondences[:, 0]
        indices_tgt = correspondences[:, 1]
        
        # Added in order to get the s2t flow on the centered tgt and source
        src_pcd_centered = src_pcd - np.mean(src_pcd, axis=0)
        tgt_pcd_centered = tgt_pcd - np.mean(tgt_pcd, axis=0)
        
        src_flow = np.array([src_pcd_centered[i] for i in indices_src])
        tgt_flow = np.array([tgt_pcd_centered[i] for i in indices_tgt])
                
        s2t_flow = tgt_flow - src_flow
        
        src_pcd_trans = file_pointers[0] + '_' + file_pointers[2] + '_se4.h5'
        tgt_pcd_trans = file_pointers[1] + '_' + file_pointers[3] + '_se4.h5'
        
        src_trans_file=h5py.File(self.path + folder_string + '/transformed/' + src_pcd_trans, "r")
        src_pcd_transform = np.array(src_trans_file['transformation'])
        
        tgt_trans_file=h5py.File(self.path + folder_string + '/transformed/' + tgt_pcd_trans, "r")
        tgt_pcd_transform_inverse = np.linalg.inv(np.array(tgt_trans_file['transformation']))
        
        rot = np.matmul(tgt_pcd_transform_inverse[:3, :3], src_pcd_transform[:3, :3])
        trans = tgt_pcd_transform_inverse[:3, :3]@src_pcd_transform[:3, 3] + tgt_pcd_transform_inverse[:3, 3]
        trans = np.expand_dims(trans, axis=0)
        trans = trans.transpose()
        
        metric_index = None
        
        return src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, rot, trans, s2t_flow, metric_index
