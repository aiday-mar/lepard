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
        if self.config.data_type == 'full_deformed':
            self.folder_type = 'FullDeformedData'
        elif self.config.data_type == 'partial_deformed':
            self.folder_type = 'PartialDeformedData'

        if split == 'train':
            if self.folder_type == 'FullDeformedData':
                self.folders = [
                                '000', '001', '003', '004', '006', '007', '009', 
                                '010', '011', '013', '014', '016', '017', '018', 
                                '020', '021', '023', '024', '026', '027', '028', 
                                '030', '031', '033', '034', '036', '037', '038',
                                '040', '041', '043', '044', '045', '047', '048', 
                                '050', '051', '053', '054', '055', '057', '058',
                                '060', '061', '063', '064', '065', '067', '068', 
                                '070', '071', '072', '074', '075', '077', '078',
                                '080', '081', '082', '084', '086', '087', '088',
                                '090', '091', '092', '094', '095', '097', '098',
                                '099', '102', '104', '105', '107', '108',
                                '109', '112', '114', '115', '117', '118',
                                '119', '121', '122', '124', '125', '127', '128',
                                '129', '131', '132', '134', '135', '136', '138', 
                                '139', '141', '142', '144', '145', '146', '148',
                                '149', '151', '152', '154', '155', '158',
                                '159', '161', '162', '165', '166', '168',
                                '169', '171', '172', '173', '175', '176', '178', 
                                '179', '181', '182', '183', '185', '186', '188',
                                '189', '190', '192', '193', '195', '196', '198', 
                                '199', '200', '202', '203', '205', '206', '208', 
                                '209', '210', '212', '213', '216', '217', 
                                '219', '220', '222', '223' # full deformed data does not have these: '224', '225', '215', '156', '163', '111', '101'
                                ]
            elif self.folder_type == 'PartialDeformedData':
                self.folders = [
                                '000', '001', '003', '004', '006', '007', '009', 
                                '010', '011', '013', '014', '016', '017', '018', 
                                '020', '021', '023', '024', '026', '027', '028', 
                                '030', '031', '033', '034', '036', '037', '038',
                                '040', '041', '043', '044', '045', '047', '048', 
                                '050', '051', '053', '054', '055', '057', '058',
                                '060', '061', '063', '064', '065', '067', '068', 
                                '070', '071', '072', '074', '075', '077', '078',
                                '080', '081', '082', '084', '086', '087', '088',
                                '090', '091', '092', '094', '095', '097', '098',
                                '099', '101', '102', '104', '105', '107', '108',
                                '109', '111', '112', '114', '115', '117', '118',
                                '119', '121', '122', '124', '125', '127', '128',
                                '129', '131', '132', '134', '135', '136', '138', 
                                '139', '141', '142', '144', '145', '146', '148',
                                '149', '151', '152', '154', '155', '156', '158',
                                '159', '161', '162', '163', '165', '166', '168',
                                '169', '171', '172', '173', '175', '176', '178', 
                                '179', '181', '182', '183', '185', '186', '188',
                                '189', '190', '192', '193', '195', '196', '198', 
                                '199', '200', '202', '203', '205', '206', '208', 
                                '209', '210', '212', '213', '215', '216', '217', 
                                '219', '220', '222', '223', '224', '225'
                                ]
        elif split == 'val':
            self.folders = [
                '005', '012', '019', '025', '032', '039', '046', '052', '059', '062', '069', '076', '083', '089', '096',
                '103', '110', '116', '123', '130', '137', '143', '150', '157', '164', '170', '177', '184', '191', '197',
                '204', '211', '218'
            ]
        elif split == 'test':
            self.folders = [
                '002', '008', '015', '022', '029', '035', '042', '049', '056', '066', '073', '079', '085', '093', '100', 
                '106', '113', '120', '126', '133', '140', '147', '153', '160', '167', '174', '180', '187', '194', '201',
                '207', '214', '221'
            ]
            
        n_files_per_folder_found = False
        path = ''
        if self.split == 'train':
            path = '/home/aiday.kyzy/dataset/Synthetic/' + self.folder_type + '/TrainingData/'
        elif self.split == 'val':
            path = '/home/aiday.kyzy/dataset/Synthetic/' + self.folder_type + '/ValidationData/'
        elif self.split == 'test':
            path = '/home/aiday.kyzy/dataset/Synthetic/' + self.folder_type + '/TestingData/'
        
        self.path = path
        for folder in os.listdir(self.path):
            self.matches[folder] = []
            for filename in os.listdir(self.path + folder + '/matches'):
                if filename != 'overlap.txt':
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
        folder_string = 'model' + str(self.folders[folder_number]).zfill(3)
        files_array = self.matches[folder_string]
        filename = files_array[idx_inside_folder]
                
        file_pointers = filename[:-4]
        file_pointers = file_pointers.split('_')
        
        if self.folder_type == 'PartialDeformedData':
            src_pcd_file = file_pointers[0] + '_' + file_pointers[2] + '.ply'
            tgt_pcd_file = file_pointers[1] + '_' + file_pointers[3] + '.ply'
            src_pcd_trans = file_pointers[0] + '_' + file_pointers[2] + '_se4.h5'
            tgt_pcd_trans = file_pointers[1] + '_' + file_pointers[3] + '_se4.h5'
        elif self.folder_type == 'FullDeformedData':
            src_pcd_file = file_pointers[0] + '.ply'
            tgt_pcd_file = file_pointers[1] + '.ply'
            src_pcd_trans = file_pointers[0] + '_se4.h5'
            tgt_pcd_trans = file_pointers[1] + '_se4.h5'
        else:
            raise Exception('Specify a valid data type')
        
        src_pcd = o3d.io.read_point_cloud(self.path + folder_string + '/transformed/' + src_pcd_file)
        src_pcd = np.array(src_pcd.points)
        tgt_pcd = o3d.io.read_point_cloud(self.path + folder_string + '/transformed/' + tgt_pcd_file)
        tgt_pcd = np.array(tgt_pcd.points)
        
        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        
        matches = np.load(self.path + folder_string + '/matches/' + filename)
        correspondences = np.array(matches['matches'])
        final_correspondences = np.empty((0,2), int)
        set_src_indices = set()
        for correspondence in correspondences:
            if correspondence[0] not in set_src_indices:
                final_correspondences = np.append(final_correspondences, np.array(np.expand_dims(correspondence, axis=0)), axis=0)
            set_src_indices.add(correspondence[0])
        correspondence = final_correspondences
        indices_src = correspondences[:, 0]
        indices_tgt = correspondences[:, 1]
        
        src_pcd_centered = src_pcd - np.mean(src_pcd, axis=0)
        tgt_pcd_centered = tgt_pcd - np.mean(tgt_pcd, axis=0)
        
        s2t_flow = np.zeros(src_pcd.shape)
        for i in range(len(indices_src)):
            src_idx = indices_src[i]
            tgt_idx = indices_tgt[i]
            s2t_flow[src_idx] = src_pcd_centered[src_idx] - tgt_pcd_centered[tgt_idx]
                        
        src_trans_file=h5py.File(self.path + folder_string + '/transformed/' + src_pcd_trans, "r")
        src_pcd_transform = np.array(src_trans_file['transformation'])
        
        tgt_trans_file=h5py.File(self.path + folder_string + '/transformed/' + tgt_pcd_trans, "r")
        tgt_pcd_transform_inverse = np.linalg.inv(np.array(tgt_trans_file['transformation']))
        
        rot = np.matmul(tgt_pcd_transform_inverse[:3, :3], src_pcd_transform[:3, :3])
        trans = tgt_pcd_transform_inverse[:3, :3]@src_pcd_transform[:3, 3] + tgt_pcd_transform_inverse[:3, 3]
        trans = np.expand_dims(trans, axis=0)
        trans = trans.transpose()
        
        metric_index = None
        
        return src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, rot, trans, s2t_flow, metric_index, None, None
