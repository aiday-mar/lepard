from models.blocks import *
from models.backbone import KPFCN
from models.transformer import RepositioningTransformer
from models.matching import Matching
from models.procrustes import SoftProcrustesLayer

class Pipeline(nn.Module):

    def __init__(self, config):
        super(Pipeline, self).__init__()
        self.config = config
        self.backbone = KPFCN(config['kpfcn_config'])
        self.pe_type = config['coarse_transformer']['pe_type']
        self.positioning_type = config['coarse_transformer']['positioning_type']
        self.coarse_transformer = RepositioningTransformer(config['coarse_transformer'])
        self.coarse_matching = Matching(config['coarse_matching'])
        self.soft_procrustes = SoftProcrustesLayer(config['coarse_transformer']['procrustes'])

    def forward(self, data,  timers=None):

        print('\n')
        print('Inside of forward method of PipelineFCGF')
        self.timers = timers

        print('\n')
        print('Before FCGF backbone')
        if self.timers: self.timers.tic('fcgf backbone encode')
        coarse_feats = self.backbone(data, phase="coarse")
        if self.timers: self.timers.toc('fcgf backbone encode')

        print('\n')
        print('Before split_feats')
        if self.timers: self.timers.tic('coarse_preprocess')
        src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask = self.split_feats(coarse_feats, data)
        data.update({ 's_pcd': s_pcd, 't_pcd': t_pcd })
        if self.timers: self.timers.toc('coarse_preprocess')

        print('\n')
        print('Before RepositioningTransformer')
        if self.timers: self.timers.tic('coarse feature transformer')
        src_feats, tgt_feats, src_pe, tgt_pe = self.coarse_transformer(src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask, data, timers=timers, feature_extractor = self.feature_extractor)
        if self.timers: self.timers.toc('coarse feature transformer')

        print('\n')
        print('Before Matching')
        if self.timers: self.timers.tic('match feature coarse')
        conf_matrix_pred, coarse_match_pred = self.coarse_matching(src_feats, tgt_feats, src_pe, tgt_pe, src_mask, tgt_mask, data, pe_type = self.pe_type)
        data.update({'conf_matrix_pred': conf_matrix_pred, 'coarse_match_pred': coarse_match_pred })
        if self.timers: self.timers.toc('match feature coarse')

        print('\n')
        print('Before SoftProcrustesLayer')
        if self.timers: self.timers.tic('procrustes_layer')
        R, t, _, _, _, _ = self.soft_procrustes(conf_matrix_pred, s_pcd, t_pcd, src_mask, tgt_mask)
        data.update({"R_s2t_pred": R, "t_s2t_pred": t})
        if self.timers: self.timers.toc('procrustes_layer')

        return data

    def split_feats(self, geo_feats, data):
   
        print('Inside of split_feats')
        print('geo_feats.shape : ', geo_feats.shape)
        pcd = data['points'][self.config['kpfcn_config']['coarse_level']]
        print('pcd.shape : ', pcd.shape)

        src_mask = data['src_mask']
        tgt_mask = data['tgt_mask']
        print('src_mask.shape : ', src_mask.shape)
        print('tgt_mask.shape : ', tgt_mask.shape)

        src_ind_coarse_split = data['src_ind_coarse_split']
        tgt_ind_coarse_split = data['tgt_ind_coarse_split']
        print('src_ind_coarse_split.shape : ', src_ind_coarse_split.shape)
        print('tgt_ind_coarse_split.shape : ', tgt_ind_coarse_split.shape)

        src_ind_coarse = data['src_ind_coarse']
        tgt_ind_coarse = data['tgt_ind_coarse']
        print('src_ind_coarse.shape : ', src_ind_coarse.shape)
        print('tgt_ind_coarse.shape : ', tgt_ind_coarse.shape)

        b_size, src_pts_max = src_mask.shape
        tgt_pts_max = tgt_mask.shape[1]

        src_feats = torch.zeros([b_size * src_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
        tgt_feats = torch.zeros([b_size * tgt_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
        src_pcd = torch.zeros([b_size * src_pts_max, 3]).type_as(pcd)
        tgt_pcd = torch.zeros([b_size * tgt_pts_max, 3]).type_as(pcd)

        src_feats[src_ind_coarse_split] = geo_feats[src_ind_coarse]
        tgt_feats[tgt_ind_coarse_split] = geo_feats[tgt_ind_coarse]
        src_pcd[src_ind_coarse_split] = pcd[src_ind_coarse]
        tgt_pcd[tgt_ind_coarse_split] = pcd[tgt_ind_coarse]

        return src_feats.view( b_size , src_pts_max , -1), \
               tgt_feats.view( b_size , tgt_pts_max , -1), \
               src_pcd.view( b_size , src_pts_max , -1), \
               tgt_pcd.view( b_size , tgt_pts_max , -1), \
               src_mask, \
               tgt_mask