# import logging
from PIL import Image
import os
import pickle
from collections import OrderedDict
from torchvision import transforms
import numpy as np
import pandas as pd
import torch
import time
import torch.nn.functional as F
import tqdm
import utils.common as common
from utils.utils import modified_kNN_score_calc
import utils.metrics as metrics
from utils.backbone import Backbone
from utils.embedding import _embed, _feature_extraction, PatchMaker, alternative_pooling
import os
from sklearn.metrics import roc_auc_score
from path_definitions import ROOT_DIR, RES_DIR, PLOT_DIR, MVTEC_DIR, EMBEDDING_DIR
from utils.datasets import MVTecDataset
from torch.utils.data import DataLoader
from time import perf_counter
from torchinfo import summary
import json

def init_weight(m):
    '''
    Used to initialize the weights of the discriminator and projection networks.
    '''
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Discriminator(torch.nn.Module):
    '''
    Discriminator network.
    '''
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden # which is alsoe by default None ???
        # print('Hidden is: ', _hidden)
        self.body = torch.nn.Sequential()
        for i in range(n_layers-1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d'%(i+1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
            # print('Hidden is: ', _hidden)
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self,x):
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):
    '''
    Feature Adaptor
    '''    
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()
        
        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes #if i == 0 else _out
            _out = out_planes
            print('Projection: ', _in, _out)
            self.layers.add_module(f"{i}fc", 
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                if layer_type > 0:
                    self.layers.add_module(f"{i}bn", 
                                           torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)
    
    def forward(self, x):
        
        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x



class SimpleNet(torch.nn.Module):
    def __init__(self, device):
        """anomaly detection class."""
        super(SimpleNet, self).__init__()
        self.device = device
        self.input_shape = (3,256,256)
        self.only_img_lvl = True
        self.measure_inference = True
        if self.measure_inference:
            self.number_of_reps = 2 # number of reps during measurement. Beacause we can assume a consistent estimator, results get more accurate with more reps
            self.warm_up_reps = 1 # before the actual measurement is done, we execute the process a couple of times without measurement to ensure that there is no influence of initialization and that the circumstances (e.g. thermal state of hardware) are representive.
        # Backbone
        self.backbone_id = 'WRN50' # analogous to model_id
        self.layers_to_extract_from = [2,3] # analogous to layers_needed
        self.quantize_qint8 = False
        self.exclude_relu = False
        # if self.quantize_qint8: because argumens have to be initialized before they can be tweaked from the outside
        # self.device = 'cpu'
        self.calibration_dataset = 'random'
        self.cpu_arch = 'x86'
        self.num_images_calib = 100
        # Embedding
        self.pooling_embedding = False
        self.pretrain_embed_dimensions = 1536#256 + 128 # for RN18
        self.target_embed_dimensions = 1536#128 + 256 # for RN18
        self.patch_size = 3
        self.patch_stride = 1
        self.embedding_size = None # TODO --> What does that do?
        # Projection
        self.pre_proj = 1 # TODO
        self.proj_layer_type = 0 # TODO
        # Discriminator
        self.dsc_layers = 2
        self.dsc_hidden = int(self.target_embed_dimensions*0.75)#1024
        self.dsc_margin = 0.0 # TODO
        # Noise
        self.noise_std = 0.015
        self.auto_noise = [0, None] # TODO
        self.noise_type = 'GAU'
        self.mix_noise = 1 # TODO --> probably just the number of classes of noise. Usually one
        # Training
        self.meta_epochs = 40
        self.aed_meta_epochs = 0 # TODO
        self.gan_epochs = 4 # TODO
        self.batch_size = 8
        self.dsc_lr = 0.0002
        self.proj_lr = 1e-4
        self.lr_scheduler = True
        self.num_workers = 12
        # Scoring 
        self.adapted_score_calc = True # TODO
        self.top_k = 0
        self.batch_size_test = 1
        # Directory
        self.output_dir = r'/home/jo/tmp 16-11/MA_complete/results/simplenet'
        # self.output_dir = r'/MA_complete/results/simplenet'
        # /home/jo/tmp 16-11/MA_complete/train_main_simplenet.py
        # self.run_id = 'none' #not used here
        self.category = 'pill'
        self.dataset_path = MVTEC_DIR
        self.time_stamp = f'{int(time.time())}'
        self.group_id = 'not_specified'
        # Data transforms
        self.load_size = 288
        self.input_size = 256
        self.category_wise_statistics = False
        if self.category_wise_statistics:
            filename = 'statistics.json'
            with open(filename, "rb") as file:
                loaded_dict = pickle.load(file)
            statistics_of_categories = loaded_dict[self.category]
            means = statistics_of_categories['means']
            stds = statistics_of_categories['stds']
        else:
            means = [0.485, 0.456, 0.406] # Imagenet
            stds = [0.229, 0.224, 0.225]
        self.data_transforms = transforms.Compose([
                transforms.Resize((self.load_size, self.load_size), Image.ANTIALIAS),
                transforms.ToTensor(),
                transforms.CenterCrop(self.input_size),
                transforms.Normalize(mean=means,
                                    std=stds)]) # from imagenet  # for each category calculate mean and std TODO

        self.gt_transforms = transforms.Compose([
                        transforms.Resize((self.load_size, self.load_size)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(self.input_size)])
        self.inv_normalize = transforms.Normalize(mean=list(np.divide(np.multiply(-1,means), stds)), std=list(np.divide(1, stds))) 
        
        # auxilary variables
        self.i_iter = 0
        

    def set_model_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.model_dir = os.path.join(self.output_dir, self.group_id, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        self.summary_dir = os.path.join(self.output_dir, self.group_id, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)
        self.latences_dir = os.path.join(self.output_dir, self.group_id, 'latences')
        os.makedirs(self.latences_dir, exist_ok=True)
        self.train_progress_dir = os.path.join(self.output_dir, self.group_id, 'train_progress')
        os.makedirs(self.train_progress_dir, exist_ok=True)

    def test(self):#, training_data, test_data):
        '''
        main function
        '''
        self.set_model_dir()

        # self.acc_filename = f'acc_{self.group_id}_{self.time_stamp}.csv'acc_filename
        state_dict = {}
        # training_data = self.train_dataloader()
        test_data = self.test_dataloader()
        ### test
        file_path = os.path.join(self.model_dir, f'discriminator_{self.category}.pth')
        if os.path.exists(file_path): # model is already trained - we therefore assume training has already be done
            state_dict = torch.load(file_path, map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                self.discriminator.to(self.device)
                
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
                    self.pre_projection.to(self.device)
            else:
                print('No discriminator in state_dict!')
                self.load_state_dict(state_dict, strict=False)

            # self.predict(training_data, "train_")
            if not self.measure_inference:
                scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            else:
                scores, segmentations, features, labels_gt, masks_gt, run_times = self.predict(test_data)

            auroc, full_pixel_auroc, anomaly_pixel_auroc = self._evaluate(scores, segmentations, features, labels_gt, masks_gt)
                        # save as csv using pandas dataframe
            #, index=[batch_idx])
            if self.measure_inference:
                pd_run_times = pd.DataFrame(run_times)
                file_path = os.path.join(self.latences_dir, self.latences_filename)
                if os.path.exists(file_path):
                    try:
                        pd_run_times_ = pd.read_csv(file_path, index_col=0)
                        pd_run_times = pd.concat([pd_run_times_, pd_run_times], axis=0)
                        pd_run_times.to_csv(file_path)
                    except:
                        print('Error while saving latences. Save under alternative file name.')
                        file_path_alt = os.path.join(self.latences_dir, f'latences_{self.category}_{self.time_stamp}.csv')
                        pd_run_times.to_csv(file_path_alt)
                else:
                    pd_run_times.to_csv(file_path)

                pd_run_times_ = pd.read_csv(file_path, index_col=0)
                pd_results = pd.DataFrame({'img_auc': [auroc]*pd_run_times_.shape[0], 'pixel_auc': [full_pixel_auroc]*pd_run_times_.shape[0]})
                try:
                    pd_run_times = pd.concat([pd_run_times_, pd_results], axis=1)
                    pd_run_times.to_csv(file_path)
                except:
                    print('Error while saving summary. Save under alternative file name.')
                    file_path_alt = os.path.join(self.latences_dir, f'summary_{self.group_id}_{self.category}_{self.time_stamp}.csv')
                    pd_run_times.to_csv(file_path_alt)
                try:
                    device = next(self.backbone.parameters()).device
                    summary_of_backbone = summary(self.backbone, (1, 3, self.load_size, self.load_size), verbose = 0, device=device)
                    estimated_total_size = (summary_of_backbone.total_input + summary_of_backbone.total_output_bytes + summary_of_backbone.total_param_bytes) / 1e6 # in MB
                    number_of_mult_adds = summary_of_backbone.total_mult_adds / 1e6 # in M
                except:
                    estimated_total_size = 0.0
                    number_of_mult_adds = 0.0

                opt_dict = {
                    'device': self.device,
                    'input_shape': self.input_shape,
                    'backbone_id': self.backbone_id,
                    'layers_to_extract_from': self.layers_to_extract_from,
                    'quantize_qint8': self.quantize_qint8,
                    'calibration_dataset': self.calibration_dataset,
                    'num_images_calib': self.num_images_calib,
                    'meta_epochs': self.meta_epochs,
                    'gan_epochs': self.gan_epochs,
                    'dsc_lr': self.dsc_lr,
                    'proj_lr': self.proj_lr,
                    'dsc_layers': self.dsc_layers,
                    'dsc_hidden': self.dsc_hidden,
                    'dsc_margin': self.dsc_margin,
                    'noise_std': self.noise_std,
                    'mix_noise': self.mix_noise,
                    'pre_proj': self.pre_proj,
                    'proj_layer_type': self.proj_layer_type,
                    'top_k': self.top_k,
                    'batch_size': self.batch_size,
                    'batch_size_test': self.batch_size_test,
                    'pretrain_embed_dimensions': self.pretrain_embed_dimensions,
                    'target_embed_dimensions': self.target_embed_dimensions,
                    'patch_size': self.patch_size,
                    'patch_stride': self.patch_stride,
                    'embedding_size': self.embedding_size,
                    'adapted_score_calc': self.adapted_score_calc,
                    'category_wise_statistics': self.category_wise_statistics,
                    'load_size': self.load_size,
                    'input_size': self.input_size,
                    'category': self.category,
                    'dataset_path': self.dataset_path,
                    'time_stamp': self.time_stamp,
                    'group_id': self.group_id,
                    'cpu_arch': self.cpu_arch,
                    'num_images_calib': self.num_images_calib,
                    'calibration_dataset': self.calibration_dataset,
                    'quantize_qint8': self.quantize_qint8,
                    'only_img_lvl': self.only_img_lvl,
                    'measure_inference': self.measure_inference,
                    'number_of_reps': self.number_of_reps,
                    'warm_up_reps': self.warm_up_reps,
                    'model_dir': self.model_dir,
                    'backbone_storage_[MB]': estimated_total_size,
                    'backbone_mult_adds_[M]': number_of_mult_adds,
                    'feature_extraction_[ms]': pd_run_times['#1 feature extraction cpu'].mean() if self.measure_inference else 0.0,
                    'embedding_of_features_[ms]': pd_run_times['#3 embedding of features cpu'].mean() if self.measure_inference else 0.0,
                    'calc_distances_[ms]': pd_run_times['#5 score patches cpu'].mean() if self.measure_inference else 0.0,
                    'calc_scores_[ms]': pd_run_times['#7 img lvl score cpu'].mean() if self.measure_inference else 0.0,
                    'total_time_[ms]': pd_run_times['#11 whole process cpu'].mean() if self.measure_inference else 0.0,
                    'img_auc_[%]': auroc                
                    }
                file_path = os.path.join(self.summary_dir, f'summary_{self.group_id}.csv')
                if os.path.exists(file_path):
                    pd_sum = pd.read_csv(file_path, index_col=0)
                    pd_sum_current = pd.Series(opt_dict).to_frame(self.category)#, index='category')
                    pd_sum = pd.concat([pd_sum, pd_sum_current], axis=1)
                else:
                    # pd_sum = pd.DataFrame({'category': self.category,'img_acc': img_auc, 'adapted_score_calc': str(self.adapted_score_calc), 'pooling_strategy': str(self.pooling_strategy)}, index='category')
                    pd_sum = pd.Series(opt_dict).to_frame(self.category)
                pd_sum.to_csv(file_path)

            return auroc, full_pixel_auroc, anomaly_pixel_auroc

        else:
            print('No checkpoint found. Starting training.')

    def train(self):
        self.set_model_dir()
        
        self.y_loss_fake = []
        self.x_loss_fake = []
        self.y_loss_true = []
        self.x_loss_true = []
        self.y_loss_total = []
        self.x_loss_total = []
        
        self.y_auc = []
        self.x_auc = []
        self.y_pseudo_auc = []
        self.x_pseudo_auc = []
        
        # reset iteration counter
        self.i_iter = 0
        
        # initialize the network
        self.forward_modules = torch.nn.ModuleDict({})
        self.backbone = Backbone(model_id=self.backbone_id, layers_needed=self.layers_to_extract_from, layer_cut=True, exclude_relu=self.exclude_relu, quantize_qint8_prepared=self.quantize_qint8).to(self.device)
        feature_dimension = self.backbone.feature_dim
        if self.quantize_qint8:
            from utils.quantize import quantize_model_into_qint8
            self.backbone = quantize_model_into_qint8(model=self.backbone, layers_needed=self.layers_to_extract_from, calibrate=self.calibration_dataset, exclude_relu=self.exclude_relu, num_images=self.num_images_calib).to('cpu')#, dataset_path=self.dataset_path)
        self.forward_modules['backbone'] = self.backbone
        self.patch_maker = PatchMaker(patchsize=self.patch_size,top_k=self.top_k, stride=self.patch_stride)
        preprocessing = common.Preprocessing(input_dims = feature_dimension, output_dim = self.pretrain_embed_dimensions).to(self.device)
        self.forward_modules['preprocessing'] = preprocessing
        
        preadapt_aggregator = common.Aggregator(target_dim = self.target_embed_dimensions).to(self.device)
        self.forward_modules['preadapt_aggregator'] = preadapt_aggregator
        
        self.anomaly_segmentor = common.RescaleSegmentor(device = self.device, target_size = self.input_shape[1])#.to(self.device)
        
        self.embedding_size = self.embedding_size if self.embedding_size is not None else self.target_embed_dimensions
        
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimensions, self.pretrain_embed_dimensions, self.pre_proj, self.proj_layer_type)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), self.proj_lr)
        
        self.discriminator = Discriminator(self.pretrain_embed_dimensions, self.dsc_layers, self.dsc_hidden).to(self.device)
        self.dsc_opt = torch.optim.AdamW(self.discriminator.parameters(), self.dsc_lr, weight_decay=1e-5)
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(self.dsc_opt, self.meta_epochs, eta_min=self.dsc_lr*.4)

        
        self.log_path = os.path.join(os.path.dirname(__file__), "results","simplenet", f"{self.group_id}", "csv")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.latences_filename = f'latences_{self.category}.csv'
        state_dict = {}
        training_data = self.train_dataloader()
        test_data = self.test_dataloader()
        
        def update_state_dict():
            state_dict["discriminator"] = OrderedDict({
                k:v.detach().cpu() 
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k:v.detach().cpu() 
                    for k, v in self.pre_projection.state_dict().items()})
        # actual training loop
        best_record = None
        
        if not self.only_img_lvl:
            if not self.measure_inference:
                scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            else:
                scores, segmentations, features, labels_gt, masks_gt, run_times = self.predict(test_data)
        else:
            if not self.measure_inference:
                scores,_,_,labels_gt,_ = self.predict(test_data)
            else:
                scores,_,_,labels_gt,_, run_times = self.predict(test_data)
            segmentations, features, masks_gt = None, None, None
        auroc, full_pixel_auroc, pro = self._evaluate(scores, segmentations, features, labels_gt, masks_gt)
        
        self.y_auc.append(auroc)
        self.x_auc.append(self.i_iter)

        if best_record is None:
            best_record = [auroc, full_pixel_auroc, pro]
            update_state_dict()
        elif auroc > best_record[0]:
            best_record = [auroc, full_pixel_auroc, pro]
            update_state_dict()
        
        for i_mepoch in range(self.meta_epochs):

            self._train_discriminator(training_data)
            if not self.only_img_lvl:
                if not self.measure_inference:
                    scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
                else:
                    scores, segmentations, features, labels_gt, masks_gt, t_1_cpu, t_2_cpu, t_3_cpu = self.predict(test_data)
                    print('Time for feature extraction: ', t_1_cpu)
                    print('Time for embedding: ', t_2_cpu)
                    print('Time for discriminator: ', t_3_cpu)
            else:
                if not self.measure_inference:
                    scores,_,_,labels_gt,_ = self.predict(test_data)
                else:
                    scores,_,_,labels_gt,_, t_1_cpu, t_2_cpu, t_3_cpu = self.predict(test_data)
                    print('Time for feature extraction: ', t_1_cpu)
                    print('Time for embedding: ', t_2_cpu)
                    print('Time for discriminator: ', t_3_cpu)
                segmentations, features, masks_gt = None, None, None
            auroc, full_pixel_auroc, pro = self._evaluate(scores, segmentations, features, labels_gt, masks_gt)
            
            self.y_auc.append(auroc)
            self.x_auc.append(self.i_iter)

            if best_record is None:
                best_record = [auroc, full_pixel_auroc, pro]
                update_state_dict()
            elif auroc > best_record[0]:
                best_record = [auroc, full_pixel_auroc, pro]
                update_state_dict()
                # elif auroc == best_record[0] and full_pixel_auroc > best_record[1]:
                #     best_record[1] = full_pixel_auroc
                #     best_record[2] = pro 
                #     update_state_dict(state_dict)

            print(f"----- {i_mepoch} I-AUROC:{round(auroc, 4)}(MAX:{round(best_record[0], 4)})"
                  f"  P-AUROC{round(full_pixel_auroc, 4)}(MAX:{round(best_record[1], 4)}) -----"
                  f"  PRO-AUROC{round(pro, 4)}(MAX:{round(best_record[2], 4)}) -----")
        
        # save train progress as json
        train_progress = {
            'y_loss_fake': self.y_loss_fake,
            'x_loss_fake': self.x_loss_fake,
            'y_loss_true': self.y_loss_true,
            'x_loss_true': self.x_loss_true,
            'y_loss_total': self.y_loss_total,
            'x_loss_total': self.x_loss_total,
            'y_auc': self.y_auc,
            'x_auc': self.x_auc,
            'y_pseudo_auc': self.y_pseudo_auc,
            'x_pseudo_auc': self.x_pseudo_auc
            }
        
        with open(os.path.join(self.train_progress_dir, f'train_progress_{self.category}.json'), 'w') as fp:
            json.dump(train_progress, fp)
        
        torch.save(state_dict, os.path.join(self.model_dir, f'discriminator_{self.category}.pth'))
        
        return best_record
    
    def _evaluate(self, scores, segmentations, features, labels_gt, masks_gt):
        if not self.only_img_lvl:
            print('Total pixel-level auc-roc score:')
            pixel_auc = roc_auc_score(masks_gt, segmentations)
            print(pixel_auc)
        else:
            pixel_auc = 0.0
        scores = np.squeeze(np.array(scores))
        print('scores', scores.shape)
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)
        scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)
        print('scores', scores.shape)
        labels_gt = np.squeeze(np.array(labels_gt))
        print('labels_gt', labels_gt.shape)
        # img_auc = roc_auc_score(labels_gt, scores)
        img_auc = roc_auc_score(labels_gt, scores)
        print('Total image-level auc-roc score:')
        print(img_auc)

        pro = 0.0
        
        return img_auc, pixel_auc, pro
        
    def _train_discriminator(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval() # so they won't be trained and stay fixed. Mainly the backbone. 
        
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()
        # self.feature_enc.eval()
        # self.feature_dec.eval()
        
        # LOGGER.info(f"Training discriminator...")
        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                all_p_interp = []
                embeddings_list = []
                for data_item in input_data:
                    self.dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()
                    # self.dec_opt.zero_grad()

                    self.i_iter += 1
                    img, _, _, _, _ = data_item#["image"]
                    img = img.to(torch.float).to(self.device)
                    backbone_device = self.device if not self.quantize_qint8 else 'cpu'
                    true_feats = _feature_extraction(img, self.forward_modules, backbone_device)
                    if self.pooling_embedding:
                        true_feats = alternative_pooling(true_feats, self.batch_size)
                        if len(true_feats.shape) == 3:
                            # true_feats = true_feats.permute(0, 2, 3, 1
                            true_feats = true_feats.view(-1, true_feats.shape[2])
                        
                    else:
                        true_feats = _embed(true_feats, self.forward_modules, self.patch_maker)
                    # print('true_feats', true_feats.shape)
                    if self.pre_proj > 0:
                        tf_device = true_feats.device
                        if tf_device != self.device:
                            true_feats = true_feats.to(self.device)
                        true_feats = self.pre_projection(true_feats)
                    noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
                    # print('noise_idxs', noise_idxs.shape)
                    noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(self.device) # (N, K)
                    # print('noise_one_hot', noise_one_hot.shape)
                    noise = torch.stack([
                        torch.normal(0, self.noise_std * 1.1**(k), true_feats.shape)
                        for k in range(self.mix_noise)], dim=1).to(self.device) # (N, K, C)
                    noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
                    # print('noise', noise.shape)
                    fake_feats = true_feats + noise
                    # print('fake_feats', fake_feats.shape)
                    combined_features = torch.cat([true_feats, fake_feats])
                    scores = self.discriminator(combined_features)
                    true_scores = scores[:len(true_feats)]
                    fake_scores = scores[len(fake_feats):]
                    th = self.dsc_margin
                    # print(th)
                    p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                    p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
                    true_loss = torch.clip(-true_scores + th, min=0)
                    fake_loss = torch.clip(fake_scores + th, min=0)
                    loss = true_loss.mean() + fake_loss.mean()

                    loss.backward()
                    if self.pre_proj > 0:
                        self.proj_opt.step()

                    self.dsc_opt.step()

                    loss = loss.detach().cpu() 
                    all_loss.append(loss.item())
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())
                    
                    self.y_loss_fake.append(fake_loss.mean().detach().cpu().item())
                    self.x_loss_fake.append(self.i_iter)
                    self.y_loss_true.append(true_loss.mean().detach().cpu().item())
                    self.x_loss_true.append(self.i_iter)
                    self.y_loss_total.append(loss.detach().cpu().item())
                    self.x_loss_total.append(self.i_iter)
                
                if len(embeddings_list) > 0:
                    self.auto_noise[1] = torch.cat(embeddings_list).std(0).mean(-1)
                
                if self.lr_scheduler:
                    self.dsc_schl.step()
                
                all_loss = sum(all_loss) / len(input_data)
                all_p_true = sum(all_p_true) / len(input_data)
                all_p_fake = sum(all_p_fake) / len(input_data)
                pseudo_auc = (all_p_true + all_p_fake) / 2
                self.y_pseudo_auc.append(pseudo_auc)
                self.x_pseudo_auc.append(self.i_iter)
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str = f"epoch:{i_epoch} loss:{round(all_loss, 5)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                if len(all_p_interp) > 0:
                    pbar_str += f" p_interp:{round(sum(all_p_interp) / len(input_data), 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)


    def predict(self, data, prefix=""):
        if not self.measure_inference:
            if isinstance(data, torch.utils.data.DataLoader):
                return self._predict_dataloader(data, prefix)
            return self._predict(data)
        else:
            t_fe_total = []
            t_em_total = []
            t_sp_total = []
            t_si_total = []
            run_times = {
                '#1 feature extraction cpu': [],
                '#2 feature extraction gpu': [],
                '#3 embedding of features cpu': [],
                '#4 embedding of features gpu': [],
                '#5 score patches cpu': [],
                '#6 score patches gpu': [],
                '#7 img lvl score cpu': [],
                '#8 img lvl score gpu': [],
                '#9 anomaly map cpu': [],
                '#10 anomaly map gpu': [],
                '#11 whole process cpu': [],
                '#12 whole process gpu': [],
                '#13 dim reduction cpu': [],
                '#14 dim reduction gpu': []          
            }
            if isinstance(data, torch.utils.data.DataLoader):
                # warm up
                for _ in range(self.warm_up_reps):
                    _ =  self._predict_dataloader(data, prefix)
                # measurement
                for _ in range(self.number_of_reps):
                    scores, masks, features, labels_gt, masks_gt, t_fe, t_em, t_sp, t_si = self._predict_dataloader(data, prefix)
                    t_fe_total.append(t_fe * 1e3) # feature extraction
                    t_em_total.append(t_em * 1e3) # embedding
                    t_sp_total.append(t_sp * 1e3) # score patches
                    t_si_total.append(t_si * 1e3) # score image
                run_times['#1 feature extraction cpu'] = t_fe_total
                run_times['#2 feature extraction gpu'] = 0.0
                run_times['#3 embedding of features cpu'] = t_em_total
                run_times['#4 embedding of features gpu'] = 0.0
                run_times['#5 score patches cpu'] = t_sp_total
                run_times['#6 score patches gpu'] = 0.0
                run_times['#7 img lvl score cpu'] = t_si_total
                run_times['#8 img lvl score gpu'] = 0.0
                run_times['#9 anomaly map cpu'] = 0.0
                run_times['#10 anomaly map gpu'] = 0.0
                run_times['#11 whole process cpu'] = np.array(t_fe_total) + np.array(t_em_total) + np.array(t_sp_total) + np.array(t_si_total)
                run_times['#12 whole process gpu'] = 0.0
                run_times['#13 dim reduction cpu'] = 0.0
                run_times['#14 dim reduction gpu'] = 0.0
                return scores, masks, features, labels_gt, masks_gt, run_times


            else:
                raise NotImplementedError('measurement for single picture not implemented yet.')

    def _predict_dataloader(self, dataloader, prefix):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        img_paths = []
        scores = []
        labels_gt = []
        
        if not self.only_img_lvl:
            masks = []
            features = []
            masks_gt = []
        
        if self.measure_inference:
            total_fe = []
            total_em = []
            total_sp = []
            total_si = []

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            if not self.only_img_lvl:
                for data in data_iterator:
                    image, mask, label, img_path, img_type = data
                    # if isinstance(data, dict):
                    labels_gt.extend(label.numpy().tolist())
                    if mask is not None:
                        masks_gt.extend(mask.numpy().tolist())
                    # image = data["image"]
                    img_paths.extend(img_path)
                    if not self.measure_inference:
                        _scores, _masks, _feats = self._predict(image)
                    else:
                        _scores, _masks, _feats, t_1_cpu, t_2_cpu, t_3_cpu = self._predict(image)
                for score, mask in zip(_scores, _masks, ):
                    scores.append(score)
                    masks.append(mask)
                return scores, masks, features, labels_gt, masks_gt, t_1_cpu, t_2_cpu, t_3_cpu
            else:
                for data in data_iterator:
                    image, mask, label, img_path, img_type = data
                    # if isinstance(data, dict):
                    labels_gt.extend(label.numpy().tolist())
                    # image = image
                    img_paths.extend(img_path)
                    if not self.measure_inference:
                        _scores, _, _ = self._predict(image)
                    else:
                        _scores, _, _, t_fe, t_em, t_sp, t_si  = self._predict(image)
                    #for score, time_fe, time_em, time_sc in zip(_scores, time_fe, time_em, time_sc):
                    for score in _scores:
                        scores.append(score)
                    if self.measure_inference:
                        total_fe.append(t_fe)
                        total_em.append(t_em)
                        total_sp.append(t_sp)
                        total_si.append(t_si)
                if self.measure_inference:
                    t_fe = np.mean(total_fe) / len(_scores)
                    t_em = np.mean(total_em) / len(_scores)
                    t_sp = np.mean(total_sp) / len(_scores)
                    t_si = np.mean(total_si) / len(_scores)
                    return scores, None, None, labels_gt, None, t_fe, t_em, t_sp, t_si
                else:
                    return scores, None, None, labels_gt, None
            
    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        # this_device = self.device
        # image_device = images.device
        # print('image_device', image_device)
        # print('1: ',self.device)
        
        images = images.to(torch.float)#.to(this_device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        with torch.no_grad():
            # feature extraction
            if not self.measure_inference:
                backbone_device = self.device if not self.quantize_qint8 else 'cpu'
                features = _feature_extraction(images, 
                                            self.forward_modules,
                                            device=backbone_device)
                if self.pooling_embedding:
                    features = alternative_pooling(features, batchsize)
                    patch_shapes = None
                else:
                    features, patch_shapes = _embed(features,
                                                    self.forward_modules,
                                                    self.patch_maker,
                                                    provide_patch_shapes=True)#, 
                                                        #evaluation=True)
                if self.pre_proj > 0:
                    if features.device != self.device:
                        features = features.to(self.device)
                        # print(f'Exectute on {self.device}')
                    features = self.pre_projection(features) #torch.Size([6272, 1536])


                score_patches = -self.discriminator(features) #torch.Size([6272, 1]) --> for each patch one score
                score_patches = score_patches.cpu().numpy()
                image_scores = score_patches.copy()

                image_scores = self.patch_maker.unpatch_scores(
                    image_scores, batchsize=batchsize
                ) # (8, 784, 1)
                # image_scores = image_scores.reshape(*image_scores.shape[:2], -1) # redundant
                image_scores = self.patch_maker.score(image_scores)

                # score_patches = self.patch_maker.unpatch_scores(
                #     score_patches, batchsize=batchsize
                # )
                if not self.only_img_lvl:
                    scales = patch_shapes[0]
                    score_patches = score_patches.reshape(batchsize, scales[0], scales[1])
                    features = features.reshape(batchsize, scales[0], scales[1], -1)
                    masks, features = self.anomaly_segmentor.convert_to_segmentation(score_patches, features)
                    return list(image_scores), list(masks), list(features)
                else:
                    return list(image_scores), None, None#, None, None
            else:
                # feature extraction
                t_0_cpu = perf_counter()
                backbone_device = self.device if not self.quantize_qint8 else 'cpu'
                features = _feature_extraction(images,
                            self.forward_modules,
                            device=backbone_device)
                # embedding
                t_1_cpu = perf_counter()
                # features, patch_shapes = _embed(features,
                #                                 self.forward_modules,
                #                                 self.patch_maker,
                #                                 provide_patch_shapes=True)
                if self.pooling_embedding:
                    features = alternative_pooling(features, batchsize)
                    patch_shapes = None
                else:
                    features, patch_shapes = _embed(features,
                                                    self.forward_modules,
                                                    self.patch_maker,
                                                    provide_patch_shapes=True)#, 
                # # projection
                if self.pre_proj > 0:
                    if features.device != self.device:
                        features = features.to(self.device)
                        # print(f'Exectute on {self.device}')
                    features = self.pre_projection(features)
                t_2_cpu = perf_counter()
                # discriminator
                score_patches = -self.discriminator(features)
                score_patches = score_patches.cpu().numpy()

                t_3_cpu = perf_counter()
                image_scores = score_patches.copy()

                image_scores = self.patch_maker.unpatch_scores(
                    image_scores, batchsize=batchsize
                )
                image_scores = self.patch_maker.score(image_scores)
                t_4_cpu = perf_counter()
                if not self.only_img_lvl:
                    scales = patch_shapes[0]
                    score_patches = score_patches.reshape(batchsize, scales[0], scales[1])
                    features = features.reshape(batchsize, scales[0], scales[1], -1)
                    masks, features = self.anomaly_segmentor.convert_to_segmentation(score_patches, features)
                    return list(image_scores), list(masks), list(features), t_1_cpu-t_0_cpu, t_2_cpu-t_1_cpu, t_3_cpu-t_2_cpu, t_4_cpu-t_3_cpu
                else:
                    return list(image_scores), None, None, t_1_cpu-t_0_cpu, t_2_cpu-t_1_cpu, t_3_cpu-t_2_cpu, t_4_cpu-t_3_cpu
    
    def train_dataloader(self):
        '''
        load training data
        uses attributes to determine which dataset to load
        '''
        image_datasets = MVTecDataset(root=os.path.join(self.dataset_path,self.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train')#, half=self.quantization)
        train_loader = DataLoader(image_datasets, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
        return train_loader

    def test_dataloader(self):
        '''
        load test data
        uses attributes to determine which dataset to load
        '''
        test_datasets = MVTecDataset(root=os.path.join(self.dataset_path,self.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')#, half=self.quantization)
        test_loader = DataLoader(test_datasets, batch_size=self.batch_size_test, shuffle=False, num_workers=self.num_workers)
        return test_loader
    

    
# def compute_imagewise_retrieval_metrics(
#     anomaly_prediction_weights, anomaly_ground_truth_labels
# ):
#     """
#     Computes retrieval statistics (AUROC, FPR, TPR).

#     Args:
#         anomaly_prediction_weights: [np.array or list] [N] Assignment weights
#                                     per image. Higher indicates higher
#                                     probability of being an anomaly.
#         anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
#                                     if image is an anomaly, 0 if not.
#     """
#     from sklearn import metrics
#     # fpr, tpr, thresholds = metrics.roc_curve(
#     #     anomaly_ground_truth_labels, anomaly_prediction_weights
#     # )
#     auroc = metrics.roc_auc_score(
#         anomaly_ground_truth_labels, anomaly_prediction_weights
#     )
    
#     # precision, recall, _ = metrics.precision_recall_curve(
#     #     anomaly_ground_truth_labels, anomaly_prediction_weights
#     # )
#     # auc_pr = metrics.auc(recall, precision)
    
#     return auroc#, "fpr": fpr, "tpr": tpr, "threshold": thresholds}
    
    
if __name__ == "__main__":
    import gc
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # model = SimpleNet(device)
    cats = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 
    'tile', 'screw', 'toothbrush', 'transistor', 'wood', 'own', 'zipper']
  
    for k, cat in enumerate(cats):
        model = SimpleNet(device)
        model.group_id = 'RN18_default'
        model.exclude_relu = True
        model.quantize_qint8 = True
        model.pooling_embedding = True
        model.backbone_id = 'RN18'
        model.layers_to_extract_from = [2,3]
        model.dsc_margin = 0.5
        model.meta_epochs = 20
        model.pre_proj = 1
        model.dsc_hidden = None
        model.target_embed_dimensions = 384
        model.pretrain_embed_dimensions = 384
        model.dsc_layers = 2
        # model.dsc_hidden = 256
        model.category = cat
        model.measure_inference = False
        model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'\n{k+1}/{len(cats)}: Training...{cat}\n')
        model.train()
        model.measure_inference = True
        model.device = "cpu"
        print(f'\n{k+1}/{len(cats)}: Testing...{cat}\n')    
        model.test()
        del model
        gc.collect()
    