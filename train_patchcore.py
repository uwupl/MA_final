
from utils.backbone import Backbone, prune_naive, prune_model_nni, prune_output_layer, quantize_model, compress_model_nni
from utils.feature_adaptor import get_feature_adaptor, FeatureAdaptor
from utils.datasets import MVTecDataset
from utils.utils import min_max_norm, heatmap_on_image, cvt2heatmap, record_gpu, modified_kNN_score_calc, calc_anomaly_map #  distance_matrix, softmax
from utils.pooling import adaptive_pooling
from utils.embedding import reshape_embedding, embedding_concat_frame, PatchMaker, _embed
from utils.search import KNN, NearestNeighbourScorer, FaissNN, ApproximateFaissNN
import utils.common as common
# from utils.embedding import PatchMaker, _embed
from utils.quantize import quantize_model_into_qint8
from utils.kcenter_greedy import kCenterGreedy
from path_definitions import ROOT_DIR, RES_DIR, PLOT_DIR, MVTEC_DIR, EMBEDDING_DIR

import os
import numpy as np
import pandas as pd
from PIL import Image
# import cv2
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import roc_auc_score
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import faiss
import pickle
from sklearn.neighbors import NearestNeighbors
import time
from time import perf_counter as record_cpu
import copy
import platform

raspberry_pi = False if platform.machine().__contains__('x86') else True # global variable to determine if we are on a raspberry pi or not

class PatchCore(pl.LightningModule):
    '''
    __init__:
        - initialize all parameters
    fit:
        on_train_start:
            - set paths
            - load initial model based on args
            - determine intial output size
            - pruning of model
            - iterative pruning
                - channel selection (optional)
                - pruning (optional)
            - quantization of model (TODO)
        training_step:
            - forward pass: feature extraction --> embedding
            - save as numpy array which is attribute of class
        training_epoch_end:
            - select channels (optional)
            - quantize model (optional, maybe not clever at this point)
            - sampling of coreset
            - choose search engine
            - save coreset
            - save model
    test:
        on_test_start:
            - load model
            - load coreset
            - initialize search engine
        test_step: (self.only_img_lvl = True)
            - devided into subfunctions: _test_step 
                feature_extraction
                feature_embedding
                calc_score_patches
                calc_img_score
            - measure inference time utilizing time.perf_counter() (optional)
            - save results (score (float) for each img)
        test_epoch_end:
            - determines img_auc (and pxl_auc if self.only_img_lvl = False) using sklearn.metrics.roc_auc_score
            - creates pandas dataframe with results and all settings
    '''
    def __init__(self):
        super(PatchCore, self).__init__()
        
        ### BASIC OPTIONS ###
        self.category = 'own' # category to be used; cats are: 'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper' and 'own'
        self.dataset_path = MVTEC_DIR # needs to be adapted in path_definitions.py
        self.only_img_lvl = True # skipps anomaly map calculation and only calculates img_auc
        
        ### BASIC BACKBONE OPTIONS ###
        self.backbone_id = "WRN50"
        self.layers_needed = [2,3]
        self.layer_cut = True
        
        ### NAMING OPTIONS ###
        self.time_stamp = f'{int(time.time())}'
        self.group_id = 'not_specified'
        
        ### ANOMALY SCORE CALCULATION ###
        self.adapted_score_calc = False # if True, alternative score calculation is used
        self.n_next_patches = 0.03 # relative number of next patches to be used for score calculation if self.adapted_score_calc = True
        self.n_neighbors = 9 # number of nearest neighbours to be used
        self.patchcore_scorer = True # if True, patchcore method is used for score calculation
        
        ### DATALOADER OPTIONS ###
        self.load_size = 288 # size of image to be loaded
        self.input_size = 256 # size of image to be used as input for model
        self.batch_size = 1 # batch size for training
        self.batch_size_test = 1 # not used until now, only for simplification of the code; only batch_size=1 is implemented
        self.num_workers = 12 # number of workers for dataloader
        
        #### NN SEARCH ####
        self.faiss_standard = False # own implementation of standard faiss search
        self.faiss_quantized = False # own implementation of quantized faiss search; lil bit faster than standard faiss search, not as accurate, not worth it
        self.patchcore_score_patches = False # patchcore search
        self.own_knn = True # search with cdist
        # following only relevant if self.own_knn = True
        self.metrices = { 
            0:'euclidean', # 0.88
            1:'minkowski', # nur mit p spannend
            2:'cityblock', # manhattan
            3:'chebyshev',
            4:'cosine',
            5:'correlation',
            6:'hamming',
            7:'jaccard',
            8:'braycurtis',
            9:'canberra',
            10:'jensenshannon',
            # 11:'matching', # sysnonym for hamming
            11:'dice',
            12:'kulczynski1',
            13:'rogerstanimoto',
            14:'russellrao',
            15:'sokalmichener',
            16:'sokalsneath',
            # 18:'wminkowski',
            17:'mahalanobis',
            18:'seuclidean',
            19:'sqeuclidean',
            }
        self.metric_id = 0 # choose metric from self.metrices
        self.metrics_p = None # only relevant if self.metric_id = 1
        
        ### POOLING ###
        self.pooling_embedding = True # if True own and faster implementation of embedding process is used
        # only relevant if self.pooling_embedding = False --> patch core neighborhood aggregation
        self.pretrain_embed_dimensions = 1024 # defines dimension of embedding before aggregation
        self.target_embed_dimensions = 1024 # basically not used
        # only relevant if self.pooling_embedding = True
        self.pooling_strategy = ['default'] # loads of options, see pooling.py; you can also use multiple strategies at once
        
        ### CORESET SUBSAMPLING ###
        self.coreset_sampling_ratio = 0.0 # ratio of coreset to be used, only relevant if self.specifc_number_of_examples == 0
        self.specific_number_of_examples = int(0) # number of examples to be used for coreset
        self.coreset_sampling_method = 'k_center_greedy' # methode to be used for subsampling; options: 'k_center_greedy', 'random_selection', 'sparse_projection'
        self.random_presampling = (False, 10000) # if the number of extracted patch feature vectors is too large, which would lead to memory issues, we can randomly select a subset of the patches before calculating the coreset
        self.multiple_coresets = (True, 5) # if True, multiple coresets are calculated and saved to compensate for the randomness of the k_center_greedy algorithm
        
        ### DIMENSIONALITY REDUCTION ###
        self.reduce_via_std = False # calcs std of each channel and sets it as selection criterion
        self.reduce_via_entropy = False # outdated; only approximated entropy is used
        self.reduce_via_entropy_normed = False # outdated; only approximated entropy is used
        self.reduce_via_real_entropy = False # use this instead! calcs entropy as selection criterion
        self.reduce_via_pca = False # pca for dimensionality reduction
        self.reduce_via_random = False # selects random channels
        self.reduce_by_heigth = True # default is True, which means the highest are chosen --> False means the lowest are chosen
        self.pool_depth = False # pool along depth dimension with adaptive pooling 1D
        self.pca = None # aux variable for pca
        self.adapt_feature = False # if True, feature adaptor is used to adapt the feature extraction process; see SimpleNet or utils.feature_adaptor.py
        self.feature_adaptor = None # aux variable for feature adaptor
        self.feature_adaptor_dict = None # aux variable for feature adaptor to set hyperparameters; not necessary
        self.pretrain_for_channel_selection = False # channel selection is done in the beginning of training in order to be able to prune the model's output layer; not maintained anymore since it was not worth it
        self.idx_chosen = None # aux variable for channel selection
        self.reduction_factor = 75 # decides how many channels REMAIN! --> 75 means 25% of channels are removed
        
        ### FEATURE TRANSFORMATION ###
        # activation functions
        self.exclude_relu = False # identity 
        self.sigmoid_in_last_layer = False # sigmoid
        self.lrelu_in_last_layer = False # leaky relu with p = 0.2
        self.softmin_in_last_layer = False # softmin
        self.need_for_own_last_layer = False # aux variable for channel selection; no need to adapt manually
        # norm and weighting
        self.normalize = False
        self.weight_by_entropy = False # weights channels by sort of entropy; outdated
        self.weight_by_real_entropy = False # weights channels by entropy; use this instead!
        self.mean = None # aux variable for normalization
        self.std = None # aux variable for normalization
        self.weights = None # aux variable for weighting
        # dataloader transformation; just leave it as it is
        self.category_wise_statistics = False # if True, mean and std are calculated for each category and used for normalization; not maintained since it has not a beneficial effect
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
                transforms.Resize((self.load_size, self.load_size), Image.LANCZOS),
                transforms.ToTensor(),
                transforms.CenterCrop(self.input_size),
                transforms.Normalize(mean=means,
                                    std=stds)]) # from imagenet  # for each category calculate mean and std TODO
        self.gt_transforms = transforms.Compose([
                        transforms.Resize((self.load_size, self.load_size)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(self.input_size)])
        self.inv_normalize = transforms.Normalize(mean=list(np.divide(np.multiply(-1,means), stds)), std=list(np.divide(1, stds))) # aus imagenet
        
        ### DEBUG OPTIONS ###
        self.save_features = False # saves features as numpy array
        if self.save_features: 
            self.features_to_store = []
        self.save_embeddings = False # saves embeddings as numpy array
        self.calc_uncertainty = False # calcs std of AUROC of multiple coresets; therefore only relevant if self.multiple_coresets[0] = True
        self.save_am = False # saves anomaly map
        
        ### RUNTIME MEASUREMENT ###
        self.measure_inference = False # measures inference time
        self.number_of_reps = 50 # number of reps during measurement. Beacause we can assume a consistent estimator, results get more accurate with more reps
        self.warm_up_reps = 10 # before the actual measurement is done, we execute the process a couple of times without measurement to ensure that there is no influence of initialization and that the circumstances (e.g. thermal state of hardware) are representive.
        
        ### HARDWARE SPECIFIC OPTIONS ###
        self.cpu_arch = 'x86' # if anything else than x86, qnnpack is used; relevant for quantization; set by script itself, no need to adapt manually
        self.cuda_active = False # if True, cuda is used for inference
        self.cuda_active_training = True # if True, cuda is used for training
        
        ### QUANTIZE BACKBONE ###
        self.quantize_model_with_nni = False # outdates; has not a beneficial effect, but maybe there is some potential with a new version of nni
        self.quantize_model_pytorch = False # outdeated; has not a beneficial effect
        self.quantize_qint8 = False #  quantizes model to qint8; only works for CPU; compatible with all backbones and all activation functions (!)
        self.calibration_dataset = 'imagenet' # options: 'target', 'mvtec', 'imagenet', 'random' or None for no calibration and loading pretrained model; only relevant if self.quantize_qint8 = True
        self.num_images_calib = 100 # number of images to be used for calibration; only relevant if self.quantize_qint8 = True
        self.quantize_qint8_torchvision = False # loads already quantized torchvision models; only RN18 and RN50 are available for now
        
        ### PRUNING BACKBONE ###
        self.prune_output_layer = (False, []) # prunes the output layer of the model; only relevant if self.pretrain_for_channel_selection = True
        self.prune_l1_unstructured = (False, 0.0) # utilizing the build-in torch pruning
        self.prune_torch_pruning = (False, 0.0) # using the pytorch-pruning library
        self.prune_structured_nni = (False, [], 'L1') # options: 'FPGM', 'L2', utilizing the nni pruning (microsoft)
        self.sparsity = 0.05
        self.iterative_pruning = (False, 0) # prunes the model iteratively
        
        ### OTHER ###
        self.criterion = torch.nn.MSELoss(reduction='sum') # despite not used in this script, it is needed for the training process since the pytorch lightning module requires it
        self.init_results_list()

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []

    def init_features(self):
        self.features = []

    def forward(self, x_t):
        '''
        extract features from model
        '''
        self.init_features()
        _ = self.backbone(x_t)
        return self.features
    
    # not used as long as self.only_img_lvl is True
    def save_anomaly_map(self, anomaly_map, input_img, gt_img, file_name, x_type):
        if anomaly_map.shape != input_img.shape:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm*255)
        hm_on_img = heatmap_on_image(heatmap, input_img)

        # save images
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)

    def train_dataloader(self):
        '''
        load training data
        uses attributes to determine which dataset to load
        '''
        image_datasets = MVTecDataset(root=os.path.join(self.dataset_path,self.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train')
        train_loader = DataLoader(image_datasets, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_loader

    def test_dataloader(self):
        '''
        load test data
        uses attributes to determine which dataset to load
        '''
        test_datasets = MVTecDataset(root=os.path.join(self.dataset_path,self.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=self.num_workers)
        return test_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        '''
        TODO
        '''
        ### temp ###
        # if self.adapt_feature:
        # initialize paths
        self.std = None
        self.mean = None
        self.weights = None
        self.log_path = os.path.join(os.path.dirname(__file__), "results","patchcore", f"{self.group_id}","csv")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.latences_filename = f'latences_{self.group_id}_{self.time_stamp}.csv'
        self.acc_filename = f'acc_{self.group_id}_{self.time_stamp}.csv'
        self.embedding_dir_path = os.path.join(EMBEDDING_DIR, self.group_id, self.category)
        # make sure that the embedding directory exists
        if not os.path.exists(self.embedding_dir_path):
            os.makedirs(self.embedding_dir_path)
    
        if not self.only_img_lvl:        #, self.sample_path, _ = prep_dirs(self.logger.log_dir, self.category)
            self.sample_path = os.path.join(RES_DIR, 'samples', self.category)
        
        if self.patchcore_scorer or self.patchcore_score_patches:
            self.anomaly_scorer = NearestNeighbourScorer(n_nearest_neighbours= 1 if self.patchcore_scorer else self.n_neighbors)
            # print('HERE: ', self.anomaly_scorer.n_nearest_neighbourss)
        # change device to cuda if qint8 quantization is used
        if self.quantize_qint8 or self.quantize_qint8_torchvision: # get it to work with cuda --> not possible with pytorch quantization
            self.cuda_active_training, self.cuda_active = False, False
        
        # self.need_for_own_last_layer = self.need_for_own_last_layer # TODO #,self.prune_output_layer[0] # if relu is last activation, but we want to prune the output layer, we need to set this to true to get own last layer
        if self.cuda_active_training or self.cuda_active:
            self.backbone = Backbone(model_id=self.backbone_id, layers_needed=self.layers_needed, layer_cut=self.layer_cut, 
                                     prune_output_layer=(False, []), prune_torch_pruning=self.prune_torch_pruning, 
                                     prune_l1_norm=self.prune_l1_unstructured, exclude_relu=self.exclude_relu, 
                                     sigmoid_in_last_layer = self.sigmoid_in_last_layer, lrelu_in_last_layer=self.lrelu_in_last_layer, softmin_in_last_layer=self.softmin_in_last_layer,
                                     need_for_own_last_layer=self.need_for_own_last_layer, 
                                     quantize_qint8_prepared=self.quantize_qint8, quantize_qint8_torchvision=self.quantize_qint8_torchvision).cuda().eval() #, prune_l1_norm=self.prune_l1_unstructured
            if self.quantize_qint8:
                raise NotImplementedError('qint8 quantization for GPU not implemented')
                # self.backbone = quantize_model_into_qint8(model=self.backbone, layers_needed=self.layers_needed, calibrate=self.calibration_dataset ,category=self.category, cpu_arch=self.cpu_arch, dataset_path=r"/mnt/crucial/UNI/IIIT_Muen/MA/MVTechAD/")
            
            self.dummy_input = torch.randn(1, 3, self.input_size, self.input_size).cuda()
        else:
            self.backbone = Backbone(model_id=self.backbone_id, layers_needed=self.layers_needed, layer_cut=self.layer_cut, 
                                     prune_output_layer=(False, []), prune_torch_pruning=self.prune_torch_pruning, prune_l1_norm=self.prune_l1_unstructured, 
                                     exclude_relu=self.exclude_relu, sigmoid_in_last_layer = self.sigmoid_in_last_layer, lrelu_in_last_layer=self.lrelu_in_last_layer,  softmin_in_last_layer=self.softmin_in_last_layer,
                                     need_for_own_last_layer=self.need_for_own_last_layer, quantize_qint8_prepared=self.quantize_qint8, quantize_qint8_torchvision=self.quantize_qint8_torchvision).eval() # prune_l1_norm=self.prune_l1_unstructured,
            if self.quantize_qint8:
                self.backbone = quantize_model_into_qint8(model=self.backbone, layers_needed=self.layers_needed, calibrate=self.calibration_dataset if not raspberry_pi else None, 
                                                          exclude_relu=self.exclude_relu, sigmoid_in_last_layer = self.sigmoid_in_last_layer, lrelu_in_last_layer=self.lrelu_in_last_layer,  softmin_in_last_layer=self.softmin_in_last_layer,
                                                          category=self.category,  num_images=self.num_images_calib, dataset_path=r"/mnt/crucial/UNI/IIIT_Muen/MA/MVTechAD/")
                if raspberry_pi:
                    # load pretrained, quantized and calibrated models; only RN18 and RN34 for now
                    file_name = f'{self.backbone_id}_layers_{self.layers_needed}_qint8_qnnpack_calib_random.pth'
                    path_to_model = os.path.join('/home/jo/MA/code/MA_complete/quantized_models', file_name)
                    self.backbone.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
            self.dummy_input = torch.randn(1, 3, self.input_size, self.input_size)
        # determine output shape of model

        features = self.feature_extraction(self.dummy_input)
        if self.pooling_embedding:
            embeddings = self.feature_embedding(features, True, 1)
        else:
            self.patch_size = 3
            self.patch_stride = 1
            self.embedding_size = None # TODO --> What does that do?
            # Projection
            self.pre_proj = 1 # TODO
            self.proj_layer_type = 0 # TODO
            self.top_k = 3
            self.patch_maker = PatchMaker(patchsize=self.patch_size,top_k=self.top_k, stride=self.patch_stride)
            preprocessing = common.Preprocessing(input_dims = self.backbone.feature_dim, output_dim = self.pretrain_embed_dimensions).to(self.device)
            self.forward_modules = torch.nn.ModuleDict({})
            self.forward_modules['preprocessing'] = preprocessing
            
            preadapt_aggregator = common.Aggregator(target_dim = self.target_embed_dimensions).to(self.device)
            self.forward_modules['preadapt_aggregator'] = preadapt_aggregator
            embeddings = _embed(features, self.forward_modules, self.patch_maker)#, True, 1)
        self.output_shape = embeddings.shape # per picture
        self.idx_chosen = np.arange(self.output_shape[1]) # initialize idx_chosen with all channels
        # self.output_shape = 
        if self.prune_structured_nni[0]: #(bool, config_list, method)  
            config_list = []
            for name, module in self.backbone.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    if name != 'model.2.block_2.final_3':  # Skip the last Conv2d layer
                    # if name != '4.2.45'
                        # continue
                        config_list.append({
                            'op_types': ['Conv2d'],  # Prune only Conv2d layers
                            'op_names': [name],  
                            'sparsity': self.sparsity
                            })
                    else:
                        print('skipping')
                        config_list.append({
                            'op_names': [name],  # Prune the specific layer
                            'exclude': True  # Exclude this layer for pruning
                        })
            
            self.prune_structured_nni = (self.prune_structured_nni[0], config_list, self.prune_structured_nni[2])
        
        if self.iterative_pruning[0]:    
            for k in range(self.iterative_pruning[1]): 
                print(f'\nIteration of iterative Pruning and/or channel selection: {k+1} of {self.iterative_pruning[1]}\n')
                if self.pretrain_for_channel_selection:
                    # print('\n1\n')
                    # print('Pretrain for channel selection ...')
                    _ = self.select_channels()#total_embeddings, pretrain=True) # also prunes the model's output layer
                ### prune temp ###
                if self.prune_torch_pruning[0]:
                    self.backbone = prune_naive(self.backbone, self.prune_torch_pruning[1])
                if self.prune_structured_nni[0]: 
                    self.backbone = prune_model_nni(self.backbone, self.prune_structured_nni[1], self.prune_structured_nni[2]) # whole net
                ### prune temp ###
        else:# self.iterative_pruning[0]:
            if self.prune_torch_pruning[0]:
                self.backbone = prune_naive(self.backbone, self.prune_torch_pruning[1])
            if self.prune_structured_nni[0]:
                self.backbone = prune_model_nni(self.backbone, self.prune_structured_nni[1], self.prune_structured_nni[2])
        
        self.backbone.eval() # to stop running_var move
        # initialize numpy array for embeddings
        self.embedding_np = np.array([])

        # if not self.pooling_embedding:
        # self.pretrain_embed_dimensions = self.output_shape[1]#256 + 128 # for RN18
        # self.target_embed_dimensions = self.output_shape[1]#128 + 256 # for RN18
        self.patch_size = 3
        self.patch_stride = 1
        self.embedding_size = None # TODO --> What does that do?
        # Projection
        self.pre_proj = 1 # TODO
        self.proj_layer_type = 0 # TODO
        self.top_k = 3
        self.patch_maker = PatchMaker(patchsize=self.patch_size,top_k=self.top_k, stride=self.patch_stride)
        preprocessing = common.Preprocessing(input_dims = self.backbone.feature_dim, output_dim = self.pretrain_embed_dimensions).to(self.device)
        self.forward_modules = torch.nn.ModuleDict({})
        self.forward_modules['preprocessing'] = preprocessing
        
        preadapt_aggregator = common.Aggregator(target_dim = self.target_embed_dimensions).to(self.device)
        self.forward_modules['preadapt_aggregator'] = preadapt_aggregator
    
    def on_test_start(self):
        
        # initialize paths
        self.latences_filename = f'latences_{self.group_id}_{self.time_stamp}.csv'
        # self.log_path = os.path.join(RES_DIR,f"{self.group_id}", "csv")
        self.log_path = os.path.join(os.path.dirname(__file__), "results","patchcore", f"{self.group_id}","csv")
        self.embedding_dir_path = os.path.join(EMBEDDING_DIR, self.group_id, self.category)
        if not os.path.exists(self.embedding_dir_path):
            os.makedirs(self.embedding_dir_path)
    
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # get Backbone
        # temp for ensuring model is loaded correctly
        # if False:
        # del self.backbone
        if self.cuda_active and torch.cuda.is_available():
            self.backbone = torch.load(os.path.join(self.embedding_dir_path,'backbone.pth')).cuda()
            if self.quantize_qint8:
                raise NotImplementedError('qint8 quantization for GPU not implemented')
        else:
            if not (self.quantize_qint8 or self.quantize_qint8_torchvision): # then it is fine to load the model entirely
                self.backbone = torch.load(os.path.join(self.embedding_dir_path,'backbone.pth'), map_location=torch.device('cpu'))#.cpu()#, map_location=torch.device('cpu'))
            else: # in this case we have to create an instance of the model first and load the state dict afterwards to get the same configuration as during training
                 
                # args have to match training!
                self.backbone = Backbone(model_id=self.backbone_id, layers_needed=self.layers_needed, layer_cut=self.layer_cut, 
                                         prune_output_layer=self.prune_output_layer, prune_torch_pruning=self.prune_torch_pruning, 
                                         prune_l1_norm=self.prune_l1_unstructured, exclude_relu=self.exclude_relu, 
                                         sigmoid_in_last_layer = self.sigmoid_in_last_layer, lrelu_in_last_layer=self.lrelu_in_last_layer,  softmin_in_last_layer=self.softmin_in_last_layer,
                                         need_for_own_last_layer=self.need_for_own_last_layer, 
                                         quantize_qint8_prepared=self.quantize_qint8, quantize_qint8_torchvision=self.quantize_qint8_torchvision).eval() 
                if self.quantize_qint8:
                    self.backbone = quantize_model_into_qint8(model=self.backbone, layers_needed=self.layers_needed, calibrate=None, 
                                                              exclude_relu=self.exclude_relu, sigmoid_in_last_layer = self.sigmoid_in_last_layer, lrelu_in_last_layer=self.lrelu_in_last_layer,  softmin_in_last_layer=self.softmin_in_last_layer,
                                                              category=self.category, dataset_path=r"/mnt/crucial/UNI/IIIT_Muen/MA/MVTechAD/")
                    if not raspberry_pi:
                        self.backbone.load_state_dict(torch.load(os.path.join(self.embedding_dir_path,'backbone.pth'), map_location=torch.device('cpu')))
                    else:
                        # load pretrained, quantized and calibrated models; only RN18 and RN34 for now
                        file_name = f'{self.backbone_id}_layers_{self.layers_needed}_qint8_qnnpack_calib_random.pth'
                        path_to_model = os.path.join('/home/jo/MA/code/MA_complete/quantized_models', file_name)
                        self.backbone.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))                
        self.backbone.eval()
        print(self.backbone)
        if (not self.pooling_embedding) or self.patchcore_scorer:
            self.patch_size = 3
            self.patch_stride = 1
            self.top_k = 3
            self.patch_maker = PatchMaker(patchsize=self.patch_size,top_k=self.top_k, stride=self.patch_stride)
        
        # load feature adaptor
        if self.adapt_feature:
            device = torch.device("cuda" if self.cuda_active and torch.cuda.is_available() else "cpu")
            self.feature_adaptor = torch.load(os.path.join(self.embedding_dir_path,'feature_adaptor.pth')).to(device)
        
        # load coreset and initialize knn search
        if True:
            if not self.multiple_coresets[0] or self.multiple_coresets[1] == 1:
                if self.patchcore_score_patches:
                    from utils.search import NearestNeighbourScorer
                    self.anomaly_scorer = NearestNeighbourScorer(n_nearest_neighbours= 1 if self.patchcore_scorer else self.n_neighbors)
                    self.anomaly_scorer.load(os.path.join(self.embedding_dir_path,'faiss_patchcore'))
                    print(self.anomaly_scorer)
                elif self.faiss_standard or self.faiss_quantized:
                    self.index = faiss.read_index(os.path.join(self.embedding_dir_path,'index.faiss'))
                    if self.cuda_active:
                        res = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(res, 0 ,self.index)
                elif self.own_knn:
                    self.embedding_coreset = pickle.load(open(os.path.join(self.embedding_dir_path, 'embedding.pickle'), 'rb'))
                    if self.metrics_p is not None: # for minkowski distance (and some other I think)
                        self.knn = KNN(torch.from_numpy(self.embedding_coreset), k=self.n_neighbors, metric=self.metrices[self.metric_id], p=self.metrics_p)
                    elif self.metric_id == 17: # mahalanobis distance
                        print('loading inv_cov')
                        # inv_cov = pickle.load(open(os.path.join(self.embedding_dir_path, 'inv_cov.pickle'), 'rb')) # from training
                        inv_cov = np.load(os.path.join(self.embedding_dir_path, 'inv_cov.npy')) 
                        self.knn = KNN(torch.from_numpy(self.embedding_coreset), k=self.n_neighbors, metric=self.metrices[self.metric_id], inv_cov=inv_cov)
                    else:
                        self.knn = KNN(torch.from_numpy(self.embedding_coreset), k=self.n_neighbors, metric=self.metrices[self.metric_id])
                    
                else:
                    self.embedding_coreset = pickle.load(open(os.path.join(self.embedding_dir_path, 'embedding.pickle'), 'rb'))
                    self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree', metric='minkowski', p=2).fit(self.embedding_coreset)
            else:
                if self.patchcore_score_patches:
                    from utils.search import NearestNeighbourScorer
                    self.anomaly_scorer = []
                    for k in range(self.multiple_coresets[1]):
                        tmp_faiss = FaissNN()
                        tmp_scorer = NearestNeighbourScorer(n_nearest_neighbours= 1 if self.patchcore_scorer else self.n_neighbors, nn_method=tmp_faiss)
                        # if k==1:
                            # tmp_scorer.nn_method.reset_index()
                        tmp_scorer.load(os.path.join(self.embedding_dir_path,f'faiss_patchcore_{k}'))
                        self.anomaly_scorer.append(copy.deepcopy(tmp_scorer))
                        del tmp_scorer
                        del tmp_faiss
                        # tmp_scorer.nn_method.reset_index()
                        
                    # self.anomaly_scorer = [NearestNeighbourScorer(n_nearest_neighbours= 1 if self.patchcore_scorer else self.n_neighbors).load(os.path.join(self.embedding_dir_path,f'faiss_patchcore_{i}')) for i in range(self.multiple_coresets[1])] 
                    # print(self.anomaly_scorer)
                    # # self.anomaly_scorer = [self.anomaly_scorer[i].load(os.path.join(self.embedding_dir_path,f'faiss_patchcore_{i}')) for i in range(self.multiple_coresets[1])]
                    print('after loading')
                    print(self.anomaly_scorer)
                elif self.faiss_standard or self.faiss_quantized:
                    self.index = [faiss.read_index(os.path.join(self.embedding_dir_path,f'index_{i}.faiss')) for i in range(self.multiple_coresets[1])]
                    if self.cuda_active:
                        res = faiss.StandardGpuResources()
                        self.index = [faiss.index_cpu_to_gpu(res, 0 ,self.index[i]) for i in range(self.multiple_coresets[1])]
                elif self.own_knn:
                    if self.metrics_p is not None:
                        self.knn = [KNN(torch.from_numpy(pickle.load(open(os.path.join(self.embedding_dir_path, f'embedding_{n}.pickle'), 'rb'))), k=self.n_neighbors, metric=self.metrices[self.metric_id], p=self.metrics_p) for n in range(self.multiple_coresets[1])]
                    elif self.metric_id == 17: # mahalanobis distance
                        print('loading inv_cov')
                        inv_cov = np.load(os.path.join(self.embedding_dir_path, 'inv_cov.npy'))  #pickle.load(open(os.path.join(self.embedding_dir_path, 'inv_cov.pickle'), 'rb')) # from training
                        self.knn = [KNN(torch.from_numpy(pickle.load(open(os.path.join(self.embedding_dir_path, f'embedding_{n}.pickle'), 'rb'))), k=self.n_neighbors, metric=self.metrices[self.metric_id], inv_cov=inv_cov) for n in range(self.multiple_coresets[1])]
                    else:
                        self.knn = [KNN(torch.from_numpy(pickle.load(open(os.path.join(self.embedding_dir_path, f'embedding_{n}.pickle'), 'rb'))), k=self.n_neighbors, metric=self.metrices[self.metric_id]) for n in range(self.multiple_coresets[1])]
                else:
                    self.nbrs = [NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree', metric='minkowski', p=2).fit(pickle.load(open(os.path.join(self.embedding_dir_path, f'embedding_{n}.pickle'), 'rb'))) for n in range(self.multiple_coresets[1])]
        # # initialize results list
        if self.adapted_score_calc: # in order to get rid of the overhead of the first iterations
            for k in range(100):
                dummy_score_patches = np.random.rand(int(self.input_size/8)**2, self.n_neighbors) + 1e-15
                _ = modified_kNN_score_calc(score_patches=dummy_score_patches, n_next_patches=self.n_next_patches)
        
        
        self.init_results_list()
        
        # summary(self.backbone, depth=5, input_size=(1,3,224,224), col_names=['input_size', 'output_size', 'trainable', 'mult_adds', 'num_params'])
        
    def training_step(self, batch, batch_idx): # save locally aware patch features
        x, _, _, _, _ = batch
        features = self.backbone(x)

        # if self.quantize_qint8:
        #     features = list([features]) # TODO: probably because of missing forward hook there is no list of tensores as an output, but directly a tensor
        if self.save_features: # only one layer at a time!!
            self.features_to_store.append(features[0].detach().cpu())        
        if not self.backbone_id.__contains__('pdn'):
            if self.pooling_embedding:
                embeddings = self.feature_embedding(features, True, 1)
                if batch_idx == 0:
                    self.embedding_np = embeddings#.detach().cpu().numpy()
                else:
                    self.embedding_np = np.vstack((self.embedding_np, embeddings))
            else:
                embedded_feature = _embed(features, self.forward_modules, self.patch_maker)#, self.pre_proj, self.proj_layer_type)
                if batch_idx == 0:
                    self.embedding_np = embedded_feature.cpu().numpy()
                else:
                    self.embedding_np = np.vstack((self.embedding_np, embedded_feature.cpu().numpy()))
                
        else:
            reshaped_feats = reshape_embedding(features.detach().cpu().numpy())
            if batch_idx == 0:
                self.embedding_np = reshaped_feats
            else:
                self.embedding_np = np.vstack((self.embedding_np, reshaped_feats))
            
    def select_channels_core(self, total_embeddings):
        '''
        Select channels based on chosen scheme. If scheme is 'std', the channels with the highest std are chosen.
        Also prune of models output channels is done here
        '''
        if self.reduce_via_std:
            percentile_std = 100-self.reduction_factor
            if self.reduce_by_heigth:
                this_idx_chosen = set(np.argwhere(np.std(total_embeddings, axis=0)>np.percentile(np.std(total_embeddings,axis=0), percentile_std))[:,0]) # chooese channels with highest std
            else:
                this_idx_chosen = set(np.argwhere(np.std(total_embeddings, axis=0)<np.percentile(np.std(total_embeddings,axis=0), percentile_std))[:,0]) # chooese channels with lowest std
            idx_chosen_set = this_idx_chosen#set(self.idx_chosen).intersection(this_idx_chosen)
            self.idx_chosen = np.array(list(idx_chosen_set), dtype=np.int32)
        
        if self.normalize: # in order to emphasize the importance of the std, we normalize the embeddings, to achieve a more uniform importance of each channel
            self.mean = np.mean(total_embeddings, axis=0)
            self.std = np.std(total_embeddings, axis=0)
            self.std = self.std + 5e-2*np.mean(self.std) # add 5% of mean to std to avoid division by zero
            # self.std[self.std<1e-5] = 1.0
            print('std: ', self.std.shape)
            total_embeddings = (total_embeddings-self.mean)/self.std
            # total_embeddings[:,self.std<1e-15] = 0.0
        
        if self.reduce_via_entropy: # this is technically not entropy, but the same idea
            percentile_entropy = 100-self.reduction_factor
            total_embeddings_copy = total_embeddings.copy()
            total_embeddings_copy[total_embeddings_copy<1e-15] = 1e-15
            entropy = -np.sum(total_embeddings_copy*np.log2(total_embeddings_copy), axis=0)#.shape
            if self.reduce_by_heigth:
                idx_chosen_set = set(np.argwhere(entropy>np.percentile(entropy, percentile_entropy))[:,0]) # chooese channels with highest entropy
            else:
                idx_chosen_set = set(np.argwhere(entropy<np.percentile(entropy, percentile_entropy))[:,0]) # chooese channels with lowest entropy
            idx_chosen_set = idx_chosen_set.intersection(set(self.idx_chosen))
            self.idx_chosen = np.array(list(idx_chosen_set), dtype=np.int32)
        
        if self.reduce_via_entropy_normed: # here we norm each channel to 1 and then compute the entropy
            percentile_entropy = 100-self.reduction_factor
            total_embeddings_copy = total_embeddings.copy()
            total_embeddings_copy[total_embeddings_copy<1e-15] = 1e-15
            normed_embeddings = total_embeddings_copy/total_embeddings_copy.sum(axis=1, keepdims=1)
            entropy = -np.sum(normed_embeddings*np.log2(normed_embeddings), axis=0)#.shape
            if self.reduce_by_heigth:
                idx_chosen_set = set(np.argwhere(entropy>np.percentile(entropy, percentile_entropy))[:,0]) # chooese channels with highest entropy
            else:
                idx_chosen_set = set(np.argwhere(entropy<np.percentile(entropy, percentile_entropy))[:,0]) # chooese channels with lowest entropy
            idx_chosen_set = idx_chosen_set.intersection(set(self.idx_chosen))
            self.idx_chosen = np.array(list(idx_chosen_set), dtype=np.int32)
            
        if self.reduce_via_real_entropy:
            from scipy.stats import entropy
            percentile_entropy = 100-self.reduction_factor
            total_embeddings_copy = total_embeddings.copy()
            num_bins = 100
            entropy_per_channel = np.zeros(total_embeddings.shape[1])
            for k in range(total_embeddings.shape[1]):
                entropy_per_channel[k] = entropy(np.histogram(total_embeddings_copy[:,k], bins=num_bins, density=True)[0])
            if self.reduce_by_heigth:
                idx_chosen_set = set(np.argwhere(entropy_per_channel>np.percentile(entropy_per_channel, percentile_entropy))[:,0]) # chooese channels with highest entropy
            else:
                idx_chosen_set = set(np.argwhere(entropy_per_channel<np.percentile(entropy_per_channel, percentile_entropy))[:,0]) # chooese channels with lowest entropy
            idx_chosen_set = idx_chosen_set.intersection(set(self.idx_chosen))
            self.idx_chosen = np.array(list(idx_chosen_set), dtype=np.int32)
        
        if self.weight_by_entropy: # since we saw TODO
            total_embeddings_copy = total_embeddings.copy()
            total_embeddings_copy[total_embeddings_copy<1e-15] = 1e-15
            normed_embeddings = total_embeddings_copy/total_embeddings_copy.sum(axis=1, keepdims=1)
            entropy = -np.sum(normed_embeddings*np.log2(normed_embeddings), axis=0)#.shape
            # self.weights = softmax(entropy) * total_embeddings.shape[1]
            self.weights = entropy / np.sum(entropy) * total_embeddings.shape[1]
            total_embeddings = np.multiply(total_embeddings, self.weights)
        if self.weight_by_real_entropy:
            from scipy.stats import entropy
            from scipy.special import softmax
            total_embeddings_copy = total_embeddings.copy()
            num_bins = 100
            entropy_per_channel = np.zeros(total_embeddings.shape[1])
            for k in range(total_embeddings.shape[1]):
                entropy_per_channel[k] = entropy(np.histogram(total_embeddings_copy[:,k], bins=num_bins, density=True)[0])
            self.weights = np.float32(softmax(entropy_per_channel)*total_embeddings.shape[1])
            total_embeddings = np.multiply(total_embeddings, self.weights)
        if self.reduce_via_random:
            idx_chosen_set = set(np.random.choice(total_embeddings.shape[1], int(total_embeddings.shape[1]*self.reduction_factor/100), replace=False))
            idx_chosen_set = idx_chosen_set.intersection(set(self.idx_chosen))
            self.idx_chosen = np.array(list(idx_chosen_set), dtype=np.int32)
        
        if self.reduce_via_pca:
            from sklearn.decomposition import PCA
            n_components = int(total_embeddings.shape[1] * self.reduction_factor / 100)
            self.pca = PCA(n_components=n_components)
            self.pca.fit(total_embeddings)
            total_embeddings = self.pca.transform(total_embeddings)
        if (self.reduce_via_entropy or self.reduce_via_entropy_normed or self.reduce_via_real_entropy or self.reduce_via_random) and self.normalize and not self.reduce_via_std:
            self.std = np.take(self.std, self.idx_chosen)#, axis=0)
            self.mean = np.take(self.mean, self.idx_chosen)#, axis=0)
            if self.weight_by_entropy or self.weight_by_real_entropy:
                self.weights = np.take(self.weights, self.idx_chosen)
        if self.reduce_via_entropy or self.reduce_via_entropy_normed or self.reduce_via_std or self.reduce_via_real_entropy or self.reduce_via_random:
            print('Number of channels chosen: ', len(self.idx_chosen))
            total_embeddings = np.take(total_embeddings, self.idx_chosen, axis=1)
        if self.save_embeddings: # just for debugging
            file_name_embeddings = input('file name for embeddings:\n')
            np.save(file_name_embeddings + '.npy', total_embeddings)
        if (self.prune_output_layer[0] and (self.reduce_via_entropy or self.reduce_via_std or self.reduce_via_entropy_normed)):# or self.prune_l1_unstructured:
            # print('Pruning ...')
            self.prune_output_layer = (True, self.idx_chosen)
            features = self.feature_extraction(self.dummy_input)
            if self.pooling_embedding:
                embeddings = self.feature_embedding(features, True, 1)
            else:
                embeddings = _embed(features, self.forward_modules, self.patch_maker)#, True, 1)
            self.output_shape = embeddings.shape # per picture

            if len(self.layers_needed) == 1:
                self.backbone = prune_output_layer(self.backbone, self.idx_chosen, self.output_shape[1], input_size = (1,3,self.input_size,self.input_size))
            else:
                device = 'cuda' if next(self.backbone.parameters()).is_cuda else 'cpu'
                self.prune_output_layer = (True, self.idx_chosen)
                self.backbone = Backbone(model_id=self.backbone_id, layers_needed=self.layers_needed, layer_cut=self.layer_cut, 
                                         prune_output_layer=self.prune_output_layer, prune_torch_pruning=self.prune_torch_pruning, 
                                         prune_l1_norm=self.prune_l1_unstructured, exclude_relu=self.exclude_relu, 
                                         sigmoid_in_last_layer = self.sigmoid_in_last_layer, lrelu_in_last_layer=self.lrelu_in_last_layer,   softmin_in_last_layer=self.softmin_in_last_layer,
                                         need_for_own_last_layer=True, quantize_qint8_prepared=self.quantize_qint8, 
                                         quantize_qint8_torchvision=self.quantize_qint8_torchvision).to(device=device).eval() # prune_l1_norm=self.prune_l1_unstructured,
            
            features = self.feature_extraction(self.dummy_input.to(device=device))
            if self.pooling_embedding:
                embeddings = self.feature_embedding(features, True, 1)
            else:
                embeddings = _embed(features, self.forward_modules, self.patch_maker)#, True, 1)
            self.output_shape = embeddings.shape # per picture
            
            
        try:
            device = 'cuda' if next(self.backbone.parameters()).is_cuda else 'cpu'
            print('Model output shape: ', self.backbone(torch.randn(1,3,self.input_size,self.input_size).to(device))[0].shape)
            print('Number of channels chosen: ', len(self.idx_chosen))
            print('shape of total_embeddings: ', total_embeddings.shape)
        except:
            print('Something has failed. Probably the device is not able to be determined.')
        return total_embeddings
        
    def select_channels(self,total_embeddings=None):
        '''
        Based on either std or entropy, channels are selected and the embedding is reduced. Also the model gets pruned accordingly, if desired.
        '''
        if not self.pretrain_for_channel_selection:
            total_embeddings = self.select_channels_core(total_embeddings)
        else:
            print('Pretrain and select channels ...')
            train_loader = self.train_dataloader()
            self.embedding_np = np.array([])
            for batch_idx, batch in enumerate(train_loader):
                batch[0] = batch[0].to(device='cuda' if self.cuda_active_training else 'cpu')
                self.training_step(batch=batch, batch_idx=batch_idx)
            total_embeddings = self.embedding_np
            total_embeddings = self.select_channels_core(total_embeddings) #TODO naming of variables
        return total_embeddings

    def on_train_epoch_end(self):#, outputs):
        if self.save_features: # just for debugging
            file_name_features = input('file name for features:\n')
            for k1, el in enumerate(self.features_to_store):
                for k2, l in enumerate(el):
                    if k1 == 0 and k2 == 0:
                        feature_save = np.expand_dims(l.cpu().numpy(), axis=0)
                    feature_save = np.append(feature_save, np.expand_dims(l.cpu().numpy(), axis=0), axis=0)
            print(feature_save.shape)
            np.save(file_name_features + '.npy', feature_save)
            
        total_embeddings = self.embedding_np

        # select channels
        self.pretrain_for_channel_selection_copy = self.pretrain_for_channel_selection#.copy()
        self.pretrain_for_channel_selection = False
        
        total_embeddings = self.select_channels(total_embeddings)#, pretrain=False)
        # Random projection
        
        if self.adapt_feature:
            if self.feature_adaptor_dict is None:
                self.feature_adaptor = get_feature_adaptor(total_embeddings, shrinking_factor=float(self.reduction_factor/100), std_factor=0.01, batch_size=32, num_workers=12, lr=0.0005, epochs=12, use_cuda=torch.cuda.is_available()) # default arguments are used
            else:
                self.feature_adaptor = get_feature_adaptor(total_embeddings, **self.feature_adaptor_dict)
            # todo device check
            torch.save(self.feature_adaptor, os.path.join(self.embedding_dir_path, 'feature_adaptor.pth'))
            device='cuda:0' if (self.cuda_active and torch.cuda.is_available()) else 'cpu'
            self.feature_adaptor = self.feature_adaptor.to(device=device)
            torch_features = torch.from_numpy(total_embeddings).to(device=device)
            total_embeddings = self.feature_adaptor(torch_features).detach().cpu().numpy() #TODO
        
        self.pretrain_for_channel_selection = self.pretrain_for_channel_selection_copy#.copy()
        if self.quantize_model_with_nni:
            self.backbone = compress_model_nni(self.backbone)
            
        if self.quantize_model_pytorch:
            self.backbone = quantize_model(self.backbone)

        if self.random_presampling[0]:
            if self.random_presampling[1] > total_embeddings.shape[0]:
                self.random_presampling[1] = total_embeddings.shape[0]
            else:
                total_embeddings = total_embeddings[np.random.choice(total_embeddings.shape[0], int(self.random_presampling[1]), replace=False)]
        
        if self.specific_number_of_examples > 0:
            self.coreset_sampling_ratio = float(self.specific_number_of_examples/total_embeddings.shape[0])
        
        if not self.multiple_coresets[0] or self.coreset_sampling_ratio == 1.0:
            if self.coreset_sampling_ratio == 1.0:
                self.embedding_coreset = total_embeddings
            else:
                if self.coreset_sampling_method.__contains__('patchcore_greedy_approx'):
                    from utils.sampler import ApproximateGreedyCoresetSampler
                    this_cuda = (self.cuda_active or self.cuda_active_training) and torch.cuda.is_available()
                    self.subsampler = ApproximateGreedyCoresetSampler(self.coreset_sampling_ratio, device=torch.device('cuda') if this_cuda else torch.device('cpu'))
                    selected_idx = self.subsampler.run(total_embeddings)
                elif self.coreset_sampling_method.__contains__('sparse_projection'): # two different implementation that yield the same result (approximately)
                    self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
                    self.randomprojector.fit(total_embeddings)
                    # Coreset Subsampling
                    selector = kCenterGreedy(total_embeddings,0,0)
                    selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*self.coreset_sampling_ratio))
                elif self.coreset_sampling_method.__contains__('k_center_greedy'):
                    # total_embeddings_copy = total_embeddings.astype(np.float32)
                    if self.cuda_active or self.cuda_active_training or torch.cuda.is_available(): # use gpu anyway if available
                        sampler = k_center_greedy.KCenterGreedy(embedding=torch.from_numpy(total_embeddings).cuda(), sampling_ratio=self.coreset_sampling_ratio)
                    else:
                        sampler = k_center_greedy.KCenterGreedy(embedding=torch.from_numpy(total_embeddings), sampling_ratio=self.coreset_sampling_ratio)
                    selected_idx = sampler.select_coreset_idxs()
                elif self.coreset_sampling_method.__contains__('random_selection'):
                    selected_idx = np.random.choice(total_embeddings.shape[0], int(total_embeddings.shape[0]*self.coreset_sampling_ratio), replace=False)
                
                self.embedding_coreset = total_embeddings[selected_idx]
        else:
            self.embedding_coreset = None
            for k in range(self.multiple_coresets[1]):
                if self.coreset_sampling_method.__contains__('patchcore_greedy_approx'):
                    if k == 0:
                        from utils.sampler import ApproximateGreedyCoresetSampler
                    this_cuda = (self.cuda_active or self.cuda_active_training) and torch.cuda.is_available()
                    self.subsampler = ApproximateGreedyCoresetSampler(self.coreset_sampling_ratio, device=torch.device('cuda') if this_cuda else torch.device('cpu'))
                    selected_idx = self.subsampler.run(total_embeddings)
                elif self.coreset_sampling_method.__contains__('sparse_projection'):
                    self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
                    self.randomprojector.fit(total_embeddings)
                    # Coreset Subsampling
                    selector = kCenterGreedy(total_embeddings,0,0)
                    selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*self.coreset_sampling_ratio))
                elif self.coreset_sampling_method.__contains__('k_center_greedy'):
                    if self.cuda_active or self.cuda_active_training or torch.cuda.is_available(): # use gpu anyway if available
                        sampler = k_center_greedy.KCenterGreedy(embedding=torch.from_numpy(total_embeddings).cuda(), sampling_ratio=self.coreset_sampling_ratio)
                    else:
                        sampler = k_center_greedy.KCenterGreedy(embedding=torch.from_numpy(total_embeddings), sampling_ratio=self.coreset_sampling_ratio)
                    selected_idx = sampler.select_coreset_idxs()
                elif self.coreset_sampling_method.__contains__('random_selection'):
                    selected_idx = np.random.choice(total_embeddings.shape[0], int(total_embeddings.shape[0]*self.coreset_sampling_ratio), replace=False)
                
                if self.embedding_coreset is None:
                    self.embedding_coreset = np.expand_dims(total_embeddings[selected_idx], 0)
                else:
                    self.embedding_coreset = np.vstack((self.embedding_coreset, np.expand_dims(total_embeddings[selected_idx], 0)))
                    
        # summary(self.backbone, depth=5, input_size=(1,3,224,224), col_names=['input_size', 'output_size', 'trainable', 'mult_adds', 'num_params'])   
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        #faiss
        # in case coreset has less samples than required for neighbors
        idx = 1 if self.multiple_coresets[0] else 0
        if self.n_neighbors > self.embedding_coreset.shape[idx]:
            self.n_neighbors = self.embedding_coreset.shape[idx]
            
        # calculate inverse covariance matrix for mahalanobis distance
        if self.metric_id == 17: # mahalanobis distance
            print('calculating inverse covariance matrix')
            inv_cov = np.linalg.inv(np.cov(total_embeddings, rowvar=False))
            np.save(os.path.join(self.embedding_dir_path, 'inv_cov.npy'), inv_cov)
        
        if not self.multiple_coresets[0] or self.coreset_sampling_ratio == 1.0:
            if self.patchcore_score_patches:
                self.anomaly_scorer.fit([self.embedding_coreset])
                dir_path = os.path.join(self.embedding_dir_path,'faiss_patchcore')
                os.makedirs(dir_path, exist_ok=True)
                self.anomaly_scorer.save_and_reset(dir_path)
            elif self.faiss_quantized:
                # if False:
                nlist = 20 if self.embedding_coreset.shape[0] > 20 else self.embedding_coreset.shape[0]
                n_probe = 5 # defaul 1 # TODO
                quantizer = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_coreset.shape[1], nlist, faiss.METRIC_L2)
                assert not self.index.is_trained
                self.index.train(self.embedding_coreset)
                assert self.index.is_trained
                self.index.add(self.embedding_coreset)
                self.index.nprobe = n_probe
                faiss.write_index(self.index,  os.path.join(self.embedding_dir_path,'index.faiss'))
            elif self.faiss_standard:
                self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
                self.index.add(self.embedding_coreset) 
                faiss.write_index(self.index,  os.path.join(self.embedding_dir_path,'index.faiss'))
            else:
                print(self.embedding_coreset.shape)
                with open(os.path.join(self.embedding_dir_path, 'embedding.pickle'), 'wb') as f:
                    pickle.dump(self.embedding_coreset, f)
        else:
            for k in range(self.multiple_coresets[1]):
                if self.patchcore_score_patches:
                    self.anomaly_scorer.fit([self.embedding_coreset[k,...]])
                    dir_path = os.path.join(self.embedding_dir_path,f'faiss_patchcore_{k}')
                    os.makedirs(dir_path, exist_ok=True)
                    self.anomaly_scorer.save_and_reset(dir_path)
                if self.faiss_quantized:
                    # if False:
                    nlist = 20 if self.embedding_coreset.shape[0] > 20 else self.embedding_coreset.shape[0]
                    n_probe = 5 # defaul 1 # TODO
                    quantizer = faiss.IndexFlatL2(self.embedding_coreset.shape[2])
                    self.index = faiss.IndexIVFFlat(quantizer, self.embedding_coreset.shape[2], nlist, faiss.METRIC_L2)
                    assert not self.index.is_trained
                    self.index.train(self.embedding_coreset[k,...])
                    assert self.index.is_trained
                    self.index.add(self.embedding_coreset[k,...])
                    self.index.nprobe = n_probe
                    faiss.write_index(self.index,  os.path.join(self.embedding_dir_path,f'index_{k}.faiss'))
                elif self.faiss_standard:
                    self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[2])
                    self.index.add(self.embedding_coreset[k]) 
                    faiss.write_index(self.index,  os.path.join(self.embedding_dir_path,f'index_{k}.faiss'))
                else:
                    print(self.embedding_coreset.shape)
                    with open(os.path.join(self.embedding_dir_path, f'embedding_{k}.pickle'), 'wb') as f:
                        pickle.dump(self.embedding_coreset[k,...], f)
        
        # save model
        if not (self.quantize_qint8 or self.quantize_qint8_torchvision):
            torch.save(self.backbone, os.path.join(self.embedding_dir_path,'backbone.pth'))
        else:
            torch.save(self.backbone.state_dict(), os.path.join(self.embedding_dir_path,'backbone.pth'))
        # torch.save(self.backbone, os.path.join(self.embedding_dir_path,'backbone.pth'))
            
    def test_step(self, batch, batch_idx):
        '''
        required func that handles not just the step istelf, but also the measurements (inference times).
        '''
        if self.measure_inference:
            # initialize dict
            if self.cuda_active and torch.cuda.is_available():
                self.backbone.to(device='cuda')
            else:
                self.backbone.to(device='cpu')
            
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
            # warm up loop
            for _ in range(self.warm_up_reps):
                _, _, _, _, _ = self._test_step(batch=batch, measure=False)
            
            # actual measurements
            ################################################
            # LOOP
            for rep in range(self.number_of_reps):
                if self.cuda_active:
                    st_gpu, et_gpu = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True) # initialize cuda timers<
                    st_gpu = record_gpu(st_gpu)
                st_cpu = record_cpu()
                if rep+1 == self.number_of_reps:
                    features, embeddings, score_patches, score, anomaly_map, t_0_cpu, t_1_cpu, t_2_cpu, t_3_cpu, t_4_cpu, t_0_gpu, t_1_gpu, t_2_gpu, t_3_gpu, t_4_gpu = self._test_step(batch=batch, measure=True)
                else:
                    _, _, _, _, _, t_0_cpu, t_1_cpu, t_2_cpu, t_3_cpu, t_4_cpu, t_0_gpu, t_1_gpu, t_2_gpu, t_3_gpu, t_4_gpu = self._test_step(batch=batch, measure=True)
                et_cpu = record_cpu()
                # gpu
                if self.cuda_active:
                    et_gpu = record_gpu(et_gpu)
                    run_times['#2 feature extraction gpu'] += [t_0_gpu.elapsed_time(t_1_gpu)]
                    run_times['#4 embedding of features gpu'] += [t_1_gpu.elapsed_time(t_2_gpu)]
                    run_times['#6 score patches gpu'] += [t_2_gpu.elapsed_time(t_3_gpu)]
                    run_times['#8 img lvl score gpu'] += [t_3_gpu.elapsed_time(t_4_gpu)]
                    run_times['#10 anomaly map gpu'] += [t_4_gpu.elapsed_time(et_gpu)]
                    run_times['#12 whole process gpu'] += [st_gpu.elapsed_time(et_gpu)]
                    run_times['#14 dim reduction gpu'] += [0.0] # not used
                else:
                    run_times['#2 feature extraction gpu'] += [0.0]
                    run_times['#4 embedding of features gpu'] += [0.0]
                    run_times['#6 score patches gpu'] += [0.0]
                    run_times['#8 img lvl score gpu'] += [0.0]
                    run_times['#10 anomaly map gpu'] += [0.0]
                    run_times['#12 whole process gpu'] += [0.0]
                    run_times['#14 dim reduction gpu'] += [0.0] # not used
                # cpu
                run_times['#1 feature extraction cpu'] += [float((t_1_cpu - t_0_cpu) * 1e3)]
                run_times['#3 embedding of features cpu'] += [float((t_2_cpu - t_1_cpu) * 1e3)]
                run_times['#5 score patches cpu'] += [float((t_3_cpu - t_2_cpu) * 1e3)]
                run_times['#7 img lvl score cpu'] += [float((t_4_cpu - t_3_cpu) * 1e3)]
                run_times['#9 anomaly map cpu'] += [float((et_cpu - t_4_cpu) * 1e3)]
                run_times['#11 whole process cpu'] += [float((et_cpu - st_cpu) * 1e3)]
                run_times['#13 dim reduction cpu'] += [0.0] # not used

                # if multiple coresets are used, we have to correct the inference times
                if self.multiple_coresets[0] and self.coreset_sampling_ratio != 1.0:
                    copy_of_img_lvlv_score_cpu_time = run_times['#7 img lvl score cpu'][-1]
                    run_times['#7 img lvl score cpu'][-1] = run_times['#7 img lvl score cpu'][-1]/self.multiple_coresets[1]
                    copy_of_score_patch_cpu = run_times['#5 score patches cpu'][-1]
                    run_times['#5 score patches cpu'][-1] = run_times['#5 score patches cpu'][-1]/self.multiple_coresets[1]
                    diff = (copy_of_img_lvlv_score_cpu_time - run_times['#7 img lvl score cpu'][-1]) + (copy_of_score_patch_cpu - run_times['#5 score patches cpu'][-1])
                    run_times['#11 whole process cpu'][-1] = run_times['#11 whole process cpu'][-1] - diff
            # LOOP
            ################################################
           
            assert len(run_times['#1 feature extraction cpu']) == self.number_of_reps, "Something went wrong!"
            
            # calc mean of measurements
            for this_entry in run_times.items():
                if len(this_entry[1]) > 0:
                    run_times[this_entry[0]] = float((sum(this_entry[1]) / len(this_entry[1])) / batch[0].size()[0]) # mean
                else:
                    run_times[this_entry[0]] = 0.0
            
            # note args used for this run and add them to dict
            # TODO
            
            # save as csv using pandas dataframe
            pd_run_times = pd.DataFrame(run_times, index=[batch_idx])
            file_path = os.path.join(self.log_path, self.latences_filename)
            if os.path.exists(file_path):
                pd_run_times_ = pd.read_csv(file_path, index_col=0)
                pd_run_times = pd.concat([pd_run_times_, pd_run_times], axis=0)
                pd_run_times.to_csv(file_path)
            else:
                pd_run_times.to_csv(file_path)
        
        else:
            _, _, score_patches, score, anomaly_map = self._test_step(batch=batch, measure=False) # calculating of scores and saving of results
            # print(score)Fprune
        if type(score_patches) == list and not self.batch_size_test == 1:
            results = (score_patches, anomaly_map)
            x_batch, gt_batch, label_batch, file_name_batch, x_type_batch = batch
            for k in range(x_batch.size()[0]):
                this_score_patches, this_anomaly_map = results[0][k], results[1][k]
                x, gt, label, file_name, x_type = x_batch[k], gt_batch[k], label_batch[k], file_name_batch[k], x_type_batch[k]
                self.eval_one_step_test(score_patches=this_score_patches, score=score, anomaly_map=this_anomaly_map, x=x, gt=gt, label=label, file_name=file_name, x_type=x_type)
        else:
            x, gt, label, file_name, x_type = batch
            self.eval_one_step_test(score_patches, score, anomaly_map, x, gt, label, file_name, x_type)
    
    @torch.inference_mode()                    
    def _test_step(self, batch, measure=False):
        '''
        basically this is one test step where one batch is processed. This func is embedded in the actual def test_step. 
        '''
        if not measure:
            # print('not measuring')
            x, _, _, _, _ = batch
            batch_size_1 = (x.shape[0] == 1)
            batch_size = x.shape[0]
            # extract embedding
            features = self.feature_extraction(x=x)
            if self.pooling_embedding:
                embeddings = self.feature_embedding(features=features, batch_size_1=batch_size_1, batch_size=batch_size)
            else:
                embeddings = _embed(features, self.forward_modules, self.patch_maker)#, batch_size_1=batch_size_1, batch_size=batch_size)
                embeddings = embeddings.detach().cpu().numpy()
            
            # print('embeddings shape in test_step: ', embeddings.shape)
            if not self.multiple_coresets[0] or self.coreset_sampling_ratio == 1.0:
                score_patches = self.calc_score_patches(embeddings=embeddings, batch_size_1=batch_size_1)
                score = self.calc_img_score(score_patches=score_patches)
                if not self.only_img_lvl:
                    anomaly_map = calc_anomaly_map(score_patches=score_patches, batch_size_1=batch_size_1, load_size=self.load_size)
                else:
                    anomaly_map = None
            else:
                score_patches = self.calc_score_patches(embeddings=embeddings, batch_size_1=batch_size_1)
                score = [self.calc_img_score(score_patches=score_patches_) for score_patches_ in score_patches]
                    
                if not self.only_img_lvl:
                    raise NotImplementedError('multiple coresets not implemented for pixel wise anomaly map')
                else:
                    anomaly_map = None
                
            return features, embeddings, score_patches, score, anomaly_map 
            
        else:
            # print('measuring')
            
            # t_00 = record_cpu()
            ############################################################
            # INITIALIZE MEASUREMENT UTILS
            # initialize cuda events
            if self.cuda_active:
                t_0_gpu, t_1_gpu, t_2_gpu, t_3_gpu, t_4_gpu = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            else:
                t_0_gpu, t_1_gpu, t_2_gpu, t_3_gpu, t_4_gpu = None, None, None, None, None
            # INITIALIZE MEASUREMENT UTILS
            ############################################################
            
            ############################################################
            # FEATURE EXTRACTION
            t_0_cpu = record_cpu()
            if self.cuda_active:
                t_0_gpu = record_gpu(t_0_gpu)
            
            x, _, _, _, _ = batch
            batch_size_1 = (x.shape[0] == 1)
            batch_size = x.shape[0]
            
            features = self.feature_extraction(x=x)
            # FEATURE EXTRACTION
            ############################################################

            ############################################################            
            # FEATURE EMBEDDING
            t_1_cpu = record_cpu()
            if self.cuda_active:
                t_1_gpu = record_gpu(t_1_gpu)
            if self.pooling_embedding:
                embeddings = self.feature_embedding(features=features, batch_size_1=batch_size_1, batch_size=batch_size)
            else:
                embeddings = _embed(features, self.forward_modules, self.patch_maker)#, batch_size_1=batch_size_1, batch_size=batch_size)
                embeddings = embeddings.detach().cpu().numpy()
            # FEATURE EMBEDDING
            ############################################################
            
            ############################################################
            # NN SEARCH // SCORE PATCHES
            t_2_cpu = record_cpu()
            if self.cuda_active:
                t_2_gpu = record_gpu(t_2_gpu)
            
            # t_0 = record_cpu()
            score_patches = self.calc_score_patches(embeddings=embeddings, batch_size_1=batch_size_1)
            # t_1 = record_cpu()
            # print('score patches time: ', (t_0 - t_1)*1000)
            # NN SEARCH // SCORE PATCHES
            ############################################################
            
            ############################################################
            # IMG LEVEL SCORE
            t_3_cpu = record_cpu()
            
            # t_0 = record_cpu()
            if self.cuda_active:
                t_3_gpu = record_gpu(t_3_gpu)
            if not self.multiple_coresets[0] or self.coreset_sampling_ratio == 1.0:
                score = self.calc_img_score(score_patches=score_patches)
            else:
                score = [self.calc_img_score(score_patches=score_patches_) for score_patches_ in score_patches]
            # IMG LEVEL SCORE
            # t_1 = record_cpu()
            # print('img lvl score time: ', (t_0 - t_1)*1000)
            ############################################################
            
            ############################################################
            # AMOMALY MAP
            t_4_cpu = record_cpu()
            if self.cuda_active:
                t_4_gpu = record_gpu(t_4_gpu)
            
            if not self.only_img_lvl:
                if not self.multiple_coresets[0] or self.coreset_sampling_ratio == 1.0:
                    anomaly_map = self.calc_anomaly_map(score_patches=score_patches, batch_size_1=batch_size_1, load_size=self.load_size)
                else:
                    raise NotImplementedError('multiple coresets not implemented for pixel wise anomaly map')
            else:
                anomaly_map = None
            # ANOMALY MAP
            ############################################################
            # t_11 = record_cpu()
            # print('entire process time: ', (t_00 - t_11)*1000)
            return features, embeddings, score_patches, score, anomaly_map, t_0_cpu, t_1_cpu, t_2_cpu, t_3_cpu, t_4_cpu, t_0_gpu, t_1_gpu, t_2_gpu, t_3_gpu, t_4_gpu
        
    @torch.inference_mode()    
    def feature_extraction(self, x):
        '''
        Pass data through backbone specified in class pactchcore
        '''
        if self.cuda_active:
            x = x.cuda()
        # else:
        out = self.backbone(x)
        return out
        
    def feature_embedding(self, features, batch_size_1, batch_size):
        '''
        embedding of features extracted in previous step. Eventually integrates dim reduction and adaptive pooling. 
        '''
        
        if self.backbone_id.__contains__('pdn'): # no embedding necessary
            flattened_features = reshape_embedding(features.detach().cpu().numpy())
            if self.own_knn:
                flattened_features = torch.from_numpy(flattened_features)
            return flattened_features
        
        selected_features = []
        
        for _, feature in enumerate(features):
            ####
            # insert dim reduction here TODO 
            # before pooling
            ####
            # pooled_features = adaptive_pooling(feature, self.pooling_strategy)#torch.nn.AvgPool2d(3, 1, 1)(feature) # TODO replace with adaptive pooling
            if type(self.pooling_strategy) == list:
                for strategy in self.pooling_strategy:
                    pooled_feature = adaptive_pooling(feature, strategy)
                    selected_features.append(pooled_feature)
            else:
                pooled_feature = adaptive_pooling(feature, self.pooling_strategy)
                selected_features.append(pooled_feature)
            ####
            # insert dim reduction here TODO 
            # after pooling
            ####
            # selected_features.append(pooled_features)
        
        concatenated_features = embedding_concat_frame(embeddings=selected_features, cuda_active=self.cuda_active)
        
        # if self.pool_depth[0]:
            # print
        
        if batch_size_1:
            flattened_features = np.array(reshape_embedding(np.array(concatenated_features)))
        else:
            flattened_features = np.array([np.array(reshape_embedding(np.array(concatenated_features[k,...].unsqueeze(0)))) for k in range(batch_size)])
        
        if ((self.reduce_via_std or self.reduce_via_entropy or self.reduce_via_entropy_normed or self.reduce_via_real_entropy or self.reduce_via_random) and not self.prune_output_layer[0]) and self.idx_chosen is not None:
            flattened_features = np.take(flattened_features, self.idx_chosen, axis=1)#indices=#[:,self.idx_with_high_std]
        
        if self.normalize and self.mean is not None:
            flattened_features = (flattened_features - self.mean) / self.std
        
        if (self.weight_by_entropy or self.weight_by_real_entropy) and self.weights is not None:
            flattened_features = np.multiply(flattened_features, self.weights)
            
        if self.adapt_feature and self.feature_adaptor is not None:
            torch_features = torch.from_numpy(flattened_features).to(device='cuda:0' if self.cuda_active and torch.cuda.is_available() else 'cpu')
            flattened_features = self.feature_adaptor(torch_features).detach().cpu().numpy()
        if self.own_knn:
            flattened_features = torch.from_numpy(flattened_features)#.to(device='cuda:0' if self.cuda_active and torch.cuda.is_available() else 'cpu')
        
        if self.reduce_via_pca and self.pca is not None:
            flattened_features = self.pca.transform(flattened_features)
        
        if self.pool_depth:
            n_components = int(flattened_features.shape[1]*self.reduction_factor/100)
            pooler = torch.nn.AdaptiveAvgPool1d(n_components)
            flattened_features = pooler(torch.from_numpy(flattened_features)).cpu().numpy()
        
        return flattened_features

    def calc_score_patches(self, embeddings, batch_size_1):
        '''
        calc score_patches from which image score and anomaly map can be derived.
        '''
        # if not self.multiple_coresets[0] or self.coreset_sampling_ratio == 1.0:
        #     if batch_size_1:
        #         if self.faiss_quantized or self.faiss_standard:
        #             score_patches, _ = self.index.search(embeddings , k=self.n_neighbors)
        #         elif self.own_knn:
        #             score_patches = self.knn(embeddings)[0].cpu().detach().numpy() # .cuda()
        #         elif self.patchcore_score_patches:
        #             # embeddings = np.asarray(embeddings)
        #             idx = 0 if self.patchcore_scorer else 1
        #             score_patches = self.anomaly_scorer.predict([embeddings])[idx]
        #             if idx == 0:
        #                 score_patches = self.patch_maker.unpatch_scores(score_patches, batchsize=1) # identity function if batch_size=1 ... currently not used
        #         else:
        #             score_patches, _ = self.nbrs.kneighbors(embeddings)
                
        #     else:
        #         if self.patchcore_score_patches:
        #             raise NotImplementedError('not implemented yet (batch_size > 1)')
        #         elif self.faiss_quantized or self.faiss_standard:
        #             score_patches = [self.index.search(element, k=self.n_neighbors)[0] for element in embeddings] # TODO
        #         elif self.own_knn:
        #             # if self.pooling_embedding:
        #             # if not self.pooling_embedding:
        #             score_patches_ = self.knn(embeddings)[0].cpu().detach().numpy()
        #             # else:
        #             #     score_patches_ = self.knn(torch.from_numpy(embeddings))[0].cpu().detach().numpy()
        #         elif self.patchcore_score_patches:
        #             raise NotImplementedError('not implemented yet (batch_size > 1)')
        #         else:
        #             score_patches = [self.nbrs.kneighbors(element) for element in embeddings]
        #     # print('score_patches shape: ', score_patches.shape)
        #     return score_patches
        
        # else:
        #     if batch_size_1:
        #         score_patches = []
        #         for k in range(self.multiple_coresets[1]):
        #             # t_0 = record_cpu()
        #             if self.patchcore_score_patches:
        #                 idx = 0 if self.patchcore_scorer else 1
        #                 this_scorer = self.anomaly_scorer[k]
        #                 # print('this_scorer: ', this_scorer)
        #                 score_patches_ = this_scorer.predict([embeddings])[idx]
        #                 # print()
        #                 if idx == 0:
        #                     score_patches_ = self.patch_maker.unpatch_scores(score_patches_, batchsize=1)
        #             elif self.faiss_quantized or self.faiss_standard:
        #                 # print(embeddings.shape)
        #                 # print(type(embeddings))
        #                 score_patches_, _ = self.index[k].search(embeddings, k=self.n_neighbors)
        #             elif self.own_knn:
        #                 # if not self.pooling_embedding:
        #                 score_patches_ = self.knn[k](embeddings)[0].cpu().detach().numpy()
        #                 # else:
        #                     # score_patches_ = self.knn[k](torch.from_numpy(embeddings))[0].cpu().detach().numpy()
        #             # elif self.patchcore_score_patches:
        #             #     idx = 0 if self.patchcore_scorer else 1
        #             #     score_patches_ = self.anomaly_scorer.predict([embeddings])[idx]
        #             #     if idx == 0:
        #             #         score_patches_ = self.patch_maker.unpatch_scores(score_patches_, batchsize=1) # identity function if batch_size=1 ... currently not used
        #             #     # score_patches_ = self.patch_maker.unpatch_scores(score_patches_, batchsize=1) # identity function if batch_size=1 ... currently not used
        #             # t_1 = record_cpu()
        #             # print('time for one search: ', (t_1-t_0)*1000)
        #             score_patches += [score_patches_]
        #     else:
        #         raise NotImplementedError('multiple coresets not implemented for batch_size > 1')
        #     return score_patches
        if not self.multiple_coresets[0] or self.coreset_sampling_ratio == 1.0:
            if batch_size_1:
                if self.faiss_quantized or self.faiss_standard:
                    score_patches, _ = self.index.search(embeddings , k=self.n_neighbors)
                elif self.own_knn:
                    score_patches = self.knn(embeddings)[0].cpu().detach().numpy() # .cuda()
                elif self.patchcore_score_patches:
                    # embeddings = np.asarray(embeddings)
                    idx = 0 if self.patchcore_scorer else 1
                    score_patches = self.anomaly_scorer.predict([embeddings])[idx]
                    if idx == 0:
                        score_patches = self.patch_maker.unpatch_scores(score_patches, batchsize=1) # identity function if batch_size=1 ... currently not used
                else:
                    score_patches, _ = self.nbrs.kneighbors(embeddings)
                
            else:
                if self.patchcore_score_patches:
                    raise NotImplementedError('not implemented yet (batch_size > 1)')
                elif self.faiss_quantized or self.faiss_standard:
                    score_patches = [self.index.search(element, k=self.n_neighbors)[0] for element in embeddings] # TODO
                elif self.own_knn:
                    # if self.pooling_embedding:
                    # if not self.pooling_embedding:
                    score_patches_ = self.knn(embeddings)[0].cpu().detach().numpy()
                    # else:
                    #     score_patches_ = self.knn(torch.from_numpy(embeddings))[0].cpu().detach().numpy()
                elif self.patchcore_score_patches:
                    raise NotImplementedError('not implemented yet (batch_size > 1)')
                else:
                    score_patches = [self.nbrs.kneighbors(element) for element in embeddings]
            # print('score_patches shape: ', score_patches.shape)
            return score_patches
        
        else:
            if batch_size_1:
                score_patches = []
                for k in range(self.multiple_coresets[1]):
                    # t_0 = record_cpu()
                    if self.patchcore_score_patches:
                        idx = 0 if self.patchcore_scorer else 1
                        this_scorer = self.anomaly_scorer[k]
                        # print('this_scorer: ', this_scorer)
                        score_patches_ = this_scorer.predict([embeddings])[idx]
                        # print()
                        if idx == 0:
                            score_patches_ = self.patch_maker.unpatch_scores(score_patches_, batchsize=1)
                    elif self.faiss_quantized or self.faiss_standard:
                        # print(embeddings.shape)
                        # print(type(embeddings))
                        score_patches_, _ = self.index[k].search(embeddings, k=self.n_neighbors)
                    elif self.own_knn:
                        # if not self.pooling_embedding:
                        score_patches_ = self.knn[k](embeddings)[0].cpu().detach().numpy()
                        # else:
                            # score_patches_ = self.knn[k](torch.from_numpy(embeddings))[0].cpu().detach().numpy()
                    # elif self.patchcore_score_patches:
                    #     idx = 0 if self.patchcore_scorer else 1
                    #     score_patches_ = self.anomaly_scorer.predict([embeddings])[idx]
                    #     if idx == 0:
                    #         score_patches_ = self.patch_maker.unpatch_scores(score_patches_, batchsize=1) # identity function if batch_size=1 ... currently not used
                    #     # score_patches_ = self.patch_maker.unpatch_scores(score_patches_, batchsize=1) # identity function if batch_size=1 ... currently not used
                    # t_1 = record_cpu()
                    # print('time for one search: ', (t_1-t_0)*1000)
                    score_patches += [score_patches_]
            else:
                raise NotImplementedError('multiple coresets not implemented for batch_size > 1')
            return score_patches
    def calc_img_score(self, score_patches):
        '''
        calculates the image score based on score_patches
        '''
        # print('score_patches in calc_img_score: ', score_patches.shape)
        if self.adapted_score_calc:
            score = modified_kNN_score_calc(score_patches=score_patches.astype(np.float64), n_next_patches=self.n_next_patches)
        elif self.patchcore_scorer:
            score_patches = score_patches.reshape(*score_patches.shape[:2], -1)
            # print(score_patches.shape)
            score = self.patch_maker.score(score_patches)[0]

        else:
            if True: # outlier removal
                sum_of_each_patch = np.sum(score_patches,axis=1)
                threshold_val = 50*np.percentile(sum_of_each_patch, 50)
                non_outlier_patches = np.argwhere(sum_of_each_patch < threshold_val).flatten()#[0]
                if len(non_outlier_patches) < score_patches.shape[0]:
                    score_patches = score_patches[non_outlier_patches]
                    print('deleted outliers: ', sum_of_each_patch.shape[0]-len(non_outlier_patches))
            N_b = score_patches[np.argmax(score_patches[:,0])].astype(np.float128) # only the closest val is relevant for selection! # this changes with adapted version.
            w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
            score = w*max(score_patches[:,0]) # Image-level score #TODO --> meaning of numbers
        # print(score.shape)
        # if 
        return score
    
    def eval_one_step_test(self, score_patches, score, anomaly_map, x, gt, label, file_name, x_type):
        '''
        Extracted evaluation of single output
        '''
        if x.dim() != 4:
            x, gt, label = x.unsqueeze(0), gt.unsqueeze(0), label.unsqueeze(0)
        
        if not self.only_img_lvl:
            gt_np = gt.cpu().numpy()[0,0].astype(int)
            self.gt_list_px_lvl.extend(gt_np.ravel()) # ravel equivalent reshape(-1); flattening of ground_truth pixel wise
            self.pred_list_px_lvl.extend(anomaly_map.ravel()) # flattening of pred pixel wise
        self.gt_list_img_lvl.append(label.cpu().numpy()[0]) # ground_truth for image wise
        if not self.multiple_coresets[0] or self.coreset_sampling_ratio == 1.0:
            self.pred_list_img_lvl.append(score) # image level score appended
        else:
            self.pred_list_img_lvl.append([score])
        self.img_path_list.extend(file_name) # same for file_name
        # save images
        if self.save_am:
            x = self.inv_normalize(x) # inverse transformation of img
            if x.dtype != torch.float32:
                x = x.to(torch.float32)
            input_x = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB) # further transformation
            self.save_anomaly_map(anomaly_map, input_x, gt_np*255, file_name[0], x_type[0]) # save of everything
        
    # def test_epoch_end(self, outputs):
    def on_test_epoch_end(self):
        
        if not self.only_img_lvl:
            print("Total pixel-level auc-roc score :")
            # print(self.gt_list_px_lvl)
            pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
            print(pixel_auc)
        else:
            pixel_auc = 0.0
        print("Total image-level auc-roc score :")
        if not self.multiple_coresets[0] or self.coreset_sampling_ratio == 1.0:
            img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        else:
            pred_list_img_lvl_np = np.array(self.pred_list_img_lvl)
            # print('pred_list_img_lvl_np shape: ', pred_list_img_lvl_np.shape)
            img_auc_list = [roc_auc_score(self.gt_list_img_lvl, list(pred_list_img_lvl_np[...,k].flatten())) for k in range(self.multiple_coresets[1])]
            if self.calc_uncertainty:
                this_file_name = f'{self.backbone_id}_{self.layers_needed}_{self.category}.npy'
                with open(os.path.join('uncertainity', this_file_name), 'wb') as f:
                    np.save(f, np.array(img_auc_list))
            
            print('img_auc_list: ', img_auc_list)
            img_auc = np.mean(img_auc_list)
        print(img_auc)
        print('test_epoch_end')
        # values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
        # self.log_dict(values) # consumes a lot of storage!
        # own logging
        self.feature_adaptor = None
        self.pca = None
        
        if self.measure_inference:
            file_path = os.path.join(self.log_path, self.latences_filename)
            pd_run_times_ = pd.read_csv(file_path, index_col=0)
            pd_results = pd.DataFrame({'img_auc': [img_auc]*pd_run_times_.shape[0], 'pixel_auc': [pixel_auc]*pd_run_times_.shape[0]})
            pd_run_times = pd.concat([pd_run_times_, pd_results], axis=1)
            pd_run_times.to_csv(file_path)
            print(f'\n\nMEAN INFERENCE TIME: {pd_run_times["#11 whole process cpu"].mean()} ms\n')
        if True:
            # get backbone stats
            try:
                device = next(self.backbone.parameters()).device
                summary_of_backbone = summary(self.backbone, (1, 3, self.load_size, self.load_size), verbose = 0, device=device)
                estimated_total_size = (summary_of_backbone.total_input + summary_of_backbone.total_output_bytes + summary_of_backbone.total_param_bytes) / 1e6 # in MB
                number_of_mult_adds = summary_of_backbone.total_mult_adds / 1e6 # in M
            except:
                estimated_total_size = 0.0
                number_of_mult_adds = 0.0
            try:
                opt_dict = {
                    'backbone': self.backbone_id,
                    'pooling_strategy': str(self.pooling_strategy),
                    'layers_needed': self.layers_needed,
                    'layer_cut': self.layer_cut,
                    'exclude_relu': self.exclude_relu,
                    'sigmoid_in_last_layer': self.sigmoid_in_last_layer,
                    'prune_output_layer': f'{self.prune_output_layer[0]} #{len(self.prune_output_layer[1])}',
                    'prune_structured_nni': f'{self.prune_structured_nni[0]} (Percentage: {self.sparsity}; Method: {self.prune_structured_nni[2]})',
                    'prune_l1_unstructured': f'{self.prune_l1_unstructured[0]} (Percentage: {self.prune_l1_unstructured[1]})',
                    'prune_pytorch_pruning': f'{self.prune_torch_pruning[0]} (Percentage: {self.prune_torch_pruning[1]})',
                    'iterative_pruning': f'{self.iterative_pruning[0]} (Iterations: {self.iterative_pruning[1]})',
                    'pretrain_for_channel_selection': self.pretrain_for_channel_selection, # TODO
                    'adapted_score_calc': self.adapted_score_calc,
                    'n_neighbors': self.n_neighbors,
                    'n_next_patches': self.n_next_patches,
                    'coreset_sampling_ratio': self.coreset_sampling_ratio,
                    'reduce_via_std': self.reduce_via_std,
                    'reduce_via_entropy': self.reduce_via_entropy,
                    'quantize_model_with_nni': self.quantize_model_with_nni,
                    'reduce_via_entropy_normed': self.reduce_via_entropy_normed,
                    'reduce_via_real_entropy': self.reduce_via_real_entropy,
                    'reduce_via_random': self.reduce_via_random,
                    'reduce_factor': self.reduction_factor,
                    'reduce_via_pca': self.reduce_via_pca,
                    'weight_by_entropy': self.weight_by_entropy,
                    'weight_by_real_entropy': self.weight_by_real_entropy,
                    'reduce_by_height': self.reduce_by_heigth,
                    'coreset_size': self.embedding_coreset.shape[0] if not self.multiple_coresets[0] else self.embedding_coreset.shape[1],
                    'resulting_feature_length': self.embedding_coreset.shape[1] if not self.multiple_coresets[0] else self.embedding_coreset.shape[2],
                    'resolution_of_patches': np.sqrt(self.output_shape[1]),
                    'normalize_output': self.normalize,
                    'faiss_standard': self.faiss_standard,
                    'faiss_quantized': self.faiss_quantized,
                    'own_knn': self.own_knn,
                    'distance_metric': self.metrices[self.metric_id],
                    'multiple_coresets': 1 if not self.multiple_coresets[0] else self.multiple_coresets[1],
                    'feature_adaptor': self.adapt_feature,
                    'own_qint8': self.quantize_qint8,
                    'torchvision_qint8': self.quantize_qint8_torchvision,
                    'pooling_strategy': str(self.pooling_strategy),
                    'category_wise_statistics': self.category_wise_statistics,
                    'backbone_storage_[MB]': estimated_total_size,
                    'backbone_mult_adds_[M]': number_of_mult_adds,
                    'feature_extraction_[ms]': pd_run_times['#1 feature extraction cpu'].mean() if self.measure_inference else 0.0,
                    'embedding_of_features_[ms]': pd_run_times['#3 embedding of features cpu'].mean() if self.measure_inference else 0.0,
                    'calc_distances_[ms]': pd_run_times['#5 score patches cpu'].mean() if self.measure_inference else 0.0,
                    'calc_scores_[ms]': pd_run_times['#7 img lvl score cpu'].mean() if self.measure_inference else 0.0,
                    'total_time_[ms]': pd_run_times['#11 whole process cpu'].mean() if self.measure_inference else 0.0,
                    'img_auc_[%]': img_auc
                    }
            except:
                print('Full opt_dict not available. Probably because only inference and therefore some parameters are not available.')
                opt_dict = {
                    'backbone': self.backbone_id,
                    'layers_needed': self.layers_needed,
                    'layer_cut': self.layer_cut,
                    'adapted_score_calc': self.adapted_score_calc,
                    'faiss_standard': self.faiss_standard,
                    'faiss_quantized': self.faiss_quantized,
                    'own_knn': self.own_knn,
                    'feature_adaptor': self.adapt_feature,
                    'own_qint8': self.quantize_qint8,
                    'torchvision_qint8': self.quantize_qint8_torchvision,
                    'feature_extraction_[ms]': pd_run_times['#1 feature extraction cpu'].mean() if self.measure_inference else 0.0,
                    'embedding_of_features_[ms]': pd_run_times['#3 embedding of features cpu'].mean() if self.measure_inference else 0.0,
                    'calc_distances_[ms]': pd_run_times['#5 score patches cpu'].mean() if self.measure_inference else 0.0,
                    'calc_scores_[ms]': pd_run_times['#7 img lvl score cpu'].mean() if self.measure_inference else 0.0,
                    'total_time_[ms]': pd_run_times['#11 whole process cpu'].mean() if self.measure_inference else 0.0,
                    'img_auc_[%]': img_auc
                    }
                    
            file_path = os.path.join(self.log_path, f'summary_{self.group_id}.csv')
            
            if os.path.exists(file_path):
                pd_sum = pd.read_csv(file_path, index_col=0)
                pd_sum_current = pd.Series(opt_dict).to_frame(self.category)#, index='category')
                pd_sum = pd.concat([pd_sum, pd_sum_current], axis=1)
            else:
                # pd_sum = pd.DataFrame({'category': self.category,'img_acc': img_auc, 'adapted_score_calc': str(self.adapted_score_calc), 'pooling_strategy': str(self.pooling_strategy)}, index='category')
                pd_sum = pd.Series(opt_dict).to_frame(self.category)
            pd_sum.to_csv(file_path)
            self.idx_chosen = None
            
            
def one_run_of_model(model, train_and_test = True):
    '''
    Executes one run of the model. All parameters are set in the model class.
    '''
    if train_and_test:
        trainer = pl.Trainer(max_epochs=1, accelerator='gpu' if model.cuda_active_training and not (model.quantize_qint8 or model.quantize_qint8_torchvision) else 'cpu', inference_mode=True, enable_model_summary=False)
        trainer.fit(model)
        trainer = pl.Trainer(max_epochs=1, accelerator='gpu' if model.cuda_active and not model.quantize_qint8 else 'cpu', inference_mode=True, enable_model_summary=True)
        trainer.test(model)
    else:
        trainer = pl.Trainer(max_epochs=1, accelerator='gpu' if model.cuda_active and not model.quantize_qint8 else 'cpu', inference_mode=True, enable_model_summary=True)
        trainer.test(model)
    

if __name__ == '__main__':

    print('start')
    
    import warnings
    warnings.filterwarnings("ignore") 
    
    from utils.testbench_utils import get_default_PatchCoreModel
    model = get_default_PatchCoreModel()
    
    one_run_of_model(model, True)
    