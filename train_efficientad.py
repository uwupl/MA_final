#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import randoalm
from tqdm import tqdm
from time import perf_counter
import json
import platform
import datetime
from sklearn.metrics import roc_auc_score
import sys

from utils.common import get_autoencoder, get_pdn_small, get_pdn_medium, get_autoencoder_own, get_pdn_own, get_pdn_own_2,\
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader

class config_helper():
    '''
    Class that holds all the config parameters and summarized training and testing results. 
    '''
    def __init__(self):
        self.run_id = 'not_specified'
        self.dataset = 'mvtec_ad'
        self.subdataset = 'zipper'
        root_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = root_dir + '/results/efficientned_ad'
        self.model_size = 'own' # --> XS
        self.weights = ''# gets overwritten anyway /mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_complete/output/pretraining/1699000941/teacher_own_tmp_state.pth'#root_dir + '/efficient_net/models/teacher_small.pth'
        self.imagenet_train_path = 'none'
        self.mvtec_ad_path = '/mnt/crucial/UNI/IIIT_Muen/MA/MVTechAD' # PATHS HAVE TO BE ADJUSTED
        self.mvtec_loco_path = './mvtec_loco_anomaly_detection' # not used
        self.train_steps = 50000
        self.test_interval = self.train_steps // 10
        self.seed = 42
        self.on_gpu = torch.cuda.is_available()
        self.on_gpu_init = self.on_gpu#.copy()
        self.out_channels = 384
        self.image_size = 256
        self.test_batch_size = 1
        self.train_batch_size = 1
        self.num_workers = 12 if platform.machine().__contains__('x86') else 4
        self.adapted_score_calc = False
        self.measure_inference_time = False
        self.save_anomaly_map = False
        self.auc_q_best = 0.0
        self.auc_best = 0.0
        self.auc_q_best_at = 0
        self.auc_best_at = 0
        self.auc_final = 0.0
        self.auc_q_final = 0.0
        self.teacher_inference = 0.0
        self.student_inference = 0.0
        self.autoencoder_inference = 0.0
        self.map_normalization_inference = 0.0
        self.datetime = ''
        self.backend = 'x86'
        self.model_base_dir = None
        
        self.run_id = input('Please enter a unique run id:\n')
        raspberry_pi = False if platform.machine().__contains__('x86') else True
            # config = config_helper()
        if raspberry_pi:
            self.output_dir = '/home/jo/MA/code/MA_complete/results/'
            self.weights = '/home/jo/MA/code/MA_complete/efficient_net/models/teacher_small.pth'
            self.mvtec_ad_path = '/home/jo/MA/MVTechAD'
            self.model_base_dir = '/home/jo/MA/code/MA_complete/quantized_models'
            self.backend = 'qnnpack'
        else:
            
            self.output_dir = root_dir + '/results/efficient_ad' # TODO add config.subdataset
            self.weights = root_dir + '/teacher_models/teacher_Xsmall.pth'
            # self.weights = root_dir + '/output/pretraining/1699000941/teacher_own_tmp_state.pth'
            # self.weights = root_dir + '/output/pretraining/1699279527/teacher_own_2_1699285493_tmp_state.pth'
            self.mvtec_ad_path = '/mnt/crucial/UNI/IIIT_Muen/MA/MVTechAD'
            self.model_base_dir = root_dir + f'/results/efficient_ad/{self.run_id}/models' # TODO: add config.subdataset
            self.backend = 'fbgemm'#x86'
        
    def save_as_json(self):
        file_name = f'config_{self.run_id}_{self.subdataset}.json'
        self.datetime = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        path = os.path.join(self.output_dir, self.run_id, 'config', file_name)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f)
            
    def reset(self):
        self.auc_q_best = 0.0
        self.auc_best = 0.0
        self.auc_q_best_at = 0
        self.auc_best_at = 0
        self.auc_final = 0.0
        self.auc_q_final = 0.0
        self.teacher_inference = 0.0
        self.student_inference = 0.0
        self.autoencoder_inference = 0.0
        self.map_normalization_inference = 0.0
        self.datetime = ''
        

# constants
# created here at the top level to ensure config is available in all functions
# if __name__ == '__main__':
config = config_helper()

already_done = []
# data loading
default_transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

def main():
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    else:
        raise Exception('Unknown config.dataset')

    cats =['bottle', 'cable', 'capsule', 'carpet', 'grid', 'own',
          'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
            'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    # cats = ['screw', 'leather', 'carpet', 'pill', 'capsule']
    additional_base_points = [0, 20, 100, 500, 1000, 2000, 3000, 4000, 7500]
    
    for k, cat in enumerate(cats):
        if cat in already_done:
            continue
        t_0 = perf_counter()
        print(f'\nProcessing {cat}: {k+1}/{len(cats)}\n ')
        
        config.subdataset = cat
        config.reset()
        # helper variables for tracking the traing progress
        y_loss_ae = []
        x_loss_ae = []
        y_loss_st = []
        x_loss_st = []
        y_loss_stae = []
        x_loss_stae = []
        y_loss_total = []
        x_loss_total = []
        y_auc = []
        x_auc = []
        y_auc_q = []
        x_auc_q = []
        best_auc = (0.0, 0) 
        best_auc_q = (0.0, 0)
        
        pretrain_penalty = True
        if config.imagenet_train_path == 'none':
            pretrain_penalty = False

        # create output dir
        train_output_dir = os.path.join(config.output_dir, config.run_id)# this is the directory that contains all the results 
        if config.save_anomaly_map: # usually not set so the following is not up to date
            test_output_dir = os.path.join(config.output_dir, 'anomaly_maps',
                                        config.dataset, config.subdataset, 'test')
        if not os.path.exists(train_output_dir):
            os.makedirs(train_output_dir)
            os.makedirs(os.path.join(train_output_dir, 'models'))
            os.makedirs(os.path.join(train_output_dir, 'train_progress'))
            os.makedirs(os.path.join(train_output_dir, 'config'))
            os.makedirs(os.path.join(train_output_dir, 'model_statistics'))
            
        if config.save_anomaly_map:
            if not os.path.exists(test_output_dir):
                os.makedirs(test_output_dir)

        # load data
        full_train_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, config.subdataset, 'train'),
            transform=transforms.Lambda(train_transform))
        test_set = ImageFolderWithPath(
            os.path.join(dataset_path, config.subdataset, 'test'))
        if config.dataset == 'mvtec_ad':
            # mvtec dataset paper recommend 10% validation set
            train_size = int(0.9 * len(full_train_set))
            validation_size = len(full_train_set) - train_size
            rng = torch.Generator().manual_seed(config.seed) # random number generator
            train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                            [train_size,
                                                                validation_size],
                                                            rng)
        elif config.dataset == 'mvtec_loco':
            train_set = full_train_set
            validation_set = ImageFolderWithoutTarget(
                os.path.join(dataset_path, config.subdataset, 'validation'),
                transform=transforms.Lambda(train_transform))
        else:
            raise Exception('Unknown config.dataset')


        train_loader = DataLoader(train_set, batch_size=config.train_batch_size, shuffle=True,
                                num_workers=config.num_workers, pin_memory=True)
        train_loader_infinite = InfiniteDataloader(train_loader)
        validation_loader = DataLoader(validation_set, batch_size=config.train_batch_size,)

        if pretrain_penalty:
            # load pretraining data for penalty
            penalty_transform = transforms.Compose([
                transforms.Resize((2 * config.image_size, 2 * config.image_size)),
                transforms.RandomGrayscale(0.3),
                transforms.CenterCrop(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                    0.225])
            ])
            penalty_set = ImageFolderWithoutTarget(config.imagenet_train_path,
                                                transform=penalty_transform)
            penalty_loader = DataLoader(penalty_set, batch_size=config.train_batch_size, shuffle=True,
                                        num_workers=config.num_workers, pin_memory=True)
            penalty_loader_infinite = InfiniteDataloader(penalty_loader)
        else:
            penalty_loader_infinite = itertools.repeat(None)

        # create models
        if config.model_size == 'small':
            teacher = get_pdn_small(config.out_channels)
            student = get_pdn_small(2 * config.out_channels)
        elif config.model_size == 'medium':
            teacher = get_pdn_medium(config.out_channels)
            student = get_pdn_medium(2 * config.out_channels)
        elif config.model_size == 'own':
            teacher = get_pdn_own(config.out_channels)
            student = get_pdn_own(2 * config.out_channels)
        elif config.model_size == 'own_2':
            teacher = get_pdn_own_2(config.out_channels)
            student = get_pdn_own_2(2 * config.out_channels)
        else:
            raise Exception('Unknown config.model_size')
        if config.model_size != 'own':
            state_dict = torch.load(config.weights, map_location='cpu')
            teacher.load_state_dict(state_dict)
        else:
            teacher = torch.load(config.weights, map_location='cpu')
        if not (config.model_size == 'own' or config.model_size == 'own_2'):
            autoencoder = get_autoencoder(config.out_channels)
        else:
            autoencoder = get_autoencoder_own(config.out_channels)

        # teacher frozen
        teacher.eval()
        student.train()
        autoencoder.train()

        if config.on_gpu:
            teacher.cuda()
            student.cuda()
            autoencoder.cuda()

        teacher_mean, teacher_std = teacher_normalization(teacher, train_loader) # TODO: check what exactly is done here

        optimizer = torch.optim.Adam(itertools.chain(student.parameters(),
                                                    autoencoder.parameters()),
                                    lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1)
        tqdm_obj = tqdm(range(config.train_steps))
        for iteration, (image_st, image_ae), image_penalty in zip(
                tqdm_obj, train_loader_infinite, penalty_loader_infinite):
            if config.on_gpu_init:
                image_st = image_st.cuda()
                image_ae = image_ae.cuda()
                if image_penalty is not None:
                    image_penalty = image_penalty.cuda()
                teacher = teacher.cuda()
                student = student.cuda()
                autoencoder = autoencoder.cuda()    
                
            # compute loss
            with torch.no_grad():
                teacher_output_st = teacher(image_st)
                teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std # normalized output of teacher --> TODO: What is the teacher?
            student_output_st = student(image_st)[:, :config.out_channels] # take the first half of channels
            distance_st = (teacher_output_st - student_output_st) ** 2 # compute the distance between teacher and student
            d_hard = torch.quantile(distance_st, q=0.999) # compute threshold. This is done in order to avoid the model to learn insignificant differences
            loss_hard = torch.mean(distance_st[distance_st >= d_hard]) # take only the values above the threshold and compute the mean

            if image_penalty is not None:
                student_output_penalty = student(image_penalty)[:, :config.out_channels]
                loss_penalty = torch.mean(student_output_penalty**2)
                loss_st = loss_hard + loss_penalty
            else:
                loss_st = loss_hard

            ae_output = autoencoder(image_ae) 
            with torch.no_grad():
                teacher_output_ae = teacher(image_ae)
                teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
            student_output_ae = student(image_ae)[:, config.out_channels:] # take the second half of channels
            distance_ae = (teacher_output_ae - ae_output)**2 # compute the distance between 
            distance_stae = (ae_output - student_output_ae)**2
            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)
            loss_total = loss_st + loss_ae + loss_stae
            # optimizer step
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            if iteration == 0:
                # set buffer
                loss_ae_buffer = []
                loss_st_buffer = []
                loss_stae_buffer = []
                loss_total_buffer = []
            
                
            if iteration % 10 == 0:
                tqdm_obj.set_description(
                    "Current loss: {:.4f}  ".format(loss_total.item()))
            
            loss_ae_buffer.append(loss_ae.item())
            loss_st_buffer.append(loss_st.item())
            loss_stae_buffer.append(loss_stae.item())
            loss_total_buffer.append(loss_total.item())
                
            if iteration % 100 == 0:
                y_loss_ae.append(np.mean(loss_ae_buffer))
                x_loss_ae.append(iteration)
                y_loss_st.append(np.mean(loss_st_buffer))
                x_loss_st.append(iteration)
                y_loss_stae.append(np.mean(loss_stae_buffer))
                x_loss_stae.append(iteration)
                y_loss_total.append(np.mean(loss_total_buffer))
                x_loss_total.append(iteration)
                # reset buffer
                loss_ae_buffer = []
                loss_st_buffer = []
                loss_stae_buffer = []
                loss_total_buffer = []
            
            
            # intermediate evaluation
            if iteration % config.test_interval == 0 or iteration in additional_base_points:
                auc, auc_q = calibrate_eval_save(teacher, student, autoencoder, 
                                    teacher_mean, teacher_std, 
                                    train_loader, validation_loader, test_set, 
                                    train_output_dir, iteration, '')
                y_auc.append(auc)
                x_auc.append(iteration)
                y_auc_q.append(auc_q)
                x_auc_q.append(iteration)
                
                if auc > best_auc[0]:
                    best_auc = (auc, iteration)
                if auc_q > best_auc_q[0]:
                    best_auc_q = (auc_q, iteration)
                
        # final evaluation
        iteration = config.train_steps
        auc, auc_q = calibrate_eval_save(teacher, student, autoencoder, 
                            teacher_mean, teacher_std, 
                            train_loader, validation_loader, test_set, 
                            train_output_dir, iteration=iteration, phase = 'final')
        y_auc.append(auc)
        x_auc.append(iteration)
        y_auc_q.append(auc_q)
        x_auc_q.append(iteration)
        
        # save training progress as json
        with open(os.path.join(train_output_dir, 'train_progress', f'train_progress_{config.subdataset}.json'), 'w') as f:
            json.dump({'y_loss_ae': y_loss_ae, 'x_loss_ae': x_loss_ae,
                        'y_loss_st': y_loss_st, 'x_loss_st': x_loss_st,
                        'y_loss_stae': y_loss_stae, 'x_loss_stae': x_loss_stae,
                        'y_loss_total': y_loss_total, 'x_loss_total': x_loss_total,
                        'y_auc': y_auc, 'x_auc': x_auc,
                        'y_auc_q': y_auc_q, 'x_auc_q': x_auc_q}, f)
        
        if auc > best_auc[0]:
            best_auc = (auc, iteration)
        if auc_q > best_auc_q[0]:
            best_auc_q = (auc_q, iteration)
            
        # update config
        config.auc_final = auc
        config.auc_q_final = auc_q
        config.auc_best = best_auc[0]
        config.auc_q_best = best_auc_q[0]
        config.auc_best_at = best_auc[1]
        config.auc_q_best_at = best_auc_q[1]
        # save config as json
        config.save_as_json()
          
        t_1 = perf_counter()
        print(f'Time taken: {round((t_1 - t_0)/60, 2)} min')

@torch.no_grad()
def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         desc='Running inference', q_flag=False):
    y_true = []
    y_score = []
    if config.measure_inference_time:
        teacher_inference_times = []
        student_inference_times = []
        autoencoder_inference_times = []
        map_normalization_inference_times = []
    if not q_flag:
        on_gpu = True if next(student.parameters()).is_cuda else False
    else:
        on_gpu = False
    for image, _, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        if on_gpu:
            image = image.cuda()
        if config.measure_inference_time:
            map_combined, _, _, teacher_inference, student_inference, autoencoder_inference, map_normalization_inference = predict(
                image=image, teacher=teacher, student=student,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
                q_ae_start=q_ae_start, q_ae_end=q_ae_end,
                measure_inference_time=config.measure_inference_time)
            teacher_inference_times.append(teacher_inference)
            student_inference_times.append(student_inference)
            autoencoder_inference_times.append(autoencoder_inference)
            map_normalization_inference_times.append(map_normalization_inference)
        else:
            map_combined, _, _ = predict(
                image=image, teacher=teacher, student=student,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
                q_ae_start=q_ae_start, q_ae_end=q_ae_end,
                measure_inference_time=config.measure_inference_time)
        if config.save_anomaly_map:
            map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
            map_combined = torch.nn.functional.interpolate(
                map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir is not None and config.save_anomaly_map:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
            tifffile.imwrite(file, map_combined)

        y_true_image = 0 if defect_class == 'good' else 1
        
        # y_score_image = np.max(map_combined) # TODO weighted average, not just max
        if config.adapted_score_calc:
            highest_scores = np.sort(map_combined.flatten())[-int(0.01*len(map_combined.flatten()))]
            weights = np.arange(1, len(highest_scores)+1)
            y_score_image = np.average(highest_scores, weights=weights)
        else:
            y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    if config.measure_inference_time:
        teacher_inference_mean = np.mean(teacher_inference_times)
        student_inference_mean = np.mean(student_inference_times)
        autoencoder_inference_mean = np.mean(autoencoder_inference_times)
        map_normalization_inference_mean = np.mean(map_normalization_inference_times)
    if config.measure_inference_time:
        return auc * 100, teacher_inference_mean, student_inference_mean, autoencoder_inference_mean, map_normalization_inference_mean
    else:
        return auc * 100

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None, measure_inference_time=False):
    
    if measure_inference_time:
        t_0 = perf_counter()
    teacher_output = teacher(image)
    device = teacher_output.device
    teacher_mean, teacher_std = teacher_mean.to(device), teacher_std.to(device)
    q_st_start = q_st_start.to(device) if q_st_start is not None else None
    q_st_end = q_st_end.to(device) if q_st_end is not None else None
    q_ae_start = q_ae_start.to(device) if q_ae_start is not None else None
    q_ae_end = q_ae_end.to(device) if q_ae_end is not None else None
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    if measure_inference_time:  
        t_1 = perf_counter()
    student_output = student(image)
    if measure_inference_time:
        t_2 = perf_counter()
    autoencoder_output = autoencoder(image)
    if measure_inference_time:
        t_3 = perf_counter()
    map_st = torch.mean((teacher_output - student_output[:, :config.out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, config.out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    if measure_inference_time:
        t_4 = perf_counter()
    if measure_inference_time:
        return map_combined, map_st, map_ae, t_1 - t_0, t_2 - t_1, t_3 - t_2, t_4 - t_3
    else:
        return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization', q_flag=False):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    if not q_flag:
        on_gpu = True if next(student.parameters()).is_cuda else False
    else:
        on_gpu = False
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        _, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader, q_flag=False):

    mean_outputs = []
    if not q_flag:
        on_gpu = True if next(teacher.parameters()).is_cuda else False
    else:
        on_gpu = False
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std


### Quantization section ###

class RandomImageDataset(torch.utils.data.Dataset):
    def __init__(self, num_images, transform=None, image_size=224):
        self.num_images = num_images
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Generate a random image
        image = np.random.randint(0, 256, size=(self.image_size, self.image_size, 3), dtype=np.uint8)

        # Convert numpy array to PIL image
        image = transforms.ToPILImage()(image)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, 0, 0, 0, 0

def quantize_model(teacher, student, autoencoder, calibration_loader=None, backend='fbgemm'):
    import torch.quantization as tq
    import torch.ao.quantization as taoq
    
    fuse_list_teacher_student = [('0','1'),('3','4'),('6','7')]
    fuse_list_autoencoder = [('0','1'),('2','3'),('4','5'),('6','7'),('8','9'),('12','13'),('16','17'),('20','21'),('24','25'),('28','29'),('32','33'),('36','37')]
    
    teacher, student, autoencoder = teacher.to('cpu'), student.to('cpu'), autoencoder.to('cpu')
    
    teacher, student, autoencoder = tq.fuse_modules(teacher, fuse_list_teacher_student), tq.fuse_modules(student, fuse_list_teacher_student), tq.fuse_modules(autoencoder, fuse_list_autoencoder)
    
    teacher, student, autoencoder = taoq.QuantWrapper(teacher), taoq.QuantWrapper(student), taoq.QuantWrapper(autoencoder)
    
    teacher.qconfig = tq.get_default_qconfig(backend)
    student.qconfig = tq.get_default_qconfig(backend)
    autoencoder.qconfig = tq.get_default_qconfig(backend)
    
    tq.prepare(teacher, inplace=True)
    tq.prepare(student, inplace=True)
    tq.prepare(autoencoder, inplace=True)
    
    if calibration_loader is not None:
        def calibrate_model(model, loader, calib_item_idx=[0]):
            for idx in calib_item_idx:
                with torch.inference_mode():
                    for inputs in loader:
                        x = inputs[idx]
                        # print(x.shape)
                        _ = model(x)
        
        calibrate_model(teacher, calibration_loader, [0])
        calibrate_model(student, calibration_loader, [0,1])
        calibrate_model(autoencoder, calibration_loader, [1])
    
    teacher = tq.convert(teacher, inplace=True)
    student = tq.convert(student, inplace=True)
    autoencoder = tq.convert(autoencoder, inplace=True)
    
    return teacher.eval(), student.eval(), autoencoder.eval()

# @torch.no_grad()  
def calibrate_eval_save(teacher, student, autoencoder, teacher_mean, teacher_std, train_loader, validation_loader, test_set, train_output_dir, iteration, phase = 'tmp'):

    # run intermediate evaluation
    teacher.eval()
    student.eval()
    autoencoder.eval()

    torch.save(teacher, os.path.join(train_output_dir, 'models',
                                        f'teacher_{phase}_{iteration}.pth'))
    torch.save(student, os.path.join(train_output_dir, 'models',
                                        f'student_{phase}_{iteration}.pth'))
    torch.save(autoencoder, os.path.join(train_output_dir, 'models',
                                        f'autoencoder_{phase}_{iteration}.pth'))

    print('Quantizing models...')
    st = perf_counter()
    teacher_q, student_q, autoencoder_q = quantize_model(teacher, student, autoencoder, calibration_loader=validation_loader)
    print(f'Quantization took {round(perf_counter() - st,2)} seconds')
    
    teacher_q_mean, teacher_q_std = teacher_normalization(teacher_q, train_loader, q_flag=True)

    torch.save(teacher_q.state_dict(), os.path.join(train_output_dir, 'models',
                                        f'teacher_q_{phase}_{iteration}.pth'))
    torch.save(student_q.state_dict(), os.path.join(train_output_dir, 'models',
                                        f'student_q_{phase}_{iteration}.pth'))
    torch.save(autoencoder_q.state_dict(), os.path.join(train_output_dir, 'models',
                                        f'autoencoder_q_{phase}_{iteration}.pth'))
    
    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization( # only done with non quantized model
        validation_loader=validation_loader, teacher=teacher,
        student=student, autoencoder=autoencoder,
        teacher_mean=teacher_mean, teacher_std=teacher_std,
        desc='Intermediate map normalization')
    
    q_st_start_q, q_st_end_q, q_ae_start_q, q_ae_end_q = map_normalization( # only done with quantized model
        validation_loader=validation_loader, teacher=teacher_q,
        student=student_q, autoencoder=autoencoder_q,
        teacher_mean=teacher_q_mean, teacher_std=teacher_q_std,
        desc='Intermediate map normalization', q_flag=True)

    # non quantized model
    auc = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start,
        q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=None, desc='Intermediate inference')
    print('Intermediate image auc - non quantized: {:.4f}'.format(auc))

    auc_q_3 = test( # seems like the best choice
        test_set=test_set, teacher=teacher_q, student=student_q,
        autoencoder=autoencoder_q, teacher_mean=teacher_q_mean,
        teacher_std=teacher_q_std, q_st_start=q_st_start,
        q_st_end=q_st_end, q_ae_start=q_ae_start,
        q_ae_end=q_ae_end, test_output_dir=None,
        desc='Intermediate inference', q_flag=True)
    print('Intermediate image auc - quantized (with teacher mean/std): {:.4f}'.format(auc_q_3))
    # save statistics
    
    statistics = {
        'q_st_start': q_st_start.item(),
        'q_st_end': q_st_end.item(),
        'q_ae_start': q_ae_start.item(),
        'q_ae_end': q_ae_end.item(),
        'q_st_start_q': q_st_start_q.item(),
        'q_st_end_q': q_st_end_q.item(),
        'q_ae_start_q': q_ae_start_q.item(),
        'q_ae_end_q': q_ae_end_q.item(),
        'teacher_mean': teacher_mean.cpu().tolist(),
        'teacher_std': teacher_std.cpu().tolist(),
        'teacher_q_mean': teacher_q_mean.cpu().tolist(),
        'teacher_q_std': teacher_q_std.cpu().tolist(),
        'auc': auc,
        'auc_q': auc_q_3
    }
    with open(os.path.join(train_output_dir, 'model_statistics', f'statistics_{phase}_{iteration}.json'), 'w') as f:
        json.dump(statistics, f)
        
    return auc, auc_q_3

            
if __name__ == '__main__':
    main()



