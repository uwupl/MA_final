import numpy as np
import numba as nb
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# from train_main import PatchCore
# import pytorch_lightning as pl
import torch
# import cv2
import os
import warnings
import time
# import gc
import shutil
import math
from path_definitions import PLOT_DIR, RES_DIR, ROOT_DIR
#### simplenet ####
import csv
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm
import locale

locale.setlocale(locale.LC_NUMERIC, 'de_DE.UTF-8')
plt.rcParams['axes.formatter.use_locale'] = True


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min) 

def record_gpu(cuda_event):
    '''
    gpu_measurement
    '''
    cuda_event.record()
    torch.cuda.synchronize()
    
    return cuda_event

def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)
    return dist

@nb.jit(nopython=True)
def modified_kNN_score_calc_old(score_patches):
    k = score_patches.shape[1]
    weights = np.divide(np.array([k-i for i in range(k)]), 1)#((k-1)*k)/2)
    # weights = np.ones(k)
    dists = np.sum(np.multiply(score_patches, weights), axis=1)
    N_b = score_patches[np.argmax(dists)]
    w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
    score = w*np.max(dists)
    return score

@nb.jit(nopython=True)
def modified_kNN_score_calc(score_patches, n_next_patches = 0.03):
    dists = np.sum(score_patches, axis=1)#, dtype=np.float64)
    # dists = np.multiply(np.sum(score_patches, axis=1), weights, axis=1)
    # dists = np.sum(np.multiply(score_patches, weights), axis=1)
    n_next_patches = int(n_next_patches*score_patches.shape[0])
    # print('n_next_patches: ', n_next_patches)
    sorted_args = np.argsort(dists)
    score = np.zeros(n_next_patches)
    for p in range(1,n_next_patches+1):    
        N_b = score_patches[sorted_args[-p]]/1000.0# Distanzen eines Testpatches 
        exp_N_b = np.exp(N_b)
        # print(exp_N_b)
        # exp_N_b[exp_N_b >= 1e25] = 1e25
        exp_N_b_sum = np.sum(exp_N_b)
        # print('exp_N_b_sum: ', exp_N_b_sum)
        # with warnings.catch_warnings():
        #     warnings.filterwarnings('error')
        #     try:
        softmax = np.divide(np.max(exp_N_b), exp_N_b_sum)
            # except:
                # softmax = 1.0
        w = 1.0 - softmax
        score[p-1] =  w*dists[sorted_args[-p]]
    # weights_2 = np.array([(n_next_patches-i) for i in range(n_next_patches)])
    # score = np.sum(np.multiply(score, weights_2))
    return np.mean(score)

@nb.jit(nopython=True)
def modified_kNN_score_calc_numba(score_patches, n_next_patches = 5, outlier_deletion = True, outlier_factor = 50):
    '''
    numba version of adapted score calculation
    '''
    if outlier_deletion:
        sum_of_each_patch = np.sum(score_patches,axis=1)
        threshold_val = outlier_factor*np.percentile(sum_of_each_patch, 50)
        non_outlier_patches = np.argwhere(sum_of_each_patch < threshold_val).flatten()#[0]
        if len(non_outlier_patches) < score_patches.shape[0]:
            score_patches = score_patches[non_outlier_patches]
            print('deleted outliers: ', sum_of_each_patch.shape[0]-len(non_outlier_patches))
    k = score_patches.shape[1]
    weights = np.array([(k-i)**2 for i in range(k)])#np.divide(np.array([(k-i)**2 for i in range(k)]), 1, dtype=np.float64) # Summe(i²) = (k*(k+1)*(2*k+1))/6
    dists = np.sum(np.multiply(score_patches, weights), axis=1, dtype=np.float64)
    sorted_args = np.argsort(dists)
    score = np.zeros(n_next_patches)
    for p in range(1,n_next_patches+1):    
        N_b = score_patches[sorted_args[-p]].astype(np.float64)
        exp_N_b = np.exp(N_b)
        # exp_N_b[exp_N_b >= 1e25] = 1e25
        exp_N_b_sum = np.sum(exp_N_b)
        # with warnings.catch_warnings():
        #     warnings.filterwarnings('error')
        #     try:
        softmax = np.divide(np.max(exp_N_b), exp_N_b_sum)
            # except:
                # softmax = 1.0
        w = np.float64(1.0 - softmax)
        score[p-1] =  w*dists[sorted_args[-p]]
    return np.mean(score)

def prep_dirs(root, category):
    # make embeddings dir
    embeddings_path = os.path.join('./', 'embeddings', category)
    os.makedirs(embeddings_path, exist_ok=True)
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    return embeddings_path, sample_path, source_code_save_path

def get_summary_df(this_run_id: str, res_path: str, save_df = False):
    '''
    Takes a run_id, reads all files and returns a dataframe with the summary of all runs
    '''
    failed_runs = []
    correction_number = 0
    
    all_items_in_results = os.listdir(res_path)
    this_run_dirs = [this_dir for this_dir in all_items_in_results if this_dir.startswith(this_run_id)]
    img_auc_total_mean = np.array([])
    img_auc_MVTechAD_total = np.array([])
    img_auc_own = np.array([])
    backbone_storage_total = np.array([])
    backbone_flops_total = np.array([])
    feature_extraction_total = np.array([])
    embedding_of_feature_total = np.array([])   
    calc_distances_total = np.array([]) 
    calc_scores_total = np.array([])    
    total_time_total = np.array([])
    coreset_size = np.array([])
    for k, run_dir in enumerate(this_run_dirs):
        # print(k)
        file_name = 'summary_' + run_dir + '.csv'
        file_path = os.path.join(res_path, run_dir,'csv',file_name)
        try:
            pd_summary = pd.read_csv(file_path, index_col=0)
            print(pd_summary.shape)
            #### temporary ####
            # cols = pd_summary.columns
            # pd_summary = pd_summary.drop(columns=cols[16:])
            #### temporary ####
            if pd_summary.shape[1] != int(16):
                # print(k)
                print('file uncomplete: ', file_path)
                failed_runs.append(k)
                correction_number += 1
                continue
        except:
            print('file not found: ', file_path)
            failed_runs.append(k)
            correction_number += 1
            continue
        img_auc_col = dict(pd_summary.loc['img_auc_[%]'])
        img_auc_mean = np.mean(np.float32(list(img_auc_col.values())))
        
        img_auc_own = np.float32(img_auc_col.pop('own'))
        # img_auc = np.float32(pd_summary.loc['img_auc_[%]'].values)
        img_auc_MVTechAD = np.mean(np.float32(list(img_auc_col.values())))
        # img_auc_own = img_auc[-1]
        # img_auc_MVTechAD = np.mean(img_auc[:-1])
        try: # also used to determine wether it's simplenet or patchcore. simplenet has no feature length
            feature_length = np.max(np.float32(pd_summary.loc['resulting_feature_length'].values))
            simplenet = False
        except:
            feature_length = None
            simplenet = True
        if not simplenet:
            backbone_storage = np.max(np.float32(pd_summary.loc['backbone_storage_[MB]'].values))
            backbone_flops = np.max(np.float32(pd_summary.loc['backbone_mult_adds_[M]'].values))
            feature_extraction = np.max(np.float32(pd_summary.loc['feature_extraction_[ms]'].values))
            embedding_of_feature = np.max(np.float32(pd_summary.loc['embedding_of_features_[ms]'].values))
            calc_distances = np.max(np.float32(pd_summary.loc['calc_distances_[ms]'].values))
            calc_scores = np.max(np.float32(pd_summary.loc['calc_scores_[ms]'].values))
            total_time = np.max(np.float32(pd_summary.loc['total_time_[ms]'].values))
        else:
            backbone_storage = np.max(np.float32(pd_summary.loc['backbone_storage_[MB]'].values))
            backbone_flops = np.max(np.float32(pd_summary.loc['backbone_mult_adds_[M]'].values))
            feature_extraction = np.mean(np.float32(pd_summary.loc['feature_extraction_[ms]'].values))
            embedding_of_feature = np.mean(np.float32(pd_summary.loc['embedding_of_features_[ms]'].values))
            calc_distances = np.mean(np.float32(pd_summary.loc['calc_distances_[ms]'].values))
            calc_scores = np.mean(np.float32(pd_summary.loc['calc_scores_[ms]'].values))
            total_time = np.mean(np.float32(pd_summary.loc['total_time_[ms]'].values))
        # coreset_size = np.max(np.float32(pd_summary.loc['coreset_size'].values))
        if (k - correction_number) == 0:
            # img_auc_total = img_auc
            img_auc_total_mean = img_auc_mean
            img_auc_total_own = img_auc_own
            img_auc_MVTechAD_total = img_auc_MVTechAD
            backbone_storage_total = backbone_storage
            backbone_flops_total = backbone_flops  
            feature_extraction_total = feature_extraction
            embedding_of_feature_total = embedding_of_feature
            calc_distances_total = calc_distances
            calc_scores_total = calc_scores
            total_time_total = total_time
            feature_length_total = feature_length
        else:
            # img_auc_total = np.vstack((img_auc_total, img_auc))
            img_auc_total_mean = np.vstack((img_auc_total_mean, img_auc_mean))
            img_auc_MVTechAD_total = np.vstack((img_auc_MVTechAD_total, img_auc_MVTechAD))
            img_auc_total_own = np.vstack((img_auc_total_own, img_auc_own))
            backbone_storage_total = np.vstack((backbone_storage_total, backbone_storage))
            backbone_flops_total = np.vstack((backbone_flops_total, backbone_flops))
            feature_extraction_total = np.vstack((feature_extraction_total, feature_extraction))
            embedding_of_feature_total = np.vstack((embedding_of_feature_total, embedding_of_feature))
            calc_distances_total = np.vstack((calc_distances_total, calc_distances))
            calc_scores_total = np.vstack((calc_scores_total, calc_scores))
            total_time_total = np.vstack((total_time_total, total_time))
            feature_length_total = np.vstack((feature_length_total, feature_length))

    if type(img_auc_total_mean) == np.ndarray:
        num_runs = img_auc_total_mean.shape[0]
    else:
        num_runs = 1
    summary_np = np.zeros((11, num_runs))
    helper_list = [img_auc_total_mean, img_auc_MVTechAD_total, img_auc_total_own, backbone_storage_total, backbone_flops_total, feature_extraction_total, embedding_of_feature_total, calc_distances_total, calc_scores_total, total_time, feature_length_total]
    for i, entry in enumerate(helper_list):
        summary_np[i, :] = entry.flatten()
    run_summary_dict = {}
    for k in range(len(img_auc_total_mean)):
        # print(k)
        for a, b in zip(summary_np[:,k].flatten(), ['img_auc_mean', 'img_auc_MVTechAD', 'img_auc_own','backbone_storage', 'backbone_flops', 'feature_extraction', 'embedding_of_feature', 'calc_distances', 'calc_scores', 'total_time', 'feature_length']):
            if k == 0:
                run_summary_dict[b] = [float(a)]
            else:
                run_summary_dict[b] += [float(a)]

    index_list = [name[len(this_run_id):] for k, name in enumerate(this_run_dirs) if k not in failed_runs]
    run_summary_df = pd.DataFrame(run_summary_dict, index=index_list)
    if save_df:
        file_path = os.path.join(res_path, 'csv', f'{int(time.time())}_summary_of_this_{this_run_id}.csv')
        run_summary_df.to_csv(file_path, index=False)
    return run_summary_df


def plot_results(labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage, feature_length, 
                 fig_size = (20,10), title = 'Comparison', only_auc = False, width = 0.4, show_f_length = False, show_storage = False, 
                 loc_legend = (1.15, 0.06), bar_labels = ['feature extraction', 'embedding', 'search', 'calc scores'], german=True,
                 save_fig = False, res_path = PLOT_DIR, show = True):
    '''
    visualizes results in bar chart
    '''
    # mpl.rc('text', usetex=True)
    # tmp 
    # res_path = os.path.join(res_path, 'rn18')
    
    
    font_scaler = 1.0
    # font = {'fontname':'FreeSerif'}
    if german:
        bar_labels_inference = ['Extraktion', 'Einbettung', 'NN-Suche', 'Anomaliegrad']
        # title = 'Vergleich'
        bar_labels_auc = ['Granulat', 'MVTecAD']
        y_label_inference = '(Lauf-) Zeit [ms]'
        delimiter = '.'
        f_l = ' Merkmalslaenge'
        locale.setlocale(locale.LC_ALL, 'de_DE')
        plt.rcParams['axes.formatter.use_locale'] = True
        # plt.rcParams['axes.
    else:
        bar_labels_inference = ['feature extraction', 'embedding', 'search', 'calc scores']
        # title = 'Comparison'
        bar_labels_auc = ['Own Dataset', 'MVTechAD (mean)']
        y_label_inference = 'run time [ms] (mean)'
        delimiter = '.'
        f_l = 'feature length'
    
    loc_legend = (1.15, 0.06)
    # latex font style
    font = {'fontname':'FreeSerif'}
    
    if show_f_length:
        new_labels = []
        for k in range(len(labels)):
            new_labels += [labels[k] + '\n' + str(int(feature_length[k])) + f_l]
        del labels
        labels = new_labels
        del new_labels
        
    if show_storage:
        new_labels = []
        for k in range(len(labels)):
            new_labels += [labels[k] + '\n' + str(round(storage[k],2)) + ' MB']
        del labels
        labels = new_labels
        del new_labels
    
    # print(labels)
    # labels[0] += '\n' + 'ReLU (default)'
    
    x = np.arange(len(labels))  # the label locations
    # width = width  # the width of the bars
    fig, ax = plt.subplots(figsize=fig_size, dpi=300)

    if not only_auc:
        ax_2 = ax.twinx()
        rects1 = ax.bar(x - 0.5*width, feature_extraction, width, label=bar_labels_inference[0], color = 'crimson')
        rects2 = ax.bar(x - 0.5*width, embedding, width, label=bar_labels_inference[1], bottom=feature_extraction, color = 'purple')
        rects3 = ax.bar(x - 0.5*width, search, width, label=bar_labels_inference[2], bottom=list(np.array(embedding) + np.array(feature_extraction)), color = 'slateblue')
        rects4 = ax.bar(x - 0.5*width, calc_distances, width, label=bar_labels_inference[3],bottom=list(np.array(embedding) + np.array(feature_extraction) + np.array(search)), color = 'darkgoldenrod')
        # rects4 = ax.bar(x - 0.5*width, anomaly_map, width, label='anomaly map',bottom=list(np.array(embedding_cpu) + np.array(feature_extraction_cpu) + np.array(search_memory)), color = 'darkgoldenrod')
        rects_1 = ax_2.bar(x + 0.25 * width, own_auc, width*0.3, label = bar_labels_auc[0], color = 'black')
        rects_2 = ax_2.bar(x + 0.75 * width, MVTechAD_auc, width*0.3, label = bar_labels_auc[1], color = 'grey')
        # rects5 = ax.bar(x + width, total_cpu, width, label='total')
        # rects3 = ax.bar(x, )

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(y_label_inference, **font)#, fontsize=8*font_scaler)
        # print(ax.get_yticks())
        y_ticks = ax.get_yticks()
        if german:
            y_ticks = [str(y).replace('.',',') for y in y_ticks]
        ax.set_yticks(ax.get_yticks(), y_ticks , **font)#, fontsize=8*font_scaler)
        ax.set_title(title, **font)#, fontsize=10*font_scaler)
        ax_2.set_ylabel('AUROC [%]', **font)#, fontsize=8*font_scaler)
        ax_2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        ax_2.set_ylim([50,105])
        ax.set_xticks(x, labels, **font)
        # l1 = ax.legend()#loc='upper right')
        # l2 = ax_2.legend()#loc='upper left')

        ax.bar_label(rects1, padding=3, fmt=own_formatter_2f, **font)#, fontsize=6*font_scaler)
        # ax.bar_label(rects2, padding=3)
        # ax.bar_label(rects3, padding=3)
        ax.bar_label(rects4, padding=3, fmt=own_formatter_2f, **font)#, fontsize=6*font_scaler)
        ax_2.bar_label(rects_1, padding=3,fmt=own_formatter_1f, **font)#, fontsize=6*font_scaler)
        ax_2.bar_label(rects_2, padding=3,fmt=own_formatter_1f, **font)#, fontsize=6*font_scaler)
        ax_2.set_yticks([50,70,80,90,100],[50,70,80,90,100],**font) # 60 excluded for legend
        # ax_2.set
        
        handles_1, labels_1 = ax.get_legend_handles_labels()
        handles_2, labels_2 = ax_2.get_legend_handles_labels()
        
        both_handles = handles_1 + handles_2
        both_labels = labels_1 + labels_2
        
        ax.legend(both_handles, both_labels, loc='lower right', bbox_to_anchor=loc_legend, bbox_transform=ax_2.transAxes, prop='FreeSerif')#, **font)#, fontsize=8*font_scaler)
    else:
        rects_1 = ax.bar(x - 0.5*width, own_auc, width, label = 'Own Auc', color = 'black')
        rects_2 = ax.bar(x + 0.5*width, MVTechAD_auc, width, label = 'MVTecAD Auc', color = 'grey')
        ax.set_ylabel('AUROC')#, fontsize=8*font_scaler)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        ax.set_title(title)#, fontsize=10*font_scaler)
        ax.set_xticks(x, labels)
        ax.legend()
        ax.bar_label(rects_1, padding=3,fmt=own_formatter_2f)
        ax.bar_label(rects_2, padding=3,fmt=own_formatter_2f)
        ax.set_yticks([50,60,70,80,90,100])
        ax.set_ylim([50,105])
    
    fig.tight_layout()    
    current_font_size = mpl.rcParams['font.size']
    mpl.rcParams.update({'font.size': current_font_size*font_scaler})
    
    # ax_2.axhline(100, linestyle='--', color='gray', alpha=0.2) let's do that in tikz
    if save_fig:
        file_name = str(int(time.time())) + title.replace(' ', '_') + '_' + '.svg'
        if not os.path.exists(res_path):
            os.makedirs(res_path) 
        plt.savefig(os.path.join(res_path, file_name), bbox_inches = 'tight')
        if True:

            import tikzplotlib as tikz
            fig = plt.gcf()
            tikzplotlib_fix_ncols(fig)
            # tikz.clean_figure()
            tikz.save(os.path.join(res_path,file_name.replace('.svg', '.tex')))
        
    if show:
        plt.show()

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def own_formatter_1f(x):
    tmp = f'{x:1.1f}'
    return tmp.replace('.',',')

def own_formatter_2f(x):
    tmp = f'{x:1.2f}'
    return tmp.replace('.',',')

def own_formatter_3f(x):
    tmp = f'{x:1.3f}'
    return tmp.replace('.',',')


def extract_vals_for_plot(summary_df: pd.DataFrame):
    '''
    Takes pandas DataFrame and extracts values for plotting
    '''
    labels = np.array(summary_df.index, dtype=str)
    feature_extraction = summary_df.loc[:, 'feature_extraction'].values
    embedding = summary_df.loc[:, 'embedding_of_feature'].values
    search = summary_df.loc[:, 'calc_distances'].values
    calc_distances = summary_df.loc[:, 'calc_scores'].values
    own_auc = summary_df.loc[:, 'img_auc_own'].values*100
    MVTechAD_auc = summary_df.loc[:, 'img_auc_MVTechAD'].values*100
    storage = summary_df.loc[:, 'backbone_storage'].values
    # coreset_size = get_coreset_size_length_inner_process(summary_df)
    feature_length = summary_df.loc[:, 'feature_length'].values
    
    return labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage, feature_length

def remove_failed_run_dirs(failed_runs: np.ndarray):
    '''
    removes failed runs from run_dirs
    '''
    dir_path = os.path.dirname(os.path.abspath(__file__))
    for folder in failed_runs:
        path = os.path.join(dir_path,'results', folder)
        if os.path.isdir(path):
            shutil.rmtree(path)
    return None

def remove_all_empty_run_dirs():
    '''
    removes all empty run dirs
    '''
    counter = 0
    dir_path =RES_DIR# os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    for folder in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, folder, 'csv')):
            if len(os.listdir(os.path.join(dir_path, folder, 'csv'))) == 0:
                counter += 1 
                shutil.rmtree(os.path.join(dir_path, folder))
    print(f'Removed {counter} empty folders')
    return None

def remove_uncomplete_runs(dir_path = RES_DIR):#r'/mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_PatchCore/'):
    '''
    checks if all csv files are complete and removes uncomplete runs
    '''
    counter = 0
    # dir_path = os.path.join(main_dir, 'results')
    for folder in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, folder, 'csv')):
            if len(os.listdir(os.path.join(dir_path, folder, 'csv'))) == 0:
                counter += 1 
                shutil.rmtree(os.path.join(dir_path, folder))
            else:
                for file in os.listdir(os.path.join(dir_path, folder, 'csv')):
                    if file.startswith('summary'):
                        try:
                            summary_df = pd.read_csv(os.path.join(dir_path, folder, 'csv', file), index_col=0)
                        except:
                            counter += 1 
                            shutil.rmtree(os.path.join(dir_path, folder))
                            break
                        if summary_df.shape[1] != int(16):
                            counter += 1 
                            shutil.rmtree(os.path.join(dir_path, folder))
                            break
    print(f'Removed {counter} empty folders')
    return None

def get_coreset_size_length_inner_process(pd_summary):
    '''
    some string operations to get the number of features used (length), returns int
    '''
    result = []
    for k in range(pd_summary.shape[0]):
        try:
            b = pd_summary.loc[:,'coreset_size'].values[0]
            res = b[b.find(' ')+1:b.find(')')]
        except:
            res = 'NA'
    
    print(res)
    return res

def sort_by_attribute(attribute, labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage, feature_length):
    '''
    returns a sorted dataframe by attribute.
    Give as attribute a copy of one of the other arguments.
    '''
    order = np.argsort(attribute)
    return labels[order], feature_extraction[order], embedding[order], search[order], calc_distances[order], own_auc[order], MVTechAD_auc[order], storage[order], feature_length[order] 

def filter_by_contain_in_label_str(labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage, feature_length ,to_contain: list, to_delete: list):
    '''
    returns a list of labels that contain to_contain and do not contain to_delete
    '''
    for pattern in to_contain:
        mask_1 = [True if label.__contains__(pattern) else False for label in labels]
        labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage, feature_length = labels[mask_1], feature_extraction[mask_1], embedding[mask_1], search[mask_1], calc_distances[mask_1], own_auc[mask_1], MVTechAD_auc[mask_1], storage[mask_1], feature_length[mask_1]
    for pattern in to_delete:
        mask_2 = [True if not label.__contains__(pattern) else False for label in labels]
        labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage, feature_length = labels[mask_2], feature_extraction[mask_2], embedding[mask_2], search[mask_2], calc_distances[mask_2], own_auc[mask_2], MVTechAD_auc[mask_2], storage[mask_2], feature_length[mask_2]
    return labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage, feature_length

def shorten_labels(labels, to_delete: list):
    '''
    returns a list of labels with shortened names
    '''
    for k in range(len(to_delete)):
        labels = [label.replace(to_delete[k],'') for label in labels]
    return labels

def get_plot_ready_data(this_run_id, res_path, to_contain, to_delete, take_n_best = None): 
    '''
    returns data, that is ready to be plotted. Specify filters by the to_contain and to_delete lists. optional.
    '''
    summary_pd = get_summary_df(this_run_id, res_path)
    labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage, feature_length = extract_vals_for_plot(summary_pd)
    # for k in range(len(coreset_size)):
        # labels[k] = labels[k] + '\n(' + str(coreset_size[k]) + ')'
    # print('Raw: #', len(labels))
    # if attribute_to_sort_by is not None:
    labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage, feature_length = sort_by_attribute(MVTechAD_auc, labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage, feature_length) #
    if take_n_best is not None:
        labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage, feature_length = labels[-take_n_best:], feature_extraction[-take_n_best:], embedding[-take_n_best:], search[-take_n_best:], calc_distances[-take_n_best:], own_auc[-take_n_best:], MVTechAD_auc[-take_n_best:], storage[-take_n_best:], feature_length[-take_n_best:]
    
    labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage, feature_length = filter_by_contain_in_label_str(labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage, feature_length, to_contain=to_contain, to_delete=to_delete)
    # print('Filtered: #',len(labels))
    # print('1: ',labels)
    labels = shorten_labels(labels, to_delete=to_contain) # and mention in title of plot instead in order to keep somehow short labels
    for k in range(len(labels)):
        labels[k] = labels[k].replace('-','\n')
    # print('2: ',labels)
    return labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage, feature_length

def remove_test_dir():
    '''
    removes test dir if it is bigger than 5GB
    '''
    if get_dir_size(os.path.join(ROOT_DIR, 'test'))/(1024*1024*1024) > 5:
        print('delete')
        # os.remove(os.path.join(os.getcwd(), 'test', 'test.txt'))
        try:
            shutil.rmtree(os.path.join(ROOT_DIR, 'test'))
            print('deleted')
        except:
            print('could not delete')  
            
def get_dir_size(path='.'):
    '''
    returns size of directory in bytes
    '''
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return np.divide(e_x, np.sum(e_x)) 


# not used
def calc_anomaly_map(score_patches, batch_size_1, load_size):
        '''
        calculates anomaly map based on score_patches
        '''
        if batch_size_1:
            anomaly_map = score_patches[:,0].reshape((int(math.sqrt(len(score_patches[:,0]))),int(math.sqrt(len(score_patches[:,0])))))
            a = int(load_size) # int, 64 
            anomaly_map_resized = cv2.resize(anomaly_map, (a, a)) # [8,8] --> [64,64]
            anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)# shape [8,8]
        else:
            anomaly_map = [score_patch[:,0].reshape((int(math.sqrt(len(score_patch[:,0]))),int(math.sqrt(len(score_patch[:,0]))))) for score_patch in score_patches]
            a = int(load_size)
            anomaly_map_resized = [cv2.resize(this_anomaly_map, (a, a)) for this_anomaly_map in anomaly_map]
            anomaly_map_resized_blur = [gaussian_filter(this_anomaly_map_resized, sigma=4) for this_anomaly_map_resized in anomaly_map_resized]
        return anomaly_map_resized_blur
    
    
    
#### simplenet ####
# import csv
# import logging
# import os
# import random

# import matplotlib.pyplot as plt
# import numpy as np
# import PIL
# import torch
# import tqdm

LOGGER = logging.getLogger(__name__)


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)

        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)
        f, axes = plt.subplots(1, 2 + int(masks_provided))
        axes[0].imshow(image.transpose(1, 2, 0))
        axes[1].imshow(mask.transpose(1, 2, 0))
        axes[2].imshow(segmentation)
        f.set_size_inches(3 * (2 + int(masks_provided)), 3)
        f.tight_layout()
        f.savefig(savename)
        plt.close()


def create_storage_folder(
    main_folder_path, project_folder, group_folder, run_name, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder, run_name)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics


