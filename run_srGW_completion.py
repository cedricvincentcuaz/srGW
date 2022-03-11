
"""
@author: cvincentcuaz
"""

import numpy as np
from srGW_algorithms.srGW_dictionary_learning import srGW_DL
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


import argparse
import os
import pickle
from tqdm import tqdm
import multiprocessing
import time
from GW_utils import utils
import torch as th
njobs = multiprocessing.cpu_count() - 1 # no parallelization performed here but could be implemented to speed up completion process

#%%
"""
    Run experiment for srGW completion on graphs described in section 4.2 
    of the main paper:
        1. 
            For K well-observed graphs {(C_k, h_k)}, estimate the target structure \bar{C} minimizing
            
            \min_{\bar{C}} \sum_{i=1}^K srGW(C_k,h_k,\bar{C}).
            
            Every observed graphs are then encoded through masses \bar{h}_k (unmixings) in the 
            target space with inner structure \bar{C}, 
            such that (C_k,h_k) is represented as (\bar{C},\bar{h}_k).
        
        2. 
            Complete partially observed graphs (create artifically knowing the ground truth from given dataset)
            by completing the graph from the srGW dictionary.
"""

# python run_srGW_completion.py -ds "imdb-b" -Ntarget [10] -lassoreg [0.] -gammareg [0.] -mode 'ADJ' -lr 0.01 -batch 32 -ep 100 -splitr 0.9 -splits 0 -compr [0.1,0.2,0.3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='srGW Completion')
    parser.add_argument('-ds','--dataset_name', type=str,help='the name of the dataset',choices=['imdb-b','imdb-m'],required=True)
    parser.add_argument('-Ntarget','--list_Ntarget', type=str,help='list of atom sizes to validate',required=True)
    parser.add_argument('-lassoreg','--list_lambda_reg', type=str,default=[0.0],help='list of regularization coefficients for promoting sparsity using Maj-Min algorithm',required=True)
    parser.add_argument('-gammareg','--list_gamma_entropy', type=str,default=[0.0],help='list of entropic regularization coefficients for using Mirror Descent algorithm',required=True)
    parser.add_argument('-mode','--graph_mode', type=str,default='ADJ',help='graph representation for graphs',choices=['ADJ','SP','LAP'])
    parser.add_argument('-lr','--learning_rate', type=float,default=0.001,help='learning rate for SGD updates - Adam is included with default values',required=True)
    parser.add_argument('-batch','--batch_size', type=int,default=16,help='batch size for stochastic updates',required=True)
    parser.add_argument('-ep','--epochs', type=int,default=100,help='epochs of srGW DL stochastic algorithm',required=True)
    parser.add_argument('-splitr','--split_rate', type=float, default=0.1,help='proportion of samples from the dataset to consider well-observed',required=True)
    parser.add_argument('-splits','--split_seed', type=int, default=0,help='seed to fix to reproduce the splitting procedure of the dataset',required=True)
    parser.add_argument('-compr','--completion_max_rates', type=list, default=[0.1, 0.2, 0.3],help='proportion of nodes to withdraw to create partially observed graphs',required=True)
    args = parser.parse_args()
    
    
    completion_init_mode_graph = ['uniform_noisy_scaleddegrees']
    degrees = False # masses of observed graphs are considered uniform
    use_optimizer = True # use Adam
    use_checkpoint = True 
    if args.graph_mode == 'ADJ':
        str_mode = ''
    else:
        str_mode = args.graph_mode
    if degrees:
        str_deg = 'degrees'
    else:
        str_deg = ''
    abspath = os.path.abspath('../')
    experiment_repo = '/results/%s/'%args.dataset_name
    init_GW = 'product'
    data_path='./real_datasets/'
    
    str_to_method = {'ADJ': 'adjacency', 'SP':'shortest_path','LAP':'laplacian',
                    'fullADJ':'augmented_adjacency','normADJ':'normalized_adjacency',
                    'SIF':'sif_distance', 'SLAP':'signed_laplacian','normLAP':'normalized_laplacian'}
    algo_seed = 0
    #run_completion_raw = False
    #run_completion_degrees=True
    use_warmstart_MM=True
    eps_inner = 1e-05
    max_iter_inner = 1000
    dtype = th.float64
    device = 'cpu'  # should be changes to use GPU instead
        
    for gamma_entropy in [float(x) for x in args.list_gamma_entropy[1:-1].split(',')]:
        for lambda_reg in [float(x) for x in args.list_lambda_reg[1:-1].split(',')]:
            for Ntarget in [int(x) for x in args.list_Ntarget[1:-1].split(',')]:
            
                if gamma_entropy ==0:
                    entropic_reg_str =''
                else:
                    entropic_reg_str = 'ENTreg%s_'%gamma_entropy
                if lambda_reg ==0:
                    reg_str = 'MMreg0.0'
                    max_iter_MM =0
                    eps_inner_MM = 0
                elif lambda_reg>0:
                    reg_str = 'MMreg%s'%lambda_reg
                    max_iter_MM = 50
                    eps_inner_MM = 10**(-5)
                else:
                    raise 'negative lambda_reg is not supported - promoting density of the OT is the goal of this regularization'
                
                warmstart_str = ''
                
                
                optim_str = {True: 'Adam', False:'SGD'}
                completion_parameters = {'split_rate':args.split_rate, 'split_seed':args.split_seed}
                print('completion_parameters:', completion_parameters)
                experiment_name = '/%s%scompletion_splitrate%s_splitseed%s_Ntarget%s_%s%s_lr%s_batch%s_epochs%s_seed%s/'%(str_mode, str_deg, args.split_rate, args.split_seed, Ntarget, entropic_reg_str, reg_str, args.lr, args.batch_size, args.epochs, algo_seed)
                print('experiment name:', experiment_name)
                method = srGW_DL(graphs=None, 
                                 masses=None, 
                                 y=None,
                                 dataset_name = args.dataset_name,
                                 mode = args.graph_mode, 
                                 Ntarget = Ntarget,
                                 experiment_repo = experiment_repo, 
                                 completion_parameters = completion_parameters, 
                                 experiment_name = experiment_name, 
                                 degrees = degrees,
                                 data_path = data_path, 
                                 dtype = dtype, 
                                 device = device)
                
                full_path = abspath + experiment_repo + experiment_name
                print('- start learning -')
                # Learn the dictionary from the noisy samples
                if args.graph_mode in ['ADJ','SP','fullADJ','normADJ']:
                    local_proj = 'nsym'
                elif args.graph_mode in ['LAP','normLAP']:
                    local_proj = 'sym'
                else:
                    raise 'unknown projection for input graph representations: %s'%args.graph_mode
                    
                if not os.path.exists(full_path):
                    
                    method.Learn_dictionary(lambda_reg=lambda_reg, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, 
                                            checkpoint_freq=5, max_iter_inner = max_iter_inner, eps_inner = eps_inner, 
                                            max_iter_MM = max_iter_MM, eps_inner_MM = eps_inner_MM,
                                            gamma_entropy=gamma_entropy, use_warmstart_MM=use_warmstart_MM, 
                                            algo_seed = algo_seed, beta_1 = 0.9, beta_2 = 0.99, use_optimizer = use_optimizer,
                                            use_checkpoint =True, proj = local_proj, init_GW = init_GW, draw_loss=False, earlystopping_patience=5)  
                
                else:
                    print('existing files:', os.listdir(full_path))
                    method.load_elements(use_checkpoint=use_checkpoint)
                    
                    method.create_srGW_operator(init_mode=init_GW, eps_inner=eps_inner, max_iter_inner=max_iter_inner, 
                                              eps_inner_MM=eps_inner_MM, max_iter_MM=max_iter_MM, lambda_reg=lambda_reg,
                                              gamma_entropy=gamma_entropy, use_warmstart_MM=use_warmstart_MM, seed=algo_seed)    
                # 1. Get best dictionary state by reconstruction error on train set
                losses_train_path = full_path+'losses_unmixings_trainG.npy'
                raw_train_graphs = method.raw_train_graphs
                raw_test_graphs = method.raw_test_graphs
                if args.graph_mode == 'ADJ':
                    train_graphs = raw_train_graphs
                    test_graphs= raw_test_graphs
                    train_masses = method.masses
                    if not degrees:
                        test_masses = [th.ones(C.shape[0], dtype=dtype, device=device)/C.shape[0] for C in test_graphs]    
                    else:
                        raise 'degrees not supported for completion tasks yet'
                else:
                    raise 'mode not supported for completion tasks yet'
                if not os.path.exists(losses_train_path):
                    print('compute unmixing for train samples ')
                    _, list_best_losses_trainG = method.compute_unmixing(use_checkpoint=use_checkpoint)
                       
                    best_idx = th.argmin(th.tensor(list_best_losses_trainG).mean(1)).item()
                    method.checkpoint_Ctarget = [method.checkpoint_Ctarget[best_idx]] # We only keep as target graphs the dictionary state leading to best results
                    method.C_target = method.checkpoint_Ctarget[0]
                    np.save(full_path+'losses_unmixings_trainG.npy', np.array(list_best_losses_trainG))
                else:
                    print('load unmixing for train samples ')
                    list_best_losses_trainG = np.load(losses_train_path)
                    best_idx = th.argmin(th.tensor(list_best_losses_trainG).mean(1))
                    method.checkpoint_Ctarget = [method.checkpoint_Ctarget[best_idx]] # We only keep as target graphs the dictionary state leading to best results
                    method.C_target = method.checkpoint_Ctarget[0]
                
                # RUN COMPLETION TASKS FOR DEGREES BASED INITIALIZATIONS
                np.random.seed(0) #fix seed for perturbations
                th.manual_seed(0)
                #initmode_to_saverstr = {'' : 'noisy_scaleddegrees',
                #                        'unif' : 'uniform_noisy_scaleddegrees'}
                for completion_warmstart in [True, False]:
                    print('using warmstart on OT for completion:', completion_warmstart)
                    if not completion_warmstart:
                        completion_warmstart_str = ''
                    else:
                        completion_warmstart_str = '_warmstartT'
                    print('[COMPLETION - degrees init] Use warmstart :', completion_warmstart)
                    
                    for completion_max_rate in args.completion_max_rates:
                        print('completion_max_rate:',completion_max_rate)
                        local_eps_inner = 1e-06
                        stacked_completion_log=[]
                        dict_filename = full_path+'/stackedcompletionlog%s_%sdegreesinitoptim_settings_maxrate%s.pkl'%(completion_warmstart_str,completion_init_mode_graph ,completion_max_rate)
                        
                        existing_evaluation_dictionary = os.path.exists(dict_filename)
                        
                        if existing_evaluation_dictionary:
                            #completion log available for all graphs ?
                            previous_results = pickle.load( open( dict_filename, "rb" ) )
                            
                        else:
                            
                            for i,trueC in tqdm(enumerate(raw_test_graphs), desc='graph completion'):
                                trueN = trueC.shape[0]
                                removed_idx = np.random.choice(np.arange(trueN), size= max(1,int(completion_max_rate*trueN)),replace=False)
                                kept_idx= [x for x in range(trueN) if (not x in removed_idx) ]
                                impC = trueC[kept_idx,:][:,kept_idx]
                                impN= impC.shape[0]
                                
                                completion_log = {'init_mode':[], 'lr':[], 'optimizer':[],'rec':[], 
                                                  'alignment_loss':[], 'acc_singlethresh':[],'precision_singlethresh':[],
                                                  'recall_singlethresh':[],'rocauc_singlethresh':[],
                                                  'trueN':trueN, 'removed_idx':removed_idx.shape[0],'time':[]}#'compC':[]}
                                completion_log['completion_lambda_reg']=[]
                                
                                list_completion_lambda_reg = [lambda_reg]
                                for completion_lambda_reg in list_completion_lambda_reg:
                                    #for local_init_mode in [initmode_to_saverstr[completion_init_mode_graph]]:
                                    for local_init_mode in [completion_init_mode_graph]:
                                        for local_lr in [0.01, 0.001]:
                                            for local_use_optimizer in [True]:
                                                start = time.time()
                                                best_completed_patch, best_loss, local_log, init_completed_patch =method.complete_patch(impC, trueN, local_lr, max_iter=5000, eps=local_eps_inner, use_optimizer=local_use_optimizer,
                                                                                                                                        proj=local_proj, use_log=True, use_warmstart=completion_warmstart, init_patch=local_init_mode)
                                                
                                                    
                                                end= time.time()
                                                # switch to numpy to be compatible with POT <= 0.8.0
                                                best_completed_patch_ = best_completed_patch.detach().cpu().numpy()
                                                init_completed_patch_ = init_completed_patch.detach().cpu().numpy()
                                                trueC_ = trueC.detach().cpu().numpy()
                                                best_loss = best_loss.item()
                                                completion_log['init_mode'].append(local_init_mode)
                                                completion_log['rec'].append(best_loss)
                                                completion_log['lr'].append(local_lr)
                                                completion_log['optimizer'].append(local_use_optimizer)
                                                completion_log['completion_lambda_reg'].append(completion_lambda_reg)
                                                
                                                local_loss, local_T = utils.np_GW2(best_completed_patch_, trueC_)
                                                aligned_rec = (local_T.T).dot(best_completed_patch_ ).dot(local_T)
                                                max_ = np.max(aligned_rec)
                                                min_ = np.min(aligned_rec)
                                                if max_ != min_: #constant outputs can occcur if the graph is fully connected
                                                    scaled_rec = (aligned_rec - min_)/(max_ - min_)
                                                
                                                    thresh_aligned_rec = np.array(scaled_rec > 0.5)
                                                else:
                                                    thresh_aligned_rec= np.ones((trueN,trueN))
                                                np.fill_diagonal(thresh_aligned_rec,0)
                                                y_true = trueC_[removed_idx,:].flatten()
                                                y_pred = thresh_aligned_rec[removed_idx,:].flatten()
                                                completion_log['acc_singlethresh'].append(accuracy_score(y_true,y_pred))
                                                completion_log['precision_singlethresh'].append(precision_score(y_true,y_pred))
                                                completion_log['recall_singlethresh'].append(recall_score(y_true,y_pred))
                                                completion_log['rocauc_singlethresh'].append(roc_auc_score(y_true,y_pred))
                                                completion_log['time'].append(end-start)
                                                
                                stacked_completion_log.append(completion_log)
                            with open(dict_filename, 'wb') as outfile:
                                pickle.dump(stacked_completion_log,outfile)
                
