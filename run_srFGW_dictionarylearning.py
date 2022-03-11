"""
@author: cvincentcuaz
"""

import numpy as np
import torch as th
from data_handler import dataloader
from srGW_algorithms.srFGW_dictionary_learning import srFGW_DL
from srGW_algorithms.srGW import initializer_semirelaxed_GW

#import pylab as pl
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score,rand_score,adjusted_rand_score
import os
import pandas as pd
import argparse
import pylab as pl
from GW_utils import GW_kernels
from tqdm import tqdm 

#%%
"""
    Run experiment for srFGW Dictionary Learning described in section 4.1 
    of the main paper:
        For K observed labeled graphs {(C_k, h_k)}, estimate the target structure (\bar{C}, \bar{F}) minimizing
        
        \min_{\bar{C}, \bar{F}} \sum_{i=1}^K srFGW(C_k, F_k, h_k, \bar{C}, \bar{F}).
        
        Every observed graphs are then encoded through masses \bar{h}_k (unmixings) in the 
        target space with inner structure \bar{C} and features \bar{F}, 
        such that (C_k, F_k, h_k) is represented as (\bar{C}, \bar{F}, \bar{h}_k).
    
    Implementation steps:
        1. Dictionary Learning:
            Learn the dictionary (\bar{C}, \bar{F}) thanks to the stochastic algorithm (see Alg.2)
        2. Clustering (evaluate unmixings):
            Compute Kmeans on the unmixings { \bar{h}_k}_{k \in [K]}
        3. Classification (evaluate embedded graphs using FGW kernels):
            Extract embedded labeled graphs { \bar{C}, \bar{F}, \bar{h}_k}_{k \in [K]}
            Then Compute FGW kernels on embedded graphs.
            Set run_classification = True (default False) if you want to reproduce these experiments.
"""

#python run_srFGW_dictionarylearning.py -ds "mutag" -Ntarget [10] -alpha [0.5] -lassoreg [0.] -gammareg [0] -mode 'ADJ' -lrC 0.01 -lrF 0.01 -batch 32 -ep 100 -seeds [0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='srFGW Dictionary Learning')
    parser.add_argument('-ds', '--dataset_name', type=str, help='the name of the dataset', choices=['mutag', 'ptc', 'bzr', 'cox2', 'enzymes', 'protein'], required=True)
    parser.add_argument('-Ntarget', '--list_Ntarget', type=str, help='list of atom sizes to validate', required=True)
    parser.add_argument('-alpha', '--list_alpha', type=str, help='list of trade-off parameter of the srFGW cost', required=True)
    parser.add_argument('-lassoreg', '--list_lambda_reg', type=str, default=[0.0], help='list of regularization coefficients for promoting sparsity using Maj-Min algorithm', required=True)
    parser.add_argument('-gammareg', '--list_gamma_entropy', type=str, default=[0.0], help='list of entropic regularization coefficients for using Mirror Descent algorithm', required=True)
    parser.add_argument('-mode', '--graph_mode', type=str, default='ADJ', help='graph representation for graphs',choices=['ADJ','SP','LAP'])
    parser.add_argument('-lrC', '--learning_rate_C', type=float, default=0.001, help='learning rate for SGD updates of the structures- Adam is included with default values', required=True)
    parser.add_argument('-lrF', '--learning_rate_F', type=float, default=0.001, help='learning rate for SGD updates of the features- Adam is included with default values', required=True)
    parser.add_argument('-batch', '--batch_size', type=int, default=16, help='batch size for stochastic updates', required=True)
    parser.add_argument('-ep', '--epochs', type=int, default=100, help='epochs of srGW DL stochastic algorithm', required=True)
    parser.add_argument('-seeds', '--list_seeds', type=str, default=str([0]), help='seed to initialize stochastic algorithm and ensure reproductibility')
    args = parser.parse_args()
    
    str_to_method = {'ADJ': 'adjacency', 'SP':'shortest_path','LAP':'laplacian',
                     'fullADJ':'augmented_adjacency','normADJ':'normalized_adjacency'}
    
    degrees=False # False = set input graph distribution to uniform
    use_optimizer=True # Use Adam Optimizer
    abspath = os.path.abspath('../')
    experiment_repo ='/results/%s/'%args.dataset_name
    init_GW = 'product'
    data_path = './real_datasets/'
    max_iter_inner = 1000 # maximum number of iterations for srGW solver
    eps_inner = 10**(-5) # 
    use_warmstart_MM= True
    if args.graph_mode=='ADJ':
        str_mode =''
    else:
        str_mode=args.graph_mode+'_'
    dtype = th.float32
    device = 'cpu'
    counting_plot = 1
    run_classification = False  # Only performed on seed 0 during experiments

    for gamma_entropy in [float(x) for x in args.list_gamma_entropy[1:-1].split(',')]:
        for lambda_reg in [float(x) for x in args.list_lambda_reg[1:-1].split(',')]:
            for alpha in [float(x) for x in args.list_alpha[1:-1].split(',')]:
                for Ntarget in [int(x) for x in args.list_Ntarget[1:-1].split(',')]:
                    for seed in [int(x) for x in args.list_seeds[1:-1].split(',')]:
                        #Detailed name for storing experiments
                        optim_str = {True: 'Adam', False:'SGD'}
                        if gamma_entropy == 0:
                            entropic_reg_str = ''
                        else:
                            entropic_reg_str = 'ENTreg%s_'% gamma_entropy
                        if lambda_reg == 0:
                            reg_str = 'reg0.0'
                            max_iter_MM = 0
                            eps_inner_MM = 0
                        elif lambda_reg > 0:
                            reg_str = 'MMreg%s'%lambda_reg
                            max_iter_MM = 50
                            eps_inner_MM = 1e-5
                        else:
                            raise 'negative lambda_reg is not supported - promoting density of the OT is the goal of this regularization'
                    
                        if not degrees:
                            experiment_name= '/%sNtarget%s_alpha%s_%s%s_%s_lrC%s_lrF%s_batch%s_epochs%s_seed%s/'%(str_mode, Ntarget, alpha, entropic_reg_str, reg_str, optim_str[use_optimizer], args.learning_rate_C, args.learning_rate_F, args.batch_size, args.epochs, seed)
                        else:
                            experiment_name= '/%sNtarget%s_alpha%s_%s%s_%s_lrC%s_lrF%s_batch%s_epochs%s_degrees_seed%s/'%(str_mode, Ntarget, alpha, entropic_reg_str, reg_str, optim_str[use_optimizer], args.learning_rate_C, args.learning_rate_F, args.batch_size, args.epochs, seed)
                        
                        # Load graphs and corresponding labels for clustering benchmark
                        if args.dataset_name in ['mutag', 'ptc']:
                            one_hot = True
                            standardized_features = False
                        else:
                            one_hot = False
                            standardized_features = True 
                        X, y = dataloader.load_local_data(data_path, args.dataset_name, one_hot=one_hot)                
                        graphs = [th.tensor(X[t].distance_matrix(method=str_to_method[args.graph_mode]), dtype=dtype, device=device) for t in range(X.shape[0])]
                        features = [th.tensor(X[t].values(), dtype=dtype, device=device) for t in range(X.shape[0])]
                        if not degrees:#uniform distributions
                            masses= [th.ones(C.shape[0], dtype=dtype, device=device)/C.shape[0] for C in graphs]                            
                        else:# compute normalized degree distribution
                            masses =[]
                            for C in graphs:
                                h = C.sum(axis=0)
                                masses.append( h / h.sum())
                        if standardized_features:
                            print('stardardizing features')
                            list_mean_F = [F.mean(axis=0) for F in features]
                            stacked_features = th.stack(list_mean_F)
                            print('stacked_features :', stacked_features.shape[0])
                            print('before norm: mean F[0] = ', stacked_features[0, :])
                            for i in range(stacked_features.shape[1]):
                                # We assume that each features component follow different distributions
                                # hence standardize features component independently
                                mean_ = stacked_features[:, i].mean()
                                std_ = stacked_features[:, i].std()
                                for F in features:
                                    F[:, i] = (F[:, i] - mean_)/std_
                            print('after norm: means F[0] = ', features[0].mean(axis=0))
                            
                        # Instantiate the class for srGW Dictionary Learning
                        method = srFGW_DL(graphs=graphs, features=features, masses=masses, y=y,
                                          alpha=alpha,
                                          dataset_name = args.dataset_name,
                                          mode=args.graph_mode, Ntarget=Ntarget,
                                          experiment_repo=experiment_repo,
                                          experiment_name=experiment_name, degrees=degrees, data_path= data_path,
                                          dtype=dtype, device=device)
                                   
                        full_path = abspath+experiment_repo+experiment_name
                        print('full_path:', full_path)
                        if os.path.exists(full_path):
                            print('-  srFGW Dictionary already existing. loading dictionary state  -')
                            method.load_elements(use_checkpoint=True)
                            #If visualization useful:
                            #batch_log_loss= np.load(full_path+'/batch_loss.npy')
                            #epoch_log_loss= np.load(full_path+'/epoch_loss.npy')
                            # Instantiate srGW solver to use as a class method depending on provided parameters
                            method.create_srFGW_operator(init_mode=init_GW, eps_inner=eps_inner, max_iter_inner=max_iter_inner, 
                                                      eps_inner_MM=eps_inner_MM, max_iter_MM=max_iter_MM, lambda_reg=lambda_reg,
                                                      gamma_entropy=gamma_entropy, use_warmstart_MM=use_warmstart_MM, seed=seed)        
       
                        else:
                            # 1. srGW Dictionary Learning
                            print('- start learning -')
                            if args.graph_mode in ['ADJ','SP','fullADJ','normADJ']:
                                local_proj = 'nsym'
                            elif args.graph_mode in ['LAP','normLAP']:
                                local_proj = 'sym'
                            else:
                                raise 'unknown projection for input graph representations: %s'%args.graph_mode
                            method.Learn_dictionary(lambda_reg=lambda_reg, epochs=args.epochs, lrC=args.learning_rate_C, lrF=args.learning_rate_F, 
                                                    batch_size=args.batch_size, checkpoint_freq=5, max_iter_inner=max_iter_inner, eps_inner=eps_inner,
                                                    max_iter_MM=max_iter_MM, eps_inner_MM=eps_inner_MM, gamma_entropy=gamma_entropy, use_warmstart_MM=use_warmstart_MM, 
                                                    algo_seed=seed, beta_1=0.9, beta_2=0.99, use_optimizer=use_optimizer,
                                                    use_checkpoint=True, proj=local_proj, init_GW =init_GW, draw_loss=False, earlystopping_patience=5)  
                           
                        # 2. Clustering: Compute euclidean kmeans on srGW unmixings
                        n_clusters = len(np.unique(method.y)) 
                   
                        if os.path.exists(full_path+'/res_unmixings_clustering.csv') :
                            pass
                        else:
                            km_embeddings_res = {'checkpoint':[],'RI':[],'adj_RI':[],'ami':[],'loss_mean':[],'loss_std':[]}
                            print('computing unmixing - classical product transport initializations.')
                            try:
                                list_unmixings = np.load(full_path+'unmixings.npy')
                                list_best_losses = np.load(full_path+'losses_unmixings.npy')
                            except:
                                list_best_T, list_best_losses = method.compute_unmixing(use_checkpoint=True)
                                list_unmixings = np.array([[T.sum(axis=0).cpu().numpy() for T in list_OT] for list_OT in list_best_T])
                                list_best_losses = np.array(list_best_losses)
                                np.save(full_path+'unmixings.npy', list_unmixings)
                                np.save(full_path+'losses_unmixings.npy', list_best_losses)
                            means_rec = list_best_losses.mean(1)
                            best_checkpoint = np.argmin(means_rec)
                            unmixings = list_unmixings[best_checkpoint]
                            km_embeddings_res = {'checkpoint':[], 'RI':[], 'adj_RI':[], 'ami':[], 'loss_mean':[], 'loss_std':[]}
                            km = KMeans(n_clusters =n_clusters, n_init=100,random_state = 0).fit(unmixings)
                            pred = km.labels_
                            ami = adjusted_mutual_info_score(method.y, pred, average_method='arithmetic')
                            rand_index = rand_score(method.y,pred)
                            adj_rand_index= adjusted_rand_score(method.y,pred)
                            km_embeddings_res['checkpoint'].append(best_checkpoint)
                            km_embeddings_res['RI'].append(rand_index)
                            km_embeddings_res['adj_RI'].append(adj_rand_index)
                            km_embeddings_res['ami'].append(ami)
                            km_embeddings_res['loss_mean'].append(np.mean(list_best_losses[best_checkpoint]))
                            km_embeddings_res['loss_std'].append(np.std(list_best_losses[best_checkpoint]))
                            pd.DataFrame(km_embeddings_res).to_csv(full_path+'res_unmixings_clustering.csv')
                          
                        # 3. Classification: Compute FGW kernels between embedded graphs thanks to the learned dictionary

                        if run_classification:
                            val_nseeds = 50
                            if (not os.path.exists(full_path+'/res_SVCclassification.csv')):# and (not os.path.exists(full_path+'/res_unmixings_SVCclassification.csv')):
                                print('-- not existing SVC classification results: start computing --')
                                res_clustering = pd.read_csv(full_path+'/res_unmixings_clustering.csv') 
                                list_unmixings = np.load(full_path+'unmixings.npy')
                                best_checkpoint = res_clustering['checkpoint'].item()
                                unmixings = list_unmixings[best_checkpoint].astype(np.float64)
                                method.Ctarget, method.Ftarget = method.checkpoint_Ctarget[best_checkpoint], method.checkpoint_Ftarget[best_checkpoint]                            
                                Cbar = np.array(method.Ctarget.clone().detach().cpu().numpy(), dtype=np.float64)
                                Fbar = np.array(method.Ftarget.clone().detach().cpu().numpy(), dtype=np.float64)
                                embedded_graphs, embedded_features, embedded_masses = GW_kernels.get_srFGW_embedded_graphs(Cbar, Fbar, unmixings)
                                
                                FGW_pairwise_distances = GW_kernels.FGW_matrix(embedded_graphs, embedded_features, embedded_masses, alpha)
                                res_classif, detailed_res_classif = GW_kernels.nested_classification_SVC(D=FGW_pairwise_distances,
                                                                                                         labels=method.y, 
                                                                                                         n_folds=10, n_iter=10,verbose=False)
                                res_classif.to_csv(full_path+'/res_SVCclassification.csv',index=False)
                                detailed_res_classif.to_csv(full_path+'/detailedres_SVCclassification.csv',index=False)
                                
                            else:
                                print('// existing SVC classification results')
                            if (not os.path.exists('%s/res_SVCclassification_validatedseeds%s.csv'%(full_path, val_nseeds))):
                                res_clustering = pd.read_csv(full_path+'/res_unmixings_clustering.csv') 
                                best_checkpoint = res_clustering['checkpoint'].item()
                                method.Ctarget, method.Ftarget = method.checkpoint_Ctarget[best_checkpoint], method.checkpoint_Ftarget[best_checkpoint]  
                                                    
                                list_OT_validated, list_loss_validated=[],[]
                                for idx in tqdm(range(len(method.graphs)), desc='unmixings validated'):
                                    OT,loss = None,np.inf
                                    C = method.graphs[idx]
                                    F = method.features[idx]
                                    h = method.masses[idx]
                                    N = C.shape[0]
                                    for local_seed in range(val_nseeds):
                                        
                                        T_init = initializer_semirelaxed_GW(init_mode='random', p=h, N1=N, N2=Ntarget, seed=local_seed, dtype=dtype, device=device)
                                        local_OT,local_loss = method.srFGW_operator(C, F, method.masses[idx], method.Ctarget, method.Ftarget, T_init)
                                        if local_loss < loss:
                                            OT, loss = local_OT, local_loss                                                
                                    list_OT_validated.append(OT)
                                    list_loss_validated.append(loss) 
                                
                                unmixings_validated = np.array([T.sum(axis=0) for T in list_OT_validated], dtype=np.float64)
                                np.save('%s/unmixings_validatedseeds%s.npy'%(full_path, val_nseeds), unmixings_validated )
                                np.save('%s/losses_unmixings_validatedseeds%s.npy'%(full_path, val_nseeds), np.array(list_loss_validated))
                                                        
                                #if (not os.path.exists(str_SVCgraphs)) :
                                print('running SVC on embedded graphs with best unmixings')
                                Cbar = np.array(method.Ctarget.clone().detach().cpu().numpy(), dtype=np.float64)
                                Fbar = np.array(method.Ftarget.clone().detach().cpu().numpy(), dtype=np.float64)
                                
                                embedded_graphs_validated, embedded_features_validated, embedded_masses_validated = GW_kernels.get_srFGW_embedded_graphs(Cbar, Fbar, unmixings_validated)
                                
                                FGW_pairwise_distances_validated = GW_kernels.FGW_matrix(embedded_graphs_validated, embedded_features_validated, embedded_masses_validated, method.alpha)
                                res_classif_validated, detailed_res_classif_validated = GW_kernels.nested_classification_SVC(D=FGW_pairwise_distances_validated,
                                                                                                                            labels=method.y, 
                                                                                                                            n_folds=10, n_iter=10,verbose=False)
                                res_classif_validated.to_csv('%s/res_SVCclassification_validatedseeds%s.csv'%(full_path, val_nseeds), index=False)
                                detailed_res_classif_validated.to_csv('%s/detailedres_SVCclassification_validatedseeds%s.csv'%(full_path, val_nseeds),index=False)
                                