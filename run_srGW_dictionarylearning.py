"""
@author: cvincentcuaz
"""

import numpy as np
import torch as th
from data_handler import dataloader
from srGW_algorithms.srGW_dictionary_learning import srGW_DL
from srGW_algorithms.srGW import initializer_semirelaxed_GW
from GW_utils import GW_kernels

#import pylab as pl
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score,rand_score,adjusted_rand_score
import os
import pandas as pd
import argparse
from tqdm import tqdm 
#%%
"""
    Run experiment for srGW Dictionary Learning described in section 4.1 
    of the main paper:
        For K observed graphs {(C_k, h_k)}, estimate the target structure \bar{C} minimizing
        
        \min_{\bar{C}} \sum_{i=1}^K srGW(C_k,h_k,\bar{C}).
        
        Every observed graphs are then encoded through masses \bar{h}_k (unmixings) in the 
        target space with inner structure \bar{C}, 
        such that (C_k,h_k) is represented as (\bar{C},\bar{h}_k).
    
    Implementation steps:
        1. Dictionary Learning:
            Learn the dictionary \bar{C} thanks to the stochastic algorithm (see Alg.2)
        2. Clustering (evaluate unmixings):
            Compute Kmeans on the unmixings { \bar{h}_k}_{k \in [K]}
        3. Classification (evaluate embedded graphs using GW kernels):
            Extract embedded graphs { \bar{C}, \bar{h}_k}_{k \in [K]}
            Then Compute GW kernels on embedded graphs.
            Set run_classification = True (default False) if you want to reproduce these experiments.
"""

# python run_srGW_dictionarylearning.py -ds "imdb-b" -Ntarget [10] -lassoreg [0.] -gammareg [0] -mode 'ADJ' -lr 0.01 -batch 32 -ep 100 -seeds [0]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='srGW Dictionary Learning')
    parser.add_argument('-ds','--dataset_name', type=str,help='the name of the dataset',choices=['imdb-b','imdb-m'],required=True)
    parser.add_argument('-Ntarget','--list_Ntarget', type=str,help='list of atom sizes to validate',required=True)
    parser.add_argument('-lassoreg','--list_lambda_reg', type=str,default=[0.0],help='list of regularization coefficients for promoting sparsity using Maj-Min algorithm',required=True)
    parser.add_argument('-gammareg','--list_gamma_entropy', type=str,default=[0.0],help='list of entropic regularization coefficients for using Mirror Descent algorithm',required=True)
    parser.add_argument('-mode','--graph_mode', type=str,default='ADJ',help='graph representation for graphs',choices=['ADJ','SP','LAP'])
    parser.add_argument('-lr','--learning_rate', type=float,default=0.001,help='learning rate for SGD updates - Adam is included with default values',required=True)
    parser.add_argument('-batch','--batch_size', type=int,default=16,help='batch size for stochastic updates',required=True)
    parser.add_argument('-ep','--epochs', type=int,default=100,help='epochs of srGW DL stochastic algorithm',required=True)
    parser.add_argument('-seeds','--list_seeds', type=str,default=str([0]),help='seed to initialize stochastic algorithm and ensure reproductibility')
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
    eps_inner = 1e-5 # 
    use_warmstart_MM= True
    if args.graph_mode == 'ADJ':
        str_mode =''
    else:
        str_mode = args.graph_mode+'_'
    dtype = th.float32
    device = 'cpu'
    counting_plot = 1
    run_classification = False  # Only performed on seed 0 during experiments

    run_classification = True
    
    for gamma_entropy in [float(x) for x in args.list_gamma_entropy[1:-1].split(',')]:
        for lambda_reg in [float(x) for x in args.list_lambda_reg[1:-1].split(',')]:
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
                        experiment_name= '/%sNtarget%s_%s%s_%s_lr%s_batch%s_epochs%s_seed%s/'%(str_mode, Ntarget, entropic_reg_str, reg_str, optim_str[use_optimizer], args.learning_rate, args.batch_size, args.epochs, seed)
                    else:
                        experiment_name= '/%sNtarget%s_%s%s_%s_lr%s_batch%s_epochs%s_degrees_seed%s/'%(str_mode, Ntarget, entropic_reg_str, reg_str, optim_str[use_optimizer], args.learning_rate, args.batch_size, args.epochs, seed)
                    
                    # Load graphs and corresponding labels for clustering benchmark
                    X,y=dataloader.load_local_data(data_path,args.dataset_name)                
                    graphs= [th.tensor(X[t].distance_matrix(method=str_to_method[args.graph_mode]), dtype=dtype, device=device) for t in range(X.shape[0])]
                    if not degrees:#uniform distributions
                        masses= [th.ones(C.shape[0], dtype=dtype, device=device)/C.shape[0] for C in graphs]                            
                    else:# compute normalized degree distribution
                        masses =[]
                        for C in graphs:
                            h = C.sum(axis=0)
                            masses.append( h / h.sum())
                        
                    # Instantiate the class for srGW Dictionary Learning
                    method=srGW_DL(graphs=graphs, masses=masses, y=y,
                                   dataset_name = args.dataset_name,
                                   mode=args.graph_mode, Ntarget=Ntarget,
                                   experiment_repo=experiment_repo,
                                   experiment_name=experiment_name, degrees=degrees, data_path= data_path,
                                   dtype=dtype, device=device)
                               
                    full_path = abspath+experiment_repo+experiment_name
                    print('full_path:', full_path)
                    if os.path.exists(full_path):
                        print('-  srGW Dictionary already existing. loading dictionary state  -')
                        method.load_elements(use_checkpoint=True)
                        
                        method.create_srGW_operator(init_mode=init_GW, eps_inner=eps_inner, max_iter_inner=max_iter_inner, 
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
                        method.Learn_dictionary(lambda_reg=lambda_reg, epochs=args.epochs, lr=args.learning_rate, batch_size=args.batch_size, checkpoint_freq=5,
                                             max_iter_inner=max_iter_inner, eps_inner=eps_inner,max_iter_MM=max_iter_MM, eps_inner_MM=eps_inner_MM,
                                             gamma_entropy=gamma_entropy, use_warmstart_MM=use_warmstart_MM, 
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
                       
                    # 3. Classification: Compute GW kernels between embedded graphs thanks to the learned dictionary

                    if run_classification:
                        val_nseeds = 50
                        if (not os.path.exists(full_path+'/res_SVCclassification.csv')):# and (not os.path.exists(full_path+'/res_unmixings_SVCclassification.csv')):
                            print('-- not existing SVC classification results: start computing --')
                            res_clustering = pd.read_csv(full_path+'/res_unmixings_clustering.csv') 
                            list_unmixings = np.load(full_path+'unmixings.npy')
                            best_checkpoint = res_clustering['checkpoint'].item()
                            unmixings = list_unmixings[best_checkpoint].astype(np.float64)
                            method.Ctarget = method.checkpoint_Ctarget[best_checkpoint]
                            Cbar = np.array(method.Ctarget.clone().detach().cpu().numpy(), dtype=np.float64)
                            embedded_graphs, embedded_masses = GW_kernels.get_srGW_embedded_graphs(Cbar, unmixings)
                            
                            GW_pairwise_distances = GW_kernels.GW_matrix(embedded_graphs, embedded_masses)
                            res_classif, detailed_res_classif = GW_kernels.nested_classification_SVC(D=GW_pairwise_distances,
                                                                                                     labels=method.y, 
                                                                                                     n_folds=10, n_iter=10,verbose=False)
                            res_classif.to_csv(full_path+'/res_SVCclassification.csv',index=False)
                            detailed_res_classif.to_csv(full_path+'/detailedres_SVCclassification.csv',index=False)
                            
                        else:
                            print('// existing SVC classification results')
                        if (not os.path.exists('%s/res_SVCclassification_validatedseeds%s.csv'%(full_path, val_nseeds))):
                            res_clustering = pd.read_csv(full_path+'/res_unmixings_clustering.csv') 
                            best_checkpoint = res_clustering['checkpoint'].item()
                            method.Ctarget = method.checkpoint_Ctarget[best_checkpoint]
                                                
                            list_OT_validated, list_loss_validated=[],[]
                            for idx in tqdm(range(len(method.graphs)), desc='unmixings validated'):
                                OT,loss = None,np.inf
                                C = method.graphs[idx]
                                h = method.masses[idx]
                                N = C.shape[0]
                                for local_seed in range(val_nseeds):
                                    
                                    T_init = initializer_semirelaxed_GW(init_mode='random', p=h, N1=N, N2=Ntarget, seed=local_seed)
                                    local_OT, local_loss = method.srGW_operator(C, method.masses[idx], method.Ctarget, T_init)
                                    if local_loss<loss:
                                        OT, loss = local_OT, local_loss                                                
                                list_OT_validated.append(OT)
                                list_loss_validated.append(loss) 
                            
                                unmixings_validated = np.array([T.sum(axis=0) for T in list_OT_validated], dtype=np.float64)
                                np.save('%s/unmixings_validatedseeds%s.npy'%(full_path, val_nseeds), unmixings_validated )
                                np.save('%s/losses_unmixings_validatedseeds%s.npy'%(full_path, val_nseeds), np.array(list_loss_validated))
                                                        
                            #if (not os.path.exists(str_SVCgraphs)) :
                            print('running SVC on embedded graphs with best unmixings')
                            Cbar = np.array(method.Ctarget.clone().detach().cpu().numpy(), dtype=np.float64)
                            
                            embedded_graphs_validated, embedded_masses_validated = GW_kernels.get_srGW_embedded_graphs(Cbar, unmixings_validated)
                            
                            GW_pairwise_distances_validated = GW_kernels.GW_matrix(embedded_graphs_validated, embedded_masses_validated)
                            res_classif_validated, detailed_res_classif_validated = GW_kernels.nested_classification_SVC(D=GW_pairwise_distances_validated,
                                                                                                                        labels=method.y, 
                                                                                                                        n_folds=10, n_iter=10,verbose=False)
                            res_classif_validated.to_csv('%s/res_SVCclassification_validatedseeds%s.csv'%(full_path, val_nseeds), index=False)
                            detailed_res_classif_validated.to_csv('%s/detailedres_SVCclassification_validatedseeds%s.csv'%(full_path, val_nseeds),index=False)
                            