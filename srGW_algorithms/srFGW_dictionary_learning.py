"""
@author: cvincentcuaz

Torch implementation of semi-relaxed Gromov-Wasserstein dictionary learning,
detailed in section 4 of the main paper. Also contains the srgw completion framework
detailed in this latter section.

It Supports all kinds of regularization for solving the unmixing problem
detailed in the section 3 of the main paper.

"""

from data_handler import dataloader
import numpy as np
from tqdm import tqdm 
import os
import pandas as pd
import pickle 
from sklearn.model_selection import train_test_split
import pylab as pl
from sklearn.cluster import KMeans
from  scipy.sparse.csgraph import shortest_path
import torch as th
import srGW_algorithms.srGW as srGW

#%%


str2rpzfunctions ={'ADJ': (lambda x: x),
                   'SP':shortest_path}
                   


class srFGW_DL():
    
    def __init__(self,
                 graphs:list=None, 
                 features:list=None,
                 masses:list=None,
                 y:list=None,
                 dataset_name:str=None,
                 mode:str='ADJ',
                 Ntarget:int=None,
                 alpha:float=0.5,
                 experiment_repo:str=None, 
                 experiment_name:str=None,
                 degrees:bool=False,
                 completion_parameters:dict={},
                 data_path:str='../data/',
                 dtype:type=th.float64,
                 device:str='cpu'):
        """
        Parameters
        ----------
        graphs: list of torch arrays (N_k, N_k). 
                If set to "None", graphs will be downloaded from the specified "dataset_name"
        features: list of torch arrays (N_k, d). 
                If set to "None", features will be downloaded from the specified "dataset_name"
        masses: list of numpy arrays matching respectively graphs sizes.
                If set to "None", computed based on downloaded graphs from "dataset_name"
        y: numpy array (N_k,) 
            If set to "None", labels will be downloaded from the specified "dataset_name". Used for analysis of the unsupervised learning process.
        dataset_name : 
            name of the dataset to experiment on. To match our data loaders it is restricted to ['imdb-b','imdb-m','balanced_clustertoy','clustertoy2C']
        mode : 
            representations for input graphs. (e.g) 'ADJ':adjacency / 'SP': shortest path 
        Ntarget: 
            size of target graph which summarizes the dataset
        alpha:
            Trade-off parameter involved in the Fused Gromov-Wasserstein loss function
        experiment_repo : 
            subrepository to save results of the experiment 
        experiment_name : 
            subrepository to save results of the experiment under the 'experiment repo' repository
        degrees: 
            either to use uniform distribution (False) for each graph, else use degree distribution (True)
        completion_parameters: 
            dict used to handle split into D_train and D_test
        data_path : 
            path where data is. The default is '../data/'.
        """
        """
        Parameters
        ----------
        dataset_name : name of the dataset to experiment on. To match our data loaders it is restricted to ['imdb-b','imdb-m','balanced_clustertoy','clustertoy2C']
        mode : representations for input graphs. (e.g) 'ADJ':adjacency / 'SP': shortest path 
        Ntarget: size of target graph which summarizes the dataset
        experiment_repo : subrepository to save results of the experiment 
        experiment_name : subrepository to save results of the experiment under the 'experiment repo' repository
        degrees: either to use uniform distribution (False) for each graph, else use degree distribution (True)
        denoising_parameters:   If learning dictionary on noisy observations
                                must contain keys ['n_samples', 'n_trials', 'list_probas']
        data_path : path where data is. The default is '../data/'.
        """
        
        self.experiment_repo= experiment_repo
        self.experiment_name = experiment_name
        print('dataset_name:', dataset_name)
        str_to_method = {'ADJ': 'adjacency', 'SP':'shortest_path','LAP':'laplacian'}
        self.degrees=degrees
        self.Ntarget = Ntarget
        self.alpha=alpha
        self.dataset_name= dataset_name
        self.mode = mode
        self.completion_parameters = completion_parameters
        self.dtype = dtype
        self.device = device

        if graphs is None:
            if dataset_name in ['mutag','ptc']: #onehot representation for features
                one_hot = True    
                X, self.y = dataloader.load_local_data(data_path, dataset_name, one_hot=one_hot)
            else:
                X, self.y = dataloader.load_local_data(data_path, dataset_name)
            
            if completion_parameters =={}:
                if self.mode in str_to_method.keys():
                    self.graphs= [th.tensor(X[t].distance_matrix(method=str_to_method[mode]), dtype=self.dtype, device=self.device) for t in range(X.shape[0])]
                    self.features = [th.tensor(X[t].values(), dtype=self.dtype, device=self.device) for t in range(X.shape[0])]
                    if not self.degrees:
                        #uniform distributions
                        self.masses= [th.ones(Cs.shape[0], dtype=self.dtype, device=self.device)/Cs.shape[0] for Cs in self.graphs] 
                    else:
                        print('computing degree distributions')
                        self.masses =[]
                        for C in self.graphs:
                            h = C.sum(axis=0)
                            self.masses.append( h / h.sum())
                else:
                    raise 'unknown mode /graph representation'
            else:
                assert self.mode =='ADJ'
                y = self.y.copy()
                list_X = [th.tensor(X[t].distance_matrix(method=str_to_method['ADJ']), dtype=self.dtype, device=self.device) for t in range(X.shape[0])]
                list_F = [th.tensor(X[t].values(), dtype=self.dtype, device=self.device) for t in range(X.shape[0])]
                N = len(list_X)
                train_idx,test_idx = train_test_split(np.arange(N), test_size=self.completion_parameters['split_rate'], stratify=np.array(y), random_state = self.completion_parameters['split_seed'])
                self.raw_train_graphs, self.raw_test_graphs= [list_X[i] for i in train_idx], [list_X[i] for i in test_idx]
                self.raw_train_features, self.raw_test_features= [list_F[i] for i in train_idx], [list_F[i] for i in test_idx]
                self.train_y, self.test_y= [y[i] for i in train_idx], [y[i] for i in test_idx]  
                self.graphs = self.raw_train_graphs
                self.features = [list_F[idx] for idx in train_idx]
                        
                if not self.degrees:
                    #uniform distributions
                    self.masses= [th.ones(C.shape[0], dtype=self.dtype, device=self.device)/C.shape[0] for C in self.raw_train_graphs]
                
                else:
                    print('computing degree distributions')
                    self.masses =[]
                    for C in self.raw_train_graphs:
                        h = C.sum(axis=0)
                        self.masses.append( h / h.sum())
                print('number of graphs in the train dataset for completion experiments:', len(self.graphs))
        else:# The graphs to learn on are already given  
            assert len(graphs) == len(features) == len(masses)    
            self.mode = mode
            self.dataset_name = dataset_name
            self.graphs = graphs
            self.features = features
            self.masses = masses
            self.y = y
            
        self.d= self.features[0].shape[1] # dimension of features
        
        self.dataset_size = len(self.graphs)
        if self.dataset_name in ['mutag', 'ptc', 'bzr', 'cox2', 'enzymes', 'protein']:
            self.undirected = True
        else:
            self.undirected = th.all([th.all(self.graphs[i] == self.graphs[i].T) for i in range(self.dataset_size)])
            print('All graphs in the dataset are undirected?', self.undirected)
        
    def init_dictionary(self, seed:int=0, use_checkpoint:bool=True, verbose:bool=False):
        """
            Initialize the graph structure atom following strategies:
                - graph atom ~ N(0.5,0.01)
            
            use_checkpoint instantiates the placeholder used for tracking with the early stopping scheme.
            
        """
        np.random.seed(seed)
        x = th.tensor(np.random.normal(loc=0.5, scale=0.01, size=(self.Ntarget,self.Ntarget)), dtype=self.dtype, device=self.device)
        
        if self.proj in ['nsym','sym']:
            self.Ctarget = (x + x.T) / 2.
        if self.proj == 'nsym':
            self.Ctarget = self.Ctarget.clamp(min=0.)
        
        #Kmeans on the features to initialize the features of the dictionary
        stacked_features = th.cat(self.features)
        stacked_features = stacked_features.detach().cpu().numpy()
        
        km = KMeans(n_clusters = self.Ntarget, n_init=10, random_state = seed).fit(stacked_features)
        self.Ftarget = th.tensor(km.cluster_centers_, dtype=self.dtype, device=self.device)
    
        if use_checkpoint:
            self.checkpoint_Ctarget = []
            self.checkpoint_Ftarget = []
        
    
    def init_optimizer(self):
        #Initialization for our numpy implementation of adam optimizer
        self.C_adam_moment1 = th.zeros((self.Ntarget, self.Ntarget), dtype=self.dtype, device=self.device)#Initialize first  moment vector
        self.C_adam_moment2 = th.zeros((self.Ntarget, self.Ntarget), dtype=self.dtype, device=self.device)#Initialize second moment vector
        self.F_adam_moment1 = th.zeros((self.Ntarget, self.d), dtype=self.dtype, device=self.device)#Initialize first  moment vector
        self.F_adam_moment2 = th.zeros((self.Ntarget, self.d), dtype=self.dtype, device=self.device)#Initialize second moment vector
        self.adam_count = 1
        
    
    def create_srFGW_operator(self,init_mode:str='product',
                             eps_inner:float=10**(-6), 
                             max_iter_inner:int=1000,
                             gamma_entropy:float=0,
                             lambda_reg:float=None,
                             eps_inner_MM:float=10**(-6), 
                             max_iter_MM:int=50, 
                             use_warmstart_MM:bool=True,
                             seed:int=0):
        """
        Parameters
        ----------
        init_mode : str, optional
            Initialization mode for srGW's conditional gradient solver among ['product','random']. The default is 'product' (i.e h_1.h_2^T) .
        eps_inner : float, optional
            convergence precision used in the CG solver. The default is 10**(-6).
        max_iter_inner : int, optional
            maximum number of iterations for the CG solver if it has not converged yet. The default is 1000.
        gamma_entropy : float, optional
            Entropic parameter used in our Mirror Descent algorithm. 
            The default is 0. If set to 0 we use the CG solver otherwise the MD solver.
        lambda_reg : float, optional
            Regularization parameter for promoting sparsity of our embeddings using the MM solver.
            The default is None. If set in [None, 0] we call the CG or MD solvers depending on gamma_entropy.
        eps_inner_MM : float, optional
            convergence precision used in the MM solver (outer loop containing CG or MD solver iteration). 
            The default is 10**(-6).
        max_iter_MM : int, optional
            maximum number of iterations for the MM solver if it has not converged yet. The default is 50.
        use_warmstart_MM : bool, optional
            Specify either to reuse the previously computed OT plan in the MM solver outer loop. The default is True.
        seed : int, optional
            Random seed if random initialization is required (init_mode=='random'). The default is 0.
        
        Returns
        -------
        Instantiate self.srGW_operator : 
            the srGW solver function used for the dictionary learning,taking as inputs,
        
            C1: np.array of size (N,N), input graph.
            h1: np.array of size (N,), input node distribution.
            C2: np.array of size (Nbar,Nbar), graph atom.
            T_init: np.array of size (N,Nbar), optional.
                    Transport plan to initialize the srGW, if set to None, related init parameters of the function will be used.
        """
        # Just create an operator for unmixing step involved in each iteration of the dictionary learning
        if lambda_reg == 0: # This condition means We do not use concave sparsity promoting regularization with MM solver.
            if gamma_entropy ==0:
                self.srFGW_operator = (lambda C1, F1, h1, C2, F2, T_init: srGW.cg_semirelaxed_fused_gromov_wasserstein(C1, F1, h1, C2, F2, self.alpha, self.undirected, init_mode, T_init, False,
                                                                                                        eps_inner, max_iter_inner, seed, False, self.device, self.dtype))
            else:
                self.srFGW_operator = (lambda C1, F1, h1, C2, F2, T_init: srGW.md_semirelaxed_fused_gromov_wasserstein(C1, F1, h1, C2, F2, gamma_entropy, self.alpha, self.undirected, init_mode, T_init, False,
                                                                                                        eps_inner, max_iter_inner, seed, False, self.device, self.dtype))
        else:  # We use Majorization-Minimization solver.
            
            self.srFGW_operator = (lambda C1, F1, h1, C2, F2, T_init: srGW.mm_lpl1_semirelaxed_fused_gromov_wasserstein(C1, F1, h1, C2, F2, self.alpha, gamma_entropy, T_init, init_mode, self.undirected,
                                                                                                         0.5, lambda_reg, False, use_warmstart_MM, eps_inner, eps_inner_MM, max_iter_inner,
                                                                                                         max_iter_MM, seed, False, False, self.dtype, self.device))
                        
    
    def Learn_dictionary(self,
                         lambda_reg:float,
                         max_iter_inner:int,
                         eps_inner:float,
                         lrC:float, lrF:float, batch_size:int, epochs:int, algo_seed:int,
                         max_iter_MM:int=0,
                         eps_inner_MM:float = 0.,
                         use_warmstart_MM:bool=True,
                         gamma_entropy:float=0.,
                         beta_1:float=0.9, 
                         beta_2:float=0.99,
                         use_optimizer:bool=True, 
                         checkpoint_freq:int=100,
                         earlystopping_patience:int = 2,
                         use_checkpoint:bool = True, 
                         proj:str= 'nsym',
                         init_GW:str='product', 
                         draw_loss:bool=False):
        """
        Stochastic Algorithm to learn srGW dictionaries 
        described in Section 4 of the main paper and Algorithm 2,
        further details in the supplementary material. 

        Parameters
        ----------
        lambda_reg : sparse regularization coefficient
        max_iter_inner : maximum number of iterations for the Conditional Gradient algorithm on {wk}
        eps_inner: precision to stop "srGW solver" based on relative variation of the loss
        max_iter_MM : maximum number of iterations for the Majorization minimization algorithm on {wk}
                    > only used if lambda_reg >0.
        eps_inner_MM: precision to stop "srGW MM- solver" based on relative variation of the loss
                    > only used if lambda_reg >0.
        gamma_entropy: regularization coefficient of the negative quadratic regularization on unmixings
        eps : precision to stop our learning process based on relative variation of the loss
        lr : Initial learning rate of Adam optimizer
        batch_size : batch size 
        algo_seed : initialization random seed
        OT_loss : GW discrepency ground cost. The default is 'square_loss'.
        beta_1 : Adam parameter on gradient. The default is 0.9.
        beta_2 : Adam parameter on gradient**2. The default is 0.99.
        use_checkpoint : To save dictionary state and corresponding unmixing at different time steps. The default is False.
        verbose : Check the good evolution of the loss. The default is False.
        """
        np.random.seed(algo_seed)
        th.manual_seed(algo_seed)
        if lambda_reg == 0:
            self.settings = {'Ntarget':self.Ntarget, 'max_iter_inner':max_iter_inner, 
                             'eps_inner':eps_inner, 'epochs':epochs,
                             'lrC':lrC, 'lrF':lrF, 'batch_size':batch_size,
                             'algo_seed':algo_seed, 'beta1':beta_1, 'beta2':beta_2,'l2_reg':0,'lambda_reg':0, 
                             'use_optimizer':use_optimizer,'init_GW':init_GW, 'proj':proj}
        else:
            self.settings = {'Ntarget':self.Ntarget, 'max_iter_FW':max_iter_inner,'eps_inner_FW':eps_inner,'max_iter_MM':max_iter_MM,'eps_inner_MM':eps_inner_MM,
                             'lrC':lrC, 'lrF':lrF, 'batch_size':batch_size,'epochs':epochs, 'algo_seed':algo_seed, 'beta1':beta_1, 'beta2':beta_2,'lambda_reg':lambda_reg, 
                             'use_optimizer':use_optimizer,'init_GW':init_GW, 'proj':proj,'use_warmstart_MM':use_warmstart_MM}

        if gamma_entropy !=0:
            self.settings['gamma_entropy'] = gamma_entropy
        self.proj = proj
        
        self.init_dictionary(algo_seed, use_checkpoint)
        algo_seed=None
        if use_optimizer:
            self.init_optimizer()
        self.create_srFGW_operator(init_mode=init_GW,eps_inner=eps_inner, max_iter_inner=max_iter_inner, 
                                    eps_inner_MM=eps_inner_MM, max_iter_MM=max_iter_MM,lambda_reg=lambda_reg,
                                    gamma_entropy=gamma_entropy,use_warmstart_MM=use_warmstart_MM,seed=algo_seed)
        
        T = len(self.graphs)
        self.log ={'batch_loss':[], 'epoch_loss':[]}
        
        best_epoch_global_rec = np.inf
        consecutive_global_rec_drops =0 # Used to decide on when to stop learning.
        
        for epoch in tqdm(range(epochs), desc='epochs'):
            seen_graphs_count = 0
            epoch_global_rec = 0
            while seen_graphs_count < self.dataset_size:
                #batch sampling
                seen_graphs_count+=batch_size
                batch_t = np.random.choice(range(T), size=batch_size, replace=False)
                #print('batch idx:', batch_t)
                best_T = []
                batch_loss = 0
                for k,t in enumerate(batch_t):
                    local_T, local_loss= self.srFGW_operator(self.graphs[t], self.features[t], self.masses[t], self.Ctarget, self.Ftarget, None)
                    best_T.append(local_T)
                    batch_loss += local_loss
                self.log['batch_loss'].append(batch_loss.item())
                epoch_global_rec += batch_loss
                    
                #Stochastic update
                gradC= th.zeros_like(self.Ctarget)
                gradF= th.zeros_like(self.Ftarget)
                for k,t in enumerate(batch_t):
                    hk = best_T[k].sum(0)
                    gradC += self.Ctarget * (hk[:, None] @ hk[None, :]) - (best_T[k].T) @ self.graphs[t] @ best_T[k]
                    gradF += th.diag(hk) @ self.Ftarget - (best_T[k].T) @ self.features[t]
                gradC *= 2. / batch_size
                gradF *= 2. / batch_size
                if not use_optimizer:
                    self.Ctarget -= lrC * gradC
                    self.Ftarget -= lrF * gradF
                else:
                    
                    C_m1_t = beta_1 * self.C_adam_moment1 + (1 - beta_1) * gradC
                    C_m2_t = beta_2 * self.C_adam_moment2 + (1 - beta_2) * (gradC**2)
                    C_m1_t_unbiased = C_m1_t / (1 - beta_1**self.adam_count)
                    C_m2_t_unbiased = C_m2_t / (1 - beta_2**self.adam_count)
                    self.Ctarget -= lrC * C_m1_t_unbiased / (th.sqrt(C_m2_t_unbiased) + 1e-15)
                    F_m1_t = beta_1 * self.F_adam_moment1 + (1-beta_1) * gradF
                    F_m2_t = beta_2 * self.F_adam_moment2 + (1-beta_2) * (gradF**2)
                    F_m1_t_unbiased = F_m1_t / (1-beta_1**self.adam_count)
                    F_m2_t_unbiased = F_m2_t / (1-beta_2**self.adam_count)
                    self.Ftarget -= lrF * F_m1_t_unbiased / (th.sqrt(F_m2_t_unbiased) + 1e-15)
                    
                    self.C_adam_moment1 = C_m1_t
                    self.C_adam_moment2 = C_m2_t
                    self.F_adam_moment1 = F_m1_t
                    self.F_adam_moment2 = F_m2_t
                    self.adam_count +=1
                #projection on nonnegative matrices
                #projection on nonnegative matrices
                if proj == 'nsym':
                    self.Ctarget = th.clamp(self.Ctarget, min=0.)
            self.log['epoch_loss'].append(epoch_global_rec.item())
        
            if epoch == 0:
                print('saved settings:', self.settings)
                self.save_elements(save_settings=True, use_checkpoint = use_checkpoint)
            elif epoch > 0 and (epoch % checkpoint_freq == 0):
                print('checkpoint_step to evaluate embeddings and decide on early stopping')
                self.save_elements(save_settings=False, use_checkpoint = use_checkpoint)
                if draw_loss:
                    pl.figure(1, (10,5))
                    pl.clf()
                    pl.subplot(121)
                    pl.plot(self.log['batch_loss'])
                    pl.title('loss evolution by batches')
                    pl.xlabel('iterations');pl.ylabel('reconstruction loss')
                    pl.subplot(122)
                    pl.plot(self.log['epoch_loss'])
                    pl.title('loss evolution by epochs')
                    pl.xlabel('iterations');pl.ylabel('reconstruction loss')
                    pl.tight_layout()
                    pl.show()      
                _, list_losses = self.compute_unmixing(use_checkpoint = False)
                mean_rec = np.mean(list_losses)
                if mean_rec < best_epoch_global_rec:
                    best_epoch_global_rec = mean_rec
                    consecutive_global_rec_drops = 0            
                    print('[unmixings check] epoch:%s / new best epoch global rec :%s'%(epoch, best_epoch_global_rec))
                else:
                    consecutive_global_rec_drops += 1
                    print('[not improved- unmixings check] epoch :%s / current epoch loss :%s / fails:%s '%(epoch, mean_rec, consecutive_global_rec_drops))
                    if consecutive_global_rec_drops > earlystopping_patience:
                        break

    def compute_unmixing(self, use_checkpoint:bool = False):
        """
        Parameters
        ----------
        use_checkpoint : bool, optional. The default is False.
            If set to False, compute unmixings on self.Ctarget (current state).
            Else if set to True, compute unmixings on all dictionary states saved in self.checkpoint_Ctarget.
        Returns
        -------
        If use_checkpoint ==False:
            best_T, best_losses: (list of np.array, list) corresponding to OT providing the unmixings and the corresponding srGW divergences. 
        If use_checkpoint ==True:
            list_best_T, list_best_losses: (list of lists of np.array, list of lists) OT and losses for each dictionary state stored in self.checkpoint_Ctarget. 
        """
        if not use_checkpoint:
            print('computing srGW unmixings on current dictionary state')
        else:
            print('computing srGW unmixings on all saved dictionary states')
        
        T= len(self.graphs)
        
        if not use_checkpoint :
            best_T = []
            best_losses = []
            for t in range(T):
                local_T, local_loss = self.srFGW_operator(self.graphs[t], self.features[t], self.masses[t], self.Ctarget, self.Ftarget, T_init=None)                    
                best_T.append(local_T)
                best_losses.append(local_loss.item())
            return best_T, best_losses
        else: #ran over all saved dictionary graph state
            list_best_T =[]
            list_best_losses=[]
            for i in range(len(self.checkpoint_Ctarget)):
                local_list_bestT = []
                local_list_bestlosses = []
                for t in range(T):
                    local_T, local_loss = self.srFGW_operator(self.graphs[t], self.features[t], self.masses[t], self.checkpoint_Ctarget[i], self.checkpoint_Ftarget[i], T_init=None)                    
                    local_list_bestT.append(local_T)
                    local_list_bestlosses.append(local_loss.item())
                    
                list_best_T.append(local_list_bestT)
                list_best_losses.append(local_list_bestlosses)
            return list_best_T, list_best_losses
        
    
    def complete_patch(self,
                       patch_graph:np.array,
                       patch_features:np.array,
                       Nfullpatch:int, 
                       lrC:float=0.01, 
                       lrF:float=0.01, 
                       max_iter:int=100,
                       eps:float=10**(-6),
                       proj:str='nsym',
                       algo_seed:int = 0,
                       use_optimizer:bool=True,
                       beta_1:float=0.9,
                       beta_2:float=0.99,
                       use_warmstart:float=False,
                       use_log:bool=False,
                       init_patch_graph:str='random',
                       init_patch_features:str='random_range'):
        np.random.seed(0)
        th.manual_seed(0)
        if not (proj in ['nsym','sym']):
            raise "only proj in ['nsym','sym'] is supported for now"
        if use_log:
            local_log = {'loss':[]}
        else:
            local_log = None
        
        Npatch = patch_graph.shape[0]
        learnable_mask_graph = th.ones((Nfullpatch, Nfullpatch), dtype=self.dtype, device=self.device)
        learnable_mask_features = th.ones((Nfullpatch, self.d), dtype=self.dtype, device=self.device)
        learnable_mask_graph[:Npatch, :Npatch] = 0.
        learnable_mask_features[:Npatch, :]= 0.
        
        assert Nfullpatch >= Npatch
        # Initialization of the patch (graph, features)
        ### 1. Dealing with init of the graph part
        if init_patch_graph == 'random':
            np.random.seed(algo_seed)
            #x = np.random.uniform(low=0.1, high=0.9, size=(self.Ntarget,self.Ntarget))
            x = th.tensor( np.random.normal(loc=0.5, scale=0.01, size=(Nfullpatch,Nfullpatch)), dtype=self.dtype, device=self.device)
            
            completed_patch_graph = (x + x.T) / 2.
            th.fill_diagonal(completed_patch_graph, 0.)# no diagonal as we do not seek for super nodes
        else:
            raise 'unsupported init_patch_graph : %s'%init_patch_graph
        ### 2. Then dealing with init of the features part
        if init_patch_features == 'random_range':
            #get the range of features and generate features randomly in this range 
            min_ = patch_features.min(0).detach().numpy()
            max_ = patch_features.max(0).detach().numpy()
            completed_patch_features = th.tensor(np.random.uniform(low=min_, high=max_, size=(Nfullpatch, self.d)), dtype=self.dtype, device=self.device)
        else:
            raise 'unsupported init_patch_features : %s'%init_patch_features

        #Init of Adam Optimizer is use_optimizer is True.
        if use_optimizer:
            graph_adam_moment1 = th.zeros((Nfullpatch, Nfullpatch), dtype=self.dtype, device=self.device)#Initialize first  moment vector
            graph_adam_moment2 = th.zeros((Nfullpatch, Nfullpatch), dtype=self.dtype, device=self.device)#Initialize second moment vector
            features_adam_moment1 = th.zeros((Nfullpatch, self.d), dtype=self.dtype, device=self.device)#Initialize first  moment vector
            features_adam_moment2 = th.zeros((Nfullpatch, self.d), dtype=self.dtype, device=self.device)#Initialize second moment vector
            adam_count = 1
        
        completed_patch_masses = th.ones(Nfullpatch, dtype=self.dtype, device=self.device)/Nfullpatch
        weight_mask_graph = completed_patch_masses[:, None] @ completed_patch_masses[None,:]
        weight_mask_features = th.diag(completed_patch_masses)
        completed_patch_graph[:Npatch, :Npatch] = patch_graph
        completed_patch_features[:Npatch, :] = patch_features
        init_completed_patch_graph = completed_patch_graph.clone()
        init_completed_patch_features = completed_patch_features.clone()
        
        curr_loss = th.tensor(1e15, dtype=self.dtype, device=self.device)
        best_loss = np.inf
        best_completed_patch_graph = completed_patch_graph.clone()
        best_completed_patch_features = completed_patch_features.clone()
        convergence_criterion = np.inf
        count = 0
        T_init = None
        
        while (convergence_criterion >= eps) and (count < max_iter):
            prev_loss = curr_loss.clone()
            #print('count :%s /curr_loss : %s'%(count,curr_loss))
            # compute transport between completed patch and dictionary
            local_OT, curr_loss = self.srFGW_operator(completed_patch_graph, completed_patch_features, completed_patch_masses, 
                                               self.Ctarget, self.Ftarget, T_init)
            if use_warmstart:
                T_init = local_OT.clone()
                
            if curr_loss < best_loss:
                best_loss  = curr_loss.item()
                best_completed_patch_graph = completed_patch_graph.clone()
                best_completed_patch_features = completed_patch_features.clone()
                
            # compute gradient to update the completed_patch
            if not use_optimizer:
                completed_patch_graph -= 2 *self.alpha * lrC * learnable_mask_graph * (completed_patch_graph *  weight_mask_graph - local_OT @ self.Ctarget @ local_OT.T)
                completed_patch_features -= 2 * (1 - self.alpha) * lrF * learnable_mask_features * (weight_mask_features @ completed_patch_features - local_OT @ self.Ftarget)
            
            else:
                graph_grad = 2 * self.alpha * learnable_mask_graph *(completed_patch_graph * weight_mask_graph - local_OT @ self.Ctarget @ local_OT.T)
                graph_m1_t = beta_1 * graph_adam_moment1 + (1 - beta_1) * graph_grad
                graph_m2_t = beta_2 * graph_adam_moment2 + (1 - beta_2) * (graph_grad**2)
                graph_m1_t_unbiased = graph_m1_t / (1 - beta_1**adam_count)
                graph_m2_t_unbiased = graph_m2_t /(1 - beta_2**adam_count)
                completed_patch_graph -= lrC * graph_m1_t_unbiased / (th.sqrt(graph_m2_t_unbiased)+1e-15)
                graph_adam_moment1 = graph_m1_t
                graph_adam_moment2 = graph_m2_t
                
                features_grad = 2 * (1 - self.alpha) * learnable_mask_features * (weight_mask_features @ completed_patch_features - local_OT @ self.Ftarget)
                features_m1_t = beta_1 * features_adam_moment1 + (1 - beta_1) * features_grad
                features_m2_t = beta_2 * features_adam_moment2 + (1 - beta_2) * (features_grad**2)
                features_m1_t_unbiased = features_m1_t / (1 - beta_1**adam_count)
                features_m2_t_unbiased = features_m2_t / (1 - beta_2**adam_count)
                completed_patch_features -= lrF * features_m1_t_unbiased / (th.sqrt(features_m2_t_unbiased)+1e-15)
                features_adam_moment1 = features_m1_t
                features_adam_moment2 = features_m2_t
                adam_count += 1
                
            if proj == 'nsym':
                completed_patch_graph[completed_patch_graph < 0.] = 0.
            if prev_loss.item() != 0:
                convergence_criterion = abs(prev_loss.item() -curr_loss.item())/ abs(prev_loss.item())
            else:
                convergence_criterion = abs(prev_loss.item() -curr_loss.item())/ abs(prev_loss.item() + 1e-15)
            count+=1
            
            if use_log:
                local_log['loss'].append(curr_loss.item())
            
        return best_completed_patch_graph, best_completed_patch_features, best_loss, local_log, init_completed_patch_graph, init_completed_patch_features
    
    def save_elements(self, save_settings=False, use_checkpoint = False):
        """
            DL saver used while learning the srGW graph atom. [ TO DO WITH PICKLE]
        """
        path = os.path.abspath('../')+self.experiment_repo
        print('path',path)
        if not os.path.exists(path+self.experiment_name):
            os.makedirs(path+self.experiment_name)
            print('made dir', path+self.experiment_name)
        if not use_checkpoint:
            pickle.dump(self.Ctarget, open(path+'%s/Ctarget.pkl'%self.experiment_name,'wb'))
            pickle.dump(self.Ftarget, open(path+'%s/Ftarget.pkl'%self.experiment_name,'wb'))

        else:
            self.checkpoint_Ctarget.append(self.Ctarget.clone())
            self.checkpoint_Ftarget.append(self.Ftarget.clone())
            print('#checkpoints:', len(self.checkpoint_Ctarget))

            pickle.dump(self.checkpoint_Ctarget, open(path+'%s/checkpoint_Ctarget.pkl'%self.experiment_name,'wb'))
            pickle.dump(self.checkpoint_Ftarget, open(path+'%s/checkpoint_Ftarget.pkl'%self.experiment_name,'wb'))

        for key in self.log.keys():
            np.save(path+'%s/%s.npy'%(self.experiment_name,key), np.array(self.log[key]))
            
        if save_settings:
            pd.DataFrame(self.settings, index=self.settings.keys()).to_csv(path+'%s/settings'%self.experiment_name)

            
    def load_elements(self, use_checkpoint=False):
        """
            DL loader for analysing learned dictionaries.
        """
        path = os.path.abspath('../')+self.experiment_repo
        if not use_checkpoint:
            self.Ctarget = pickle.load(open(path+'%s/Ctarget.pkl'%self.experiment_name, 'rb'))
            self.Ftarget = pickle.load(open(path+'%s/Ftarget.pkl'%self.experiment_name, 'rb'))
            self.Ntarget = self.Ctarget.shape[0]
        else:
            self.checkpoint_Ctarget = pickle.load(open(path+'%s/checkpoint_Ctarget.pkl'%self.experiment_name, 'rb'))
            self.checkpoint_Ftarget = pickle.load(open(path+'%s/checkpoint_Ftarget.pkl'%self.experiment_name, 'rb'))
            self.Ntarget = self.checkpoint_Ctarget[0].shape[-1]

            
            