
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as sklearn_split
from GW_utils import utils

#%%

def get_srGW_embedded_graphs(C, unmixings):
    embedded_graphs = []
    embedded_masses = []
    for h in unmixings:
        kept_idx = np.argwhere(h>0)[:,0]
        embC = C[kept_idx,:][:,kept_idx]
        embedded_graphs.append(embC)
        subh = h[kept_idx]
        # to avoid approximation errors
        subh /= np.sum(subh)
        embedded_masses.append(subh)
    return embedded_graphs, embedded_masses

def get_srFGW_embedded_graphs(C, F, unmixings, round_ = 12):
    embedded_graphs = []
    embedded_features = []
    embedded_masses = []
    for h in unmixings:
        kept_idx = np.argwhere(h>0)[:,0]
        embC = C[kept_idx,:][:,kept_idx]
        embF = F[kept_idx,:]
        embedded_graphs.append(embC)
        embedded_features.append(embF)
        subh = h[kept_idx]
        # to avoid approximation errors
        subh /= np.sum(subh)
        embedded_masses.append(subh)
    return embedded_graphs, embedded_features, embedded_masses


def Kernel_Matrix_precomputed(D:np.array,
                              gamma:float):
    
    return np.exp(-gamma*D)

def FGW_matrix(graphs,features, masses,alpha):
    """
        Compute pairwise FGW matrix 

    """
    n=len(graphs)
    D = np.zeros((n,n), dtype=np.float64)
   
    for i in tqdm(range(n-1)):
        for j in range (i+1, n):
            
            dist,T= utils.numpy_FGW_loss(C1=graphs[i], C2=graphs[j],A1=features[i],A2= features[j],p=masses[i], q=masses[j], alpha=alpha)
            D[i,j]= dist
            D[j,i]= dist
    return D

def GW_matrix(graphs,masses):
    """
        Compute pairwise GW matrix 
    """
    n= len(graphs)
    D = np.zeros((n,n), dtype=np.float64)

    for i in tqdm(range(n-1)):
        for j in range (i+1, n):
            
            dist,T= utils.np_GW2(C1=graphs[i],C2=graphs[j],p=masses[i],q=masses[j])
            D[i,j]= dist
            D[j,i]= dist
    return D


def euclidean_distance_matrix(X:np.array):
    ones_ = np.ones((X.shape[1], X.shape[0]))
    F1 = (X**2).dot(ones_)
    F2 = ones_.T.dot((X**2).T)
    return F1+F2 - 2*X.dot(X.T)
    
def nested_classification_SVC(D:np.array,
                            labels:np.array, 
                            n_folds:int=10, n_iter:int=10,verbose:bool=False):
    """
    Parameters
    ----------
    D : np.array (#observation, #observation)
        Pairwise distance matrix between observations.
    labels : np.array
        list of labels associated to the observations in the same order than used for computing D.
    n_folds : int, optional
        Number of folds used in the stratified cross validation of SVC hyperparameters. The default is 10.
    n_iter : int, optional
        Number of times to repeat the experiment with different random seed for (train/val/test) dataset splits. The default is 10.
    verbose : bool, optional
        To keep track of the learning process. The default is False.
    Returns
    -------
    res_best_svc : Pandas Dataframe containing results of the nested cross validation.
    """
    res_best_svc={'C':[], 'gamma':[],'val_mean_acc':[],'test_acc':[]}
    D[D<=10**(-15)]=0
    size= D.shape[0]
    assert len(labels)==size
    full_res_SVC = {'C':[],'gamma':[],'val_mean_acc':[],'n_iter':[]}
    end_index=0
    for i in tqdm(range(n_iter)): # do the nested CV
        #Stratified (n-folds) cross validation of Support vector machine hyperparameters
        start_index = end_index
        #print('i= %s / start_index = %s / end_index= %s'%(i,start_index,end_index))
        if verbose:
            print('n_iter:',i)
        k_fold=StratifiedKFold(n_splits=n_folds,random_state=i,shuffle=True)
        idx_train,idx_test,y_train,y_test=sklearn_split(np.arange(size),labels, test_size=0.1, stratify=labels, random_state=i)
        #res_SVC = {'C':[],'gamma':[],}
        
        for C in [10**x for x in range(-7,8)]:
            for gamma in [2**k for k in np.linspace(-10,10)]:
                end_index+=1
                local_mean_train = []
                local_mean_val = []
                for k,(idx_subtrain, idx_valid) in enumerate(k_fold.split(idx_train,y_train)):
                    if verbose:
                        print('fold:',k)
                    true_idx_subtrain=[idx_train[i] for i in idx_subtrain]
                    true_idx_valid=[idx_train[i] for i in idx_valid]
        
                    y_subtrain = np.array([labels[i] for i in true_idx_subtrain])
                    y_val=np.array([labels[i] for i in true_idx_valid])
                    
                    clf= SVC(C=C, kernel="precomputed",max_iter=5*10**6,random_state=0)
                    G_subtrain = Kernel_Matrix_precomputed(D[true_idx_subtrain,:][:,true_idx_subtrain],gamma=gamma)
                    if verbose:
                        print('check G_subtrain: sum/ nan / inf', G_subtrain.sum(), np.isnan(G_subtrain).sum(), (G_subtrain ==np.inf).sum())
                    clf.fit(G_subtrain,y_subtrain)
                    #print('n_iter_:', clf.n_iter_)
                    train_score= clf.score(G_subtrain,y_subtrain)
                    G_val = Kernel_Matrix_precomputed(D[true_idx_valid, :][:,true_idx_subtrain],gamma=gamma)
                    if verbose: 
                        print('check G_val: sum/ nan / inf', G_val.sum(), np.isnan(G_val).sum(), (G_val ==np.inf).sum())
                    val_score = clf.score(G_val,y_val)
                    local_mean_train.append(train_score)
                    local_mean_val.append(val_score)
                if verbose:
                    print('C:%s / gamma:%s / train: %s / val : %s'%(C,gamma,np.mean(local_mean_train), np.mean(local_mean_val)))
                full_res_SVC['C'].append(C)
                full_res_SVC['gamma'].append(gamma)
                full_res_SVC['val_mean_acc'].append(np.mean(local_mean_val))
                full_res_SVC['n_iter'].append(i)
                #res_SVC[(C,gamma)]=np.mean(local_mean_val)
        #print('i= %s / start_index = %s / end_index= %s'%(i,start_index,end_index))

        # Get best set of SVC hyperparameters on the validation dataset
        acc_values = full_res_SVC['val_mean_acc'][start_index:end_index]
        best_idx = np.argmax(acc_values)
        relocated_best_idx = np.arange(start_index,end_index)[best_idx]
        #best_idx = np.argmax(list(res_SVC.values()))
        #best_key = list(res_SVC.keys())[best_idx]
        #res_best_svc['C'].append(best_key[0])
        #res_best_svc['gamma'].append(best_key[1])
        #res_best_svc['val_mean_acc'].append(res_SVC[best_key])
        best_C = full_res_SVC['C'][relocated_best_idx]
        best_gamma = full_res_SVC['gamma'][relocated_best_idx]
        res_best_svc['C'].append(best_C)
        res_best_svc['gamma'].append(best_gamma)
        res_best_svc['val_mean_acc'].append(acc_values[best_idx])
        
        #clf= SVC(C=best_key[0], kernel="precomputed",random_state=0)
        clf= SVC(C=best_C,kernel="precomputed",random_state=0)
        #G_train =Kernel_Matrix_precomputed(D[idx_train,:][:,idx_train],gamma=best_key[1])
        G_train =Kernel_Matrix_precomputed(D[idx_train,:][:,idx_train],gamma=best_gamma)
        if verbose: 
            print('check G_full_train: sum/ nan / inf', G_train.sum(), np.isnan(G_train).sum(), (G_train ==np.inf).sum())
                        
        clf.fit(G_train, y_train)
        #G_test = Kernel_Matrix_precomputed(D[idx_test,:][:,idx_train],gamma=best_key[1])
        G_test = Kernel_Matrix_precomputed(D[idx_test,:][:,idx_train],gamma=best_gamma)
        if verbose: 
            print('check G_test: sum/ nan / inf', G_test.sum(), np.isnan(G_test).sum(), (G_test ==np.inf).sum())
        res_best_svc['test_acc'].append(clf.score(G_test,y_test))
    print('done computing SVC on the graphs dataset')
    return pd.DataFrame(res_best_svc), pd.DataFrame(full_res_SVC)
