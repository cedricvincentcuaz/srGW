from data_handler.graph_class import Graph,wl_labeling
import networkx as nx
#from utils import per_section,indices_to_one_hot
from collections import defaultdict
import numpy as np
import math
import os
from tqdm import tqdm
import pickle
import pandas as pd
#%%
def indices_to_one_hot(number, nb_classes,label_dummy=-1):
    """Convert an iterable of indices to one-hot encoded labels."""
    
    if number==label_dummy:
        return np.zeros(nb_classes)
    else:
        return np.eye(nb_classes)[number]


def per_section(it, is_delimiter=lambda x: x.isspace()):
    ret = []
    for line in it:
        if is_delimiter(line):
            if ret:
                yield ret  # OR  ''.join(ret)
                ret = []
        else:
            ret.append(line.rstrip())  # OR  ret.append(line)
    if ret:
        yield ret
        
def data_streamer(data_path,batchsize_bylabel, selected_labels,balanced_shapes=False,sampling_seed=None,return_idx = False):
    batch_graphs, batch_labels = [],[]
    if not (sampling_seed is None):
        np.random.seed(sampling_seed)
    if return_idx:
        batch_idx=[]
    if not balanced_shapes:
        for label in selected_labels:
            files = os.listdir(data_path+'/label%s/'%label)
            
            file_idx = np.random.choice(range(len(files)), size=batchsize_bylabel,replace=False)
            for idx in file_idx:
                    
                batch_graphs.append(np.load(data_path+'/label%s/'%label+files[idx]))
                batch_labels.append(label)
                if return_idx:
                    ls = file_idx[idx].split('.')
                    batch_idx.append(int(ls[0][:5]))
        if return_idx:
            return batch_graphs,batch_labels,batch_idx
        else:
            return batch_graphs,batch_labels
    else:
        shapes={}
        graphidx_shapes={}
        for label in selected_labels:
            files = os.listdir(data_path+'/label%s/'%label)
            shapes[label]=[]
            graphidx_shapes[label]=[]
            print('label = ', label)
            for filename in tqdm(files):
                local_idx = int(filename.split('.')[0][5:])
                graphidx_shapes[label].append(local_idx)
                shapes[label].append(np.load(data_path+'/label%s/'%label+filename).shape[0])
            unique_shapes= np.unique(shapes[label])
            sizebylabel = batchsize_bylabel//len(unique_shapes)
            for local_shape in unique_shapes:
                local_idx_list = np.argwhere(shapes[label]==local_shape)[:,0]
                sampled_idx = np.random.choice(local_idx_list, size=sizebylabel, replace=False)
                for idx in sampled_idx:
                    graphidx = graphidx_shapes[label][idx]
                    batch_graphs.append(np.load(data_path+'/label%s/graph%s.npy'%(label,graphidx)))
                    batch_labels.append(label)
                    
        return batch_graphs,batch_labels
    
def load_local_data(data_path,name,one_hot=False,attributes=True,use_node_deg=False):
    """ Load local datasets - modified version
    Parameters
    ----------
    data_path : string
                Path to the data. Must link to a folder where all datasets are saved in separate folders
    name : string
           Name of the dataset to load. 
           Choices=['mutag','ptc','nci1','imdb-b','imdb-m','enzymes','protein','protein_notfull','bzr','cox2','synthetic','aids','cuneiform'] 
    one_hot : integer
              If discrete attributes must be one hotted it must be the number of unique values.
    attributes :  bool, optional
                  For dataset with both continuous and discrete attributes. 
                  If True it uses the continuous attributes (corresponding to "Node Attr." in [5])
    use_node_deg : bool, optional
                   Wether to use the node degree instead of original labels. 
    Returns
    -------
    X : array
        array of Graph objects created from the dataset
    y : array
        classes of each graph    
    References
    ----------    
    [5] Kristian Kersting and Nils M. Kriege and Christopher Morris and Petra Mutzel and Marion Neumann 
        "Benchmark Data Sets for Graph Kernels"
    """
    name_to_path_discretefeatures={'mutag':data_path+'/MUTAG_2/',
                                   'ptc':data_path+'/PTC_MR/',
                                   'triangles':data_path+'/TRIANGLES/'}
    name_to_path_realfeatures={'enzymes':data_path+'/ENZYMES_2/',
                               'protein':data_path+'/PROTEINS_full/',
                               'protein_notfull':data_path+'/PROTEINS/',
                               'bzr':data_path+'/BZR/',
                               'cox2':data_path+'/COX2/'}
    name_to_rawnames={'mutag':'MUTAG', 'ptc':'PTC_MR','triangles':'TRIANGLES',
                      'enzymes':'ENZYMES','protein':'PROTEINS_full','protein_notfull':'PROTEINS',
                      'bzr':'BZR','cox2':'COX2',
                      'imdb-b':'IMDB-BINARY', 'imdb-m':'IMDB-MULTI','reddit':'REDDIT-BINARY','collab':'COLLAB'}
    if name in ['mutag','ptc','triangles']:
        dataset = build_dataset_discretefeatures(name_to_rawnames[name],
                                                 name_to_path_discretefeatures[name],
                                                 one_hot=one_hot)
    elif name in ['enzymes','protein', 'protein_notfull','bzr','cox2']:
        dataset = build_dataset_realfeatures(name_to_rawnames[name], name_to_path_realfeatures[name],
                                             type_attr='real',use_node_deg=use_node_deg)
    elif name in ['imdb-b','imdb-m','reddit', 'collab']:
        rawname  = name_to_rawnames[name]
        dataset = build_dataset_withoutfeatures(rawname, data_path+'/%s/'%rawname,use_node_deg= use_node_deg)
    else:
        raise 'unknown dataset'
    X,y=zip(*dataset)
    return np.array(X),np.array(y)
    
def build_noisy_circular_graph(N=20,mu=0,sigma=0.3,with_noise=False,structure_noise=False,p=None):
    g=Graph()
    g.add_nodes(list(range(N)))
    for i in range(N):
        noise=float(np.random.normal(mu,sigma,1))
        if with_noise:
            g.add_one_attribute(i,math.sin((2*i*math.pi/N))+noise)
        else:
            g.add_one_attribute(i,math.sin(2*i*math.pi/N))
        g.add_edge((i,i+1))
        if structure_noise:
            randomint=np.random.randint(0,p)
            if randomint==0:
                if i<=N-3:
                    g.add_edge((i,i+2))
                if i==N-2:
                    g.add_edge((i,0))
                if i==N-1:
                    g.add_edge((i,1))
    g.add_edge((N,0))
    noise=float(np.random.normal(mu,sigma,1))
    if with_noise:
        g.add_one_attribute(N,math.sin((2*N*math.pi/N))+noise)
    else:
        g.add_one_attribute(N,math.sin(2*N*math.pi/N))
    return g

def load_largegraphs(data_path, dataset_name,undirected=True):
    abspath = os.path.abspath('./')
    name_to_file = {'EU':'eu-email.p',
                    'village':'India_database.p',
                    'amazon':'amazon.p',
                    'wikicats':'wikicats.p'}
    database = pickle.load(open(abspath+data_path+name_to_file[dataset_name],'rb'))
    if not undirected:# directed graphs we could experimentally switch to undirected graphs
        assert dataset_name in ['EU','wikicats']
        if dataset_name in ['EU','wikicats']:
            G = nx.to_numpy_array(database['G'])
            node_labels = database['labels']
        else:
            raise 'unknown dataset name'
    else: #undirected graphs
        assert dataset_name in ['EU','amazon','wikicats', 'village']
        if dataset_name in ['EU','amazon', 'wikicats']:
            G = nx.to_numpy_array(database['G'].to_undirected())
            node_labels = database['labels']
        
        elif dataset_name in ['village']:
            node_labels = database['label']
            num_nodes = len(node_labels)
            G_ = nx.Graph()
            for i in range(num_nodes):
                G_.add_node(i)
            for edge in database['edges']:
                G_.add_edge(edge[0], edge[1])
            G= nx.adjacency_matrix(G_).toarray()
        else:
            raise 'unknown dataset name'
    return G, node_labels
#%%


def histog(X,bins=10):
    node_length=[]
    for graph in X:
        node_length.append(len(graph.nodes()))
    return np.array(node_length),{'histo':np.histogram(np.array(node_length),bins=bins),'med':np.median(np.array(node_length))
            ,'max':np.max(np.array(node_length)),'min':np.min(np.array(node_length))}

def node_labels_dic(path,name):
    node_dic=dict()
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            node_dic[k]=int(elt)
            k=k+1
    return node_dic

def node_attr_dic(path,name):
    node_dic=dict()
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            node_dic[k]=[float(x) for x in elt.split(',')]
            k=k+1
    return node_dic

def graph_label_list(path,name,real=False):
    graphs=[]
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            if real:
                graphs.append((k,float(elt)))
            else:
                graphs.append((k,int(elt)))
            k=k+1
    return graphs
    
def graph_indicator(path,name):
    data_dict = defaultdict(list)
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            data_dict[int(elt)].append(k)
            k=k+1
    return data_dict

def compute_adjency(path,name):
    adjency= defaultdict(list)
    with open(path+name) as f:
        sections = list(per_section(f))
        for elt in sections[0]:
            adjency[int(elt.split(',')[0])].append(int(elt.split(',')[1]))
    return adjency


def all_connected(X):
    a=[]
    for graph in X:
        a.append(nx.is_connected(graph.nx_graph))
    return np.all(a)

#%% TO FACTORIZE !!!!!!!!!!!

def build_dataset_discretefeatures(dataset_name,path,one_hot=False):
    assert dataset_name in ['MUTAG','PTC_MR','TRIANGLES']
    name_to_ncategories={'MUTAG':7, 'PTC_MR':18}
    n_categories = name_to_ncategories[dataset_name]
    graphs=graph_label_list(path,'%s_graph_labels.txt'%dataset_name)
    adjency=compute_adjency(path,'%s_A.txt'%dataset_name)
    data_dict=graph_indicator(path,'%s_graph_indicator.txt'%dataset_name)
    node_dic=node_labels_dic(path,'%s_node_labels.txt'%dataset_name) 
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if one_hot:
                attr=indices_to_one_hot(node_dic[node],n_categories)
                g.add_one_attribute(node,attr)
            else:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data



def build_dataset_realfeatures(dataset_name,path,type_attr='label',use_node_deg=False):
    assert dataset_name in ['PROTEINS_full','PROTEINS','ENZYMES','BZR','COX2']
    if type_attr=='label':
        node_dic=node_labels_dic(path,'%s_node_labels.txt'%dataset_name)
    if type_attr=='real':
        node_dic=node_attr_dic(path,'%s_node_attributes.txt'%dataset_name)
    graphs=graph_label_list(path,'%s_graph_labels.txt'%dataset_name)
    adjency=compute_adjency(path,'%s_A.txt'%dataset_name)
    data_dict=graph_indicator(path,'%s_graph_indicator.txt'%dataset_name)
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data


def build_dataset_withoutfeatures(dataset_name, path, use_node_deg=False):
    assert dataset_name in ['IMDB-MULTI','IMDB-BINARY','REDDIT-BINARY','COLLAB']
    graphs=graph_label_list(path,'%s_graph_labels.txt'%dataset_name)
    adjency=compute_adjency(path,'%s_A.txt'%dataset_name)
    data_dict=graph_indicator(path,'%s_graph_indicator.txt'%dataset_name)
    data=[]
    for i in tqdm(graphs,desc='loading graphs'):
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            #g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data

#%% READ EXPERIMENT RESULTS
def reader_results_FGWdictionary(dataset_name:str, 
                         str_selection:list,
                         excluded_str_selection:list,
                         unmixing_validation:bool=False,
                         parameters:list =['Ntarget','lrC','lrF','init_mode_graph','batch_size','algo_seed','l2_reg'],
                         aggreg_params='algo_seed',
                         compute_statistics=True, target_resfile= 'res_clustering.csv', target_unmixingsfile= 'unmixings.npy'):
    abs_path = os.path.abspath('../results/')
    full_path ='%s/%s/'%(abs_path,dataset_name)
    list_experiments = []
    res = {}
    for p in parameters:
        res[p]=[]
    res['loss'], res['RI'], res['best_RI']= [],[],[]
    res['init_features']=[]
    if compute_statistics:
        res['involved_components']=[]
        res['mean_components']=[]
        res['min_components']=[]
        res['max_components']=[]
    for subrepo in os.listdir(full_path):
        if np.all([str_ in subrepo for str_ in str_selection]) and (not np.any([str_ in subrepo for str_ in excluded_str_selection])):
            local_path='%s/%s/'%(full_path,subrepo)
            try:
            # load necessary files
                settings = pd.read_csv(local_path+'/settings')
                if not  ('_seed' in subrepo):
                    if not unmixing_validation:
                        local_res = pd.read_csv(local_path+'/res_clustering.csv')
                        if compute_statistics:
                            local_OT = pickle.load(open(local_path+'/OT_unmixings.pkl','rb'))
                    else:
                        local_res=pd.read_csv(local_path +'/res_clustering_100seeds.csv')
                        if compute_statistics:
                            local_OT = pickle.load(open(local_path+'/OT_unmixings_100seeds.pkl','rb'))
                    #print('local_res:', local_res)
                    #complete the summary dictionary
                    for p in parameters:
                        if p =='alpha':
                            for x in subrepo.split('_'):
                                if 'alpha' in x:
                                    res['alpha'].append(np.float(x[5:]))
                        elif p in ['gamma_entropy','lambda_reg']:
                            if p in settings.keys():
                                res[p].append(settings[p].iloc[0])
                            else:
                                res[p].append(0)
                        else:
                            res[p].append(settings[p].iloc[0])
                    best_idx_dist = np.argmin(local_res['loss_mean'].values)
                    res['loss'].append(local_res['loss_mean'].values[best_idx_dist])
                    res['RI'].append(local_res['RI'].values[best_idx_dist])
                    res['best_RI'].append(np.max(local_res['RI'].values))
                    if compute_statistics:
                        unmixings = np.array([np.sum(T,axis=0) for T in local_OT[best_idx_dist]])
                        sums=np.sum(unmixings,axis=0)
                        res['involved_components'].append(np.sum(sums>10**(-15)))
                        count_components = [np.sum(x>10**(-15)) for x in unmixings]
                        res['mean_components'].append(np.mean(count_components))
                        res['max_components'].append(np.max(count_components))
                        res['min_components'].append(np.min(count_components))
                    if 'Finitkmeans' in subrepo:
                        res['init_features'].append('kmeans')
                    elif 'Finitrange' in subrepo:
                        res['init_features'].append('range')
                    else:
                        res['init_features'].append('random')
                else:# we changed the storage method because it was too memory intensive
                    if not unmixing_validation:
                        local_res = pd.read_csv(local_path+target_resfile)
                        if compute_statistics:
                            unmixings = np.load(local_path+target_unmixingsfile)
                    else:
                        local_res=pd.read_csv(local_path +'/res_clustering_100seeds.csv')
                        if compute_statistics:
                            unmixings = np.load(local_path+'/unmixings_100seeds.npy')
                    #print('local_res:', local_res)
                    #complete the summary dictionary
                    for p in parameters:
                        if p =='alpha':
                            for x in subrepo.split('_'):
                                if 'alpha' in x:
                                    res['alpha'].append(np.float(x[5:]))
                        elif p=='use_warmstart':
                            if not p in settings.keys():
                                res[p].append(False)
                            else:
                                res[p].append(settings[p].iloc[0])
                        elif p in ['gamma_entropy','lambda_reg']:
                            if p in settings.keys():
                                res[p].append(settings[p].iloc[0])
                            else:
                                res[p].append(0)
                        else:
                            res[p].append(settings[p].iloc[0])
                    best_idx_dist = np.argmin(local_res['loss_mean'].values)
                    res['loss'].append(local_res['loss_mean'].values[best_idx_dist])
                    res['RI'].append(local_res['RI'].values[best_idx_dist])
                    res['best_RI'].append(np.max(local_res['RI'].values))
                    if compute_statistics:
                        sums=np.sum(unmixings[best_idx_dist],axis=0)
                        res['involved_components'].append(np.sum(sums>10**(-15)))
                        count_components = [np.sum(x>10**(-15)) for x in unmixings[best_idx_dist]]
                        res['mean_components'].append(np.mean(count_components))
                        res['max_components'].append(np.max(count_components))
                        res['min_components'].append(np.min(count_components))
                    if 'Finitkmeans' in subrepo:
                        res['init_features'].append('kmeans')
                    elif 'Finitrange' in subrepo:
                        res['init_features'].append('range')
                    else:
                        res['init_features'].append('random')
                list_experiments.append(subrepo)
            except:
                continue
    for key in res.keys():
        print('key: %s / len res: %s'%(key,len(res[key])))
    stacked_df = pd.DataFrame(res)
    
    print('stacked_df built ! shape: ', stacked_df.shape)
    
    aggreg_dict = {}
    fixed_params= []
    exception_keys = ['RI','best_RI','loss']
    if compute_statistics:
        exception_keys+=['max_components', 'mean_components', 'min_components','involved_components']
    
    for key in res.keys():
        if not key in exception_keys+[aggreg_params]:
            aggreg_dict[key]=list(np.unique(res[key]))
            fixed_params.append(key)
    print('fixed params:', fixed_params)
    aggreg_df_instantiated = False
    idx_to_explore = list(range(stacked_df.shape[0]))
    first_key = fixed_params[0]
    count =0
    nan_count=0
    mean_mapper = {}
    std_mapper= {}
    for key in exception_keys:
        mean_mapper[key]= 'mean_%s'%key
        std_mapper[key]= 'std_%s'%key
    while idx_to_explore !=[]:
        if count ==0:
            print('len idx_to_explore:', len(idx_to_explore))
            print('selected_idx:', idx_to_explore[0])
        selected_exp = stacked_df.iloc[idx_to_explore[0]]
        sub_df = stacked_df[stacked_df[first_key]==selected_exp[first_key]]
        for param in fixed_params[1:]:
            sub_df= sub_df[sub_df[param]==selected_exp[param]]
            if count ==0:
                print('param: %s / sub_df shape: %s'%(param,sub_df.shape))
            
        if not aggreg_df_instantiated:
            mean_aggreg_df = sub_df[exception_keys].mean(axis=0).to_frame().T
            std_aggreg_df = sub_df[exception_keys].std(axis=0).to_frame().T
            mean_aggreg_df.rename(mean_mapper,axis=1,inplace=True)
            std_aggreg_df.rename(std_mapper,axis=1,inplace=True)            
            for key in fixed_params:
                mean_aggreg_df[key] = sub_df[key].iloc[0]
                std_aggreg_df[key] = sub_df[key].iloc[0]
            #print('aggreg_df(n_exp=%s) - shape :'%n_exp,aggreg_df.shape)
            aggreg_df_instantiated = True
        else:
            mean_local_df = sub_df[exception_keys].mean(axis=0).to_frame().T
            std_local_df = sub_df[exception_keys].std(axis=0).to_frame().T
            mean_local_df.rename(mean_mapper,axis=1,inplace=True)
            std_local_df.rename(std_mapper,axis=1,inplace=True)            
            for key in fixed_params:
                try:
                    mean_local_df[key] = sub_df[key].iloc[0]
                    std_local_df[key] = sub_df[key].iloc[0]
                except:
                    nan_count+=1
                    mean_local_df[key] = np.nan
                    std_local_df[key] = np.nan
                    #raise 'empty df error'
                    continue
            mean_aggreg_df = pd.concat([mean_aggreg_df.copy(),mean_local_df.copy()])
            std_aggreg_df = pd.concat([std_aggreg_df.copy(),std_local_df.copy()])
            
        if count ==0:
            print('sub_df.index:', sub_df.index)
        for idx in sub_df.index.to_list():
            if count ==0:
                print('removed_idx:', idx)
            idx_to_explore.remove(idx)
        count+=1
    print('mean_aggreg_df: %s / std_aggreg_df: %s'%(mean_aggreg_df.shape,std_aggreg_df.shape))
    aggreg_df = pd.merge(mean_aggreg_df,std_aggreg_df)
    return stacked_df,aggreg_df, list_experiments
    
def reader_results_GWdictionary(dataset_name:str, 
                         str_selection:list,
                         excluded_str_selection:list,
                         unmixing_validation:bool=False,
                         parameters:list =['Ntarget','lr','init_mode_graph','batch_size','algo_seed','l2_reg'],
                         aggreg_params:str='algo_seed',
                         compute_statistics:bool=True,
                         target_resfile:str ='res_clustering.csv', 
                         target_unmixingsfile:str= 'unmixings.npy' ,
                         verbose:bool=False):
    abs_path = os.path.abspath('../results/')
    full_path ='%s/%s/'%(abs_path,dataset_name)
    list_experiments = []
    res = {}
    for p in parameters:
        res[p]=[]
    res['loss'], res['RI'], res['best_RI']= [],[],[]
    if compute_statistics:
        res['involved_components']=[]
        res['mean_components']=[]
        res['min_components']=[]
        res['max_components']=[]
    for subrepo in os.listdir(full_path):
        if np.all([str_ in subrepo for str_ in str_selection]) and (not np.any([str_ in subrepo for str_ in excluded_str_selection])):
            local_path='%s/%s/'%(full_path,subrepo)
            try:
                # load necessary files
                settings = pd.read_csv(local_path+'/settings')
                if not unmixing_validation:
                    local_res = pd.read_csv(local_path+target_resfile)
                    if compute_statistics:  
                        unmixings = np.load(local_path+target_unmixingsfile)
                else:
                    local_res=pd.read_csv(local_path +'/res_clustering_100seeds.csv')
                    if compute_statistics:  
                        unmixings = np.load(local_path+'/unmixings_100seeds.npy')
                #print('local_res:', local_res)
                #complete the summary dictionary
                for p in parameters:
                    if p in ['use_warmstart','gamma_entropy','lambda_reg']:
                        if not p in settings.keys():
                            if p=='use_warmstart':
                                res[p].append(False)
                            else:
                                res[p].append(0)
                        else:
                            res[p].append(settings[p].iloc[0])
                    
                    else:
                        res[p].append(settings[p].iloc[0])
                best_idx_dist = np.argmin(local_res['loss_mean'].values)
                res['loss'].append(local_res['loss_mean'].values[best_idx_dist])
                res['RI'].append(local_res['RI'].values[best_idx_dist])
                res['best_RI'].append(np.max(local_res['RI'].values))
                if compute_statistics:
                    sums=np.sum(unmixings[best_idx_dist],axis=0)
                    res['involved_components'].append(np.sum(sums>10**(-15)))
                    count_components = [np.sum(x>10**(-15)) for x in unmixings[best_idx_dist]]
                    res['mean_components'].append(np.mean(count_components))
                    res['max_components'].append(np.max(count_components))
                    res['min_components'].append(np.min(count_components))
                    
                list_experiments.append(subrepo)
            except:
                if verbose:
                    print('FAILED FULLY LOADING EXPERIMENT: ', subrepo)
                continue
            #except:
            #    continue
    for key in res.keys():
        print('key: %s / len res: %s'%(key,len(res[key])))
    stacked_df = pd.DataFrame(res)
    
    print('stacked_df built ! shape: ', stacked_df.shape)
    
    aggreg_dict = {}
    fixed_params= []
    exception_keys = ['RI','best_RI','loss']
    if compute_statistics:
        exception_keys+=['max_components', 'mean_components', 'min_components','involved_components']
    
    for key in res.keys():
        if not key in exception_keys+[aggreg_params]:
            aggreg_dict[key]=list(np.unique(res[key]))
            fixed_params.append(key)
    print('fixed params:', fixed_params)
    aggreg_df_instantiated = False
    idx_to_explore = list(range(stacked_df.shape[0]))
    first_key = fixed_params[0]
    count =0
    nan_count=0
    mean_mapper = {}
    std_mapper= {}
    for key in exception_keys:
        mean_mapper[key]= 'mean_%s'%key
        std_mapper[key]= 'std_%s'%key
    while idx_to_explore !=[]:
        if count ==0:
            print('len idx_to_explore:', len(idx_to_explore))
            print('selected_idx:', idx_to_explore[0])
        selected_exp = stacked_df.iloc[idx_to_explore[0]]
        sub_df = stacked_df[stacked_df[first_key]==selected_exp[first_key]]
        for param in fixed_params[1:]:
            sub_df= sub_df[sub_df[param]==selected_exp[param]]
            if count ==0:
                print('param: %s / sub_df shape: %s'%(param,sub_df.shape))
            
        if not aggreg_df_instantiated:
            mean_aggreg_df = sub_df[exception_keys].mean(axis=0).to_frame().T
            std_aggreg_df = sub_df[exception_keys].std(axis=0).to_frame().T
            mean_aggreg_df.rename(mean_mapper,axis=1,inplace=True)
            std_aggreg_df.rename(std_mapper,axis=1,inplace=True)            
            for key in fixed_params:
                mean_aggreg_df[key] = sub_df[key].iloc[0]
                std_aggreg_df[key] = sub_df[key].iloc[0]
            #print('aggreg_df(n_exp=%s) - shape :'%n_exp,aggreg_df.shape)
            aggreg_df_instantiated = True
        else:
            mean_local_df = sub_df[exception_keys].mean(axis=0).to_frame().T
            std_local_df = sub_df[exception_keys].std(axis=0).to_frame().T
            mean_local_df.rename(mean_mapper,axis=1,inplace=True)
            std_local_df.rename(std_mapper,axis=1,inplace=True)            
            for key in fixed_params:
                try:
                    mean_local_df[key] = sub_df[key].iloc[0]
                    std_local_df[key] = sub_df[key].iloc[0]
                except:
                    nan_count+=1
                    mean_local_df[key] = np.nan
                    std_local_df[key] = np.nan
                    #raise 'empty df error'
                    continue
            mean_aggreg_df = pd.concat([mean_aggreg_df.copy(),mean_local_df.copy()])
            std_aggreg_df = pd.concat([std_aggreg_df.copy(),std_local_df.copy()])
            
        if count ==0:
            print('sub_df.index:', sub_df.index)
        for idx in sub_df.index.to_list():
            if count ==0:
                print('removed_idx:', idx)
            idx_to_explore.remove(idx)
        count+=1
    print('mean_aggreg_df: %s / std_aggreg_df: %s'%(mean_aggreg_df.shape,std_aggreg_df.shape))
    aggreg_df = pd.merge(mean_aggreg_df,std_aggreg_df)
    return stacked_df,aggreg_df, list_experiments


def reader_results_GWdenoising(dataset_name:str, 
                         str_selection:list,
                         parameters:list =[ 'Ntarget','batch_size','lr','init_mode_graph','l2_reg']):
    """
    For all denoising experiment in the repository satisfying str_selection
    get:
        - averaged loss on noisy graphs
        - averaged loss on true graphs
        - averaged number of components of the unmixings
        - the number of components involved
        - the min and max number of components
    """
    abs_path = os.path.abspath('../results/')
    full_path ='%s/%s/'%(abs_path,dataset_name)
    list_experiments = []
    res = {}
    for p in parameters:
        res[p]=[]
    res['n_trials'],res['n_samples'], res['noise_type']=[],[],[]
    res['noisy_loss'], res['true_loss'], res['val_noisy_loss'], res['val_true_loss']=[],[],[],[]
    res['involved_components'],res['mean_components'], res['min_components'], res['max_components']=[], [], [], []
    res['val_involved_components'],res['val_mean_components'], res['val_min_components'], res['val_max_components']=[], [], [], []
    
    for subrepo in os.listdir(full_path):
        if np.all([str_ in subrepo for str_ in str_selection]):
            local_path='%s/%s/'%(full_path,subrepo)
            try:
            # load necessary files
                settings = pd.read_csv(local_path+'/settings')
        
                noisy_OT = pickle.load(open(local_path+'/OT_unmixings_noisyG.pkl','rb'))
                noisy_losses = np.load(local_path+'/losses_unmixings_noisyG.npy')
                val_noisy_OT = pickle.load(open(local_path+'/OT_unmixings_noisyG_50seeds.pkl','rb'))
                val_noisy_losses = np.load(local_path+'/losses_unmixings_noisyG_50seeds.npy') 
                #true_OT = pickle.load(open(local_path+'/OT_unmixings_trueG.pkl','rb'))
                true_losses = np.load(local_path+'/losses_unmixings_sampledtrueG.npy')
                #val_true_OT = pickle.load(open(local_path+'/OT_unmixings_trueG_50seeds.pkl','rb'))
                val_true_losses = np.load(local_path+'/losses_unmixings_sampledtrueG_50seeds.npy')
                best_idx_loss = np.argmin([np.mean(losses ) for losses in noisy_losses])
                noisy_unmixings = np.array([np.sum(T,axis=0) for T in noisy_OT[best_idx_loss]])
                val_best_idx_loss = np.argmin([np.mean(losses ) for losses in val_noisy_losses])
                val_noisy_unmixings = np.array([np.sum(T,axis=0) for T in val_noisy_OT[val_best_idx_loss]])
                denoising_parameters = pickle.load(open(local_path+'/denoising_parameters.pkl','rb'))
                # intel on the type of denoising experiment
                res['n_trials'].append(denoising_parameters['n_trials'])
                res['n_samples'].append(denoising_parameters['n_samples'])
                if 'denoisingsymmetric' in subrepo:
                    res['noise_type'].append('symmetric')
                elif 'denoisingnegative' in subrepo:
                    res['noise_type'].append('negative')
                elif 'denoisingpositive' in subrepo:
                    res['noise_type'].append('positive')
                else:
                    raise 'unknown noise type'
                # parameters of the dictionary
                for p in parameters:
                    res[p].append(settings[p].iloc[0])
                # statistic on the embedding 
                sums=np.sum(noisy_unmixings,axis=0)
                res['involved_components'].append(np.sum(sums>10**(-15)))
                count_components = [np.sum(x>10**(-15)) for x in noisy_unmixings]
                res['mean_components'].append(np.mean(count_components))
                res['max_components'].append(np.max(count_components))
                res['min_components'].append(np.min(count_components))
                sums=np.sum(val_noisy_unmixings,axis=0)
                res['val_involved_components'].append(np.sum(sums>10**(-15)))
                count_components = [np.sum(x>10**(-15)) for x in val_noisy_unmixings]
                res['val_mean_components'].append(np.mean(count_components))
                res['val_max_components'].append(np.max(count_components))
                res['val_min_components'].append(np.min(count_components))
                res['noisy_loss'].append(np.mean(noisy_losses[best_idx_loss]))
                res['true_loss'].append(np.mean(true_losses[best_idx_loss]))
                res['val_noisy_loss'].append(np.mean(val_noisy_losses[val_best_idx_loss]))
                res['val_true_loss'].append(np.mean(val_true_losses[val_best_idx_loss]))
                
                list_experiments.append(subrepo)
            except:
                continue
    return res,list_experiments            
    