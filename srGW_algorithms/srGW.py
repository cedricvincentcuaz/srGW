"""
    @author: cvincentcuaz

    We mimic the gromov-wasserstein implementation of the 
    Python Optimal Transport Library (POT). With a slight enhancement of its
    0.8.0 version to handle asymmetric structures and to be able to provide
    initial transport plans different from the classical product measure.    

    ref: Rémi Flamary, Nicolas Courty, Alexandre Gramfort, Mokhtar Z. Alaya, 
    Aurélie Boisbunon, Stanislas Chambon, Laetitia Chapel, Adrien Corenflos, 
    Kilian Fatras, Nemo Fournier, Léo Gautheron, Nathalie T.H. Gayraud, 
    Hicham Janati, Alain Rakotomamonjy, Ievgen Redko, Antoine Rolet, Antony Schutz, 
    Vivien Seguy, Danica J. Sutherland, Romain Tavenard, Alexander Tong, Titouan Vayer,
    
    POT Python Optimal Transport library,
    Journal of Machine Learning Research, 22(78):1−8, 2021.
    Website: https://pythonot.github.io/        
    
    
"""

import numpy as np
import torch as th

# =============================================================================
# TORCH IMPLEMENTATION OF SEMI-RELAXED (FUSED) GROMOV-WASSERSTEIN - HANDLE CPU or GPU
#   1. Conditional gradient descent algorithms for srGW, srFGW
#   2. Mirror Descent algorithm for entropic srGW, srFGW
#   3. Majorization-Maximization algorithms for sparsity promoted srGW, srFGW, plus their entropic versions 
# =============================================================================

#%% utils

def initializer_semirelaxed_GW(init_mode, p, N1, N2, seed=0, dtype=th.float64, device='cuda:0'):
    if init_mode == 'product':
        q= th.ones(N2, dtype=dtype, device=device) / N2
        T= p[:, None] @ q[None, :]
    
    elif init_mode == 'random': 
        if not (seed is None):
            th.manual_seed(seed)
        T = th.rand(size=(N1, N2), dtype=dtype, device=device)
        # scaling to satisfy first marginal constraints
        scale = p / T.sum(axis=1)
        T *= scale[:, None]
    
    elif init_mode == 'random_product':
        if not (seed is None):
            th.manual_seed(seed)
        seed=None
        q = th.rand(size=N2,dtype=dtype, device=device)
        q /= q.sum()
        T = p[:, None] @ q[None, :]
    else:
        raise 'unknown init mode'
    return T

def init_matrix_GW2(C1, C2, p, q, ones_p, ones_q):
    f1_ , f2_ = C1 ** 2 , C2 ** 2
    constC1 = f1_ @ ( p[:, None] @ ones_q[None, :] )
    constC2 = (ones_p[:, None] @ q[None, :]) @ f2_
    constC = constC1 + constC2
    return constC, C1, 2*C2 


def init_matrix_asymGW2(C1, C2, p, q, ones_p, ones_q):
    f1_ , f2_ = C1**2/2. , C2**2/2.
    constC1 = f1_ @ ( p[:, None] @ ones_q[None, :] )
    constC2 = (ones_p[:, None] @ q[None, :]) @ f2_.T
    constC = constC1 + constC2
    return constC, C1, C2 

def tensor_product(constC, hC1, hC2, T):
    A = - hC1 @ T @ (hC2.T)
    tens = constC + A
    return tens



#%% Conditional gradient descent algorithms

# =============================================================================
# Contains:
#   - (cg_semirelaxed) CG for generalized cost of the form: \alpha * <L(C_1, C_2) \otimes T, T> + <M, T>
#   - (cg_semirelaxed_gromov_wasserstein) i.e \alpha = 1;  M = 0
#   - (cg_semirelaxed_fused_gromov_wasserstein) i.e \alpha = FGW trade-off parameter; M = (1 - alpha)* features distance matrix
# =============================================================================


def cg_semirelaxed(C1:th.Tensor,
                   p:th.Tensor,
                   C2:th.Tensor, 
                   alpha:float=1.,
                   linear_cost:th.Tensor=None,
                   init_mode:str='product',
                   T_init:th.Tensor=None,
                   symmetry:bool=True,
                   use_log:bool = False,
                   eps:float=10**(-5),
                   max_iter:int=1000,
                   seed:int=0,
                   verbose:bool=False,
                   device:str='cpu',
                   dtype:type=th.float32):
    """ 
        Conditional gradient algorithm for semi-relaxed (fused) gromov-wasserstein, optionally with a linear OT cost:
            
            \min_{T}   \alpha * <L(C_1, C_2) \otimes T, T>  + < M, T> 
        
        The implementation corresponds to a generalization of the Frank-Wolfe algorithm detailed
        in Algorithm 1. Section 3.2 of the main paper.
        This general form is discussed in Algorithm 3. of section 7.3.1 in the supplementary material.
    """
    N1 = C1.shape[0]
    N2 = C2.shape[0]
    
    if T_init is None:
        T= initializer_semirelaxed_GW(init_mode, p, N1, N2, seed=seed, dtype=dtype, device=device)
    else:
        assert T_init.shape == (N1,N2)  # shape constraints
        T = T_init.clone()
    
    if symmetry is None:
        symmetry = th.all( C1 == C1.T) and th.all(C2 == C2.T)
    # Get gradient from initial starting point
    q= T.sum(axis=0) 
    ones_p = th.ones(N1, dtype=dtype, device=device)
    ones_q = th.ones(N2, dtype=dtype, device=device)
    if symmetry:
        constC, hC1, hC2 = init_matrix_GW2(C1, C2, p, q, ones_p, ones_q)
        G = 2 * tensor_product(constC, hC1, hC2, T)
    else:
        constC, hC1, hC2 = init_matrix_asymGW2(C1, C2, p, q, ones_p, ones_q)
        constCt, hC1t, hC2t = init_matrix_asymGW2(C1.T, C2.T, p, q, ones_p, ones_q)            
        subG = tensor_product(constC, hC1, hC2, T)
        subGt = tensor_product(constCt, hC1t, hC2t,T)
        G = (subG + subGt)
    G *= alpha
    srgw_loss = 0.5 * th.sum(G * T)  # We consider as srgw_loss alpha* <L(C_1, C_2) \otimes T, T>
    add_linear_cost = not (linear_cost is None)
    if add_linear_cost:
        linear_loss = (linear_cost * T).sum()
        current_loss = srgw_loss + linear_loss
        G += linear_cost
    else:
        current_loss = srgw_loss
    #current_loss = f1
    if use_log:
        log={}
        log['loss'] = [current_loss.item()]
        
    convergence_criterion = np.inf
    outer_count=0

    while (convergence_criterion > eps) and (outer_count < max_iter):
        previous_loss = current_loss.clone()
        # 0. Gradient known from evaluation of the  cost function
        # 1. Direction finding by solving each subproblem on rows
        min_, _ = G.min(axis=1)
        X = (G == min_[:, None]).type(dtype)
        X *= (p / X.sum(axis=1))[:, None]
        # 3. Exact line-search step
        # Compute litteral expressions of coefficients a*\gamma^2 +b \gamma +c
        qX = X.sum(axis=0)
        if symmetry:
            constCX, hC1X, hC2X = init_matrix_GW2(C1, C2, p, qX, ones_p, ones_q)
            GX = 2 * alpha * tensor_product (constCX, hC1X, hC2X, X)  # we do not include the linear cost in this gradient wrt X
            GXX = 0.5 * (GX * X).sum()
            GXT = 0.5 * (GX * T).sum()
            
            a = srgw_loss + GXX - 2 * GXT
            b = 2 * (GXT - srgw_loss)
        else:
            constCX, hC1X, hC2X = init_matrix_asymGW2(C1, C2, p, qX)
            constCXt, hC1Xt, hC2Xt = init_matrix_asymGW2(C1.T, C2.T, p, qX)
            subGX = tensor_product(constCX, hC1X, hC2X, X)
            subGXt = tensor_product(constCXt, hC1Xt, hC2Xt, X)
            GX = alpha * (subGX + subGXt)  # we do not include the linear cost in this gradient wrt X
            GXX = 0.5 * (GX * X).sum()
            subGXt_dotT = (subGXt * T).sum() # \sum_ijkl (C_ij - Cbar_kl)^2 X_ik T_jl
            subGTt_dotX = (subGt * X).sum() # \sum_ijkl (C_ij - Cbar_kl)^2 T_ik X_jl
            a = srgw_loss + GXX - subGXt_dotT - subGTt_dotX
            b = - 2 * srgw_loss + subGXt_dotT + subGTt_dotX
        
        if add_linear_cost:
            linear_loss_X = (linear_cost * X).sum()
            b += linear_loss_X - linear_loss
            
        if a>0:
            gamma = min(1, max(0, -b.item()/(2*a.item())))
        elif a+b<0:
            gamma=1
        else:
            gamma=0
        T = (1 - gamma) * T + gamma * X 
        current_loss += a * (gamma **2) + b * gamma 
        if add_linear_cost:            
            linear_loss = (1 - gamma) * linear_loss + gamma * linear_loss_X
            srgw_loss = current_loss - linear_loss
            G = (1 - gamma) * G + gamma * (GX + linear_cost)
            
        else:
            srgw_loss = current_loss
            G = (1 - gamma) * G + gamma * GX
        outer_count+=1
        if use_log:
            log['loss'].append(current_loss.item())
        if previous_loss != 0:
            convergence_criterion = abs(previous_loss.item() - current_loss.item())/ abs(previous_loss.item())
        else:
            convergence_criterion = abs(previous_loss.item() - current_loss.item())/ abs(previous_loss.item() + 10 ** (- 15))
        
    if use_log:
        return T, current_loss, log
    else:
        return T, current_loss
    
def cg_semirelaxed_gromov_wasserstein(C1:th.Tensor,
                                      p:th.Tensor,
                                      C2:th.Tensor, 
                                      init_mode:str='product',
                                      T_init:th.Tensor=None,
                                      symmetry:bool=True,
                                      use_log:bool = False,
                                      eps:float=10**(-5),
                                      max_iter:int=1000,
                                      seed:int=0,
                                      verbose:bool=False,
                                      device:str='cpu',
                                      dtype:type=th.float32):
    """ 
        Conditional gradient algorithm for semi-relaxed gromov-wasserstein:
            
            \min_{T}   <L(C_1, C_2) \otimes T, T>
        
        The implementation uses the generalization of the Frank-Wolfe algorithm detailed
        in Algorithm 1. Section 3.2 of the main paper.
        This general form is discussed in Algorithm 3. of section 7.3.1 in the supplementary material.
        
        It comes down to consider:
            - linear_cost = 0 (corresponds to None)
            - alpha = 1.
    """
    return cg_semirelaxed(C1, p, C2, 1., None, init_mode, T_init, 
                          symmetry, use_log, eps, max_iter, seed,
                          verbose, device, dtype)


def cg_semirelaxed_fused_gromov_wasserstein(C1:th.Tensor,
                                            A1:th.Tensor,
                                            p:th.Tensor,
                                            C2:th.Tensor,
                                            A2:th.Tensor,
                                            alpha:float,
                                            symmetry:bool=True, 
                                            init_mode:str='product',
                                            T_init:th.Tensor=None,
                                            use_log:bool=False,
                                            eps:float=10**(-5),
                                            max_iter:int=1000,
                                            seed:int=0,
                                            verbose:bool=False,
                                            device:str='cpu',
                                            dtype:type=th.float32):
    """ 
        Conditional gradient algorithm for semi-relaxed fused gromov-wasserstein:
            
            \min_{T}   \alpha * <L(C_1, C_2) \otimes T, T> + (1-\alpha) * <D, T>
        
        The implementation uses the generalization of the Frank-Wolfe algorithm detailed
        in Algorithm 1. Section 3.2 of the main paper.
        This general form is discussed in Algorithm 3. of section 7.3.1 in the supplementary material.
        
        It comes down to consider:
            - linear_cost = (1-\alpha) * D 
            - alpha = \alpha
    """
    N1 = A1.shape[0]
    N2 = A2.shape[0]
    d = A1.shape[1]
    # Compute matrix of euclidean distances between features
    FS2 = (A1**2) @ th.ones((d, N2), dtype=dtype, device=device)
    FT2 = th.ones((N1, d), dtype=dtype, device=device) @ (A2**2).T
    D = FS2 + FT2 - 2 * A1 @ A2.T
    
    return cg_semirelaxed(C1, p, C2, alpha, (1 - alpha) * D, init_mode, T_init, 
                          symmetry, use_log, eps, max_iter, seed, verbose, device, dtype)

#%% mirror descent algorithms using KL geometry (entropic regularization)

# =============================================================================
# Contains:
#   - (md_semirelaxed) Mirror Descent for generalized cost of the form: \alpha * <L(C_1, C_2) \otimes T, T> + <M, T>
#   - (md_semirelaxed_gromov_wasserstein) i.e \alpha = 1;  M = 0
#   - (md_semirelaxed_fused_gromov_wasserstein) i.e \alpha = FGW trade-off parameter; M = (1 - alpha)* features distance matrix
# =============================================================================


def md_semirelaxed(C1:th.Tensor,
                   p:th.Tensor,
                   C2:th.Tensor, 
                   gamma_entropy:float,
                   alpha:float=1.,
                   linear_cost:th.Tensor=None,
                   init_mode:str='product',
                   T_init:th.Tensor=None,
                   symmetry:bool=True,
                   use_log:bool = False,
                   eps:float=10**(-5),
                   max_iter:int=1000,
                   seed:int=0,
                   verbose:bool=False,
                   device:str='cpu',
                   dtype:type=th.float32):
    
    """ 
        Mirror descent algorithm using KL geometry for semi-relaxed (fused) gromov-wasserstein, optionally with a linear OT cost:
            
            \min_{T}   \alpha * <L(C_1, C_2) \otimes T, T>  + < M, T> 
        
        The implementation corresponds to a generalization of the mirror-descent algorithm detailed
        in Section 3.2 of the main paper.
        This general form is discussed in Algorithm 4. of section 7.3.2 in the supplementary material.
    """
    assert gamma_entropy>0
    N1 = C1.shape[0]
    N2 = C2.shape[0]
    
    if T_init is None:
        T= initializer_semirelaxed_GW(init_mode, p, N1, N2, seed=seed, dtype=dtype, device=device)
    else:
        assert T_init.shape == (N1,N2)  # shape constraints
        T = T_init.clone()
    
    if symmetry is None:
        symmetry = th.all( C1 == C1.T) and th.all(C2 == C2.T)
    
    # Get gradient from initial starting point
    q = T.sum(axis=0)
    ones_p = th.ones(N1, dtype=dtype, device=device)
    ones_q = th.ones(N2, dtype=dtype, device=device)
    if symmetry:
        constC, hC1, hC2 = init_matrix_GW2(C1, C2, p, q, ones_p, ones_q)
        G = 2 * alpha * tensor_product(constC, hC1, hC2, T)
    else:
        constC, hC1, hC2 = init_matrix_asymGW2(C1, C2, p, q, ones_p, ones_q)
        constCt, hC1t, hC2t = init_matrix_asymGW2(C1.T, C2.T, p, q, ones_p, ones_q)            
        subG = tensor_product(constC, hC1, hC2, T)
        subGt = tensor_product(constCt, hC1t, hC2t,T)
        G = alpha * (subG + subGt)
        
    current_loss = 0.5 * (G * T).sum()
    add_linear_cost = not (linear_cost is None)
    if add_linear_cost:
        linear_loss = (linear_cost * T).sum()
        current_loss += linear_loss
        G += linear_cost
        
    if use_log:
        log={}
        log['loss']=[current_loss.item()]
    
    convergence_criterion = np.inf
    outer_count=0
    while (convergence_criterion > eps) and (outer_count < max_iter):
        previous_loss = current_loss
        #1. Compute M_k(epsilon) = 2\alpha (L(C1,C2) \otimes T_k) + M - gamma_entropie* log(T_k)
        # single Bregman projection
        M= G - gamma_entropy * th.log(T)
        K= th.exp( - M / gamma_entropy)
        scaling = p / K.sum(axis=1)
        T = th.diag(scaling) @ K
        q = T.sum(axis=0)
        if symmetry:
            constC, hC1, hC2 = init_matrix_GW2(C1, C2, p, q, ones_p, ones_q)
            G = 2 * alpha * tensor_product(constC, hC1, hC2, T)
        else:
            constC, hC1, hC2 = init_matrix_asymGW2(C1, C2, p, q, ones_p, ones_q)
            constCt, hC1t, hC2t = init_matrix_asymGW2(C1.T, C2.T, p, q, ones_p, ones_q)            
            subG = tensor_product(constC, hC1, hC2, T)
            subGt = tensor_product(constCt, hC1t, hC2t,T)
            G = alpha * (subG + subGt)
            
        current_loss = 0.5 * (G * T).sum()
        if add_linear_cost:
            linear_loss = (linear_cost * T).sum()
            current_loss += linear_loss
            G += linear_cost

        outer_count+=1
        if use_log:
            log['loss'].append(current_loss.item())
        if previous_loss != 0:
            convergence_criterion = abs(previous_loss.item() - current_loss.item())/ abs(previous_loss.item())
        else:
            convergence_criterion = abs(previous_loss.item() - current_loss.item())/ abs(previous_loss.item() + 1e-15)
       
    if use_log:
        return T, current_loss, log
    else:
        return T, current_loss


def md_semirelaxed_gromov_wasserstein(C1:th.Tensor,
                                      p:th.Tensor,
                                      C2:th.Tensor,
                                      gamma_entropy:float,
                                      init_mode:str='product',
                                      T_init:th.Tensor=None,
                                      symmetry:bool=True,
                                      use_log:bool = False,
                                      eps:float=10**(-5),
                                      max_iter:int=1000,
                                      seed:int=0,
                                      verbose:bool=False,
                                      device:str='cpu',
                                      dtype:type=th.float32):
    """ 
        Mirror descent algorithm using KL geometry for semi-relaxed gromov-wasserstein:
            
            \min_{T}   <L(C_1, C_2) \otimes T, T> 
        
        The implementation corresponds to a generalization of the mirror-descent algorithm detailed
        in Section 3.2 of the main paper.
        This general form is discussed in Algorithm 4. of section 7.3.2 in the supplementary material.
        For srGW, it comes down to consider:
            - linear_cost = 0 (corresponds to None)
            - alpha = 1.
    """
    return md_semirelaxed(C1, p, C2, gamma_entropy, 1., None,
                          init_mode, T_init, symmetry, use_log, 
                          eps, max_iter, seed, verbose, device, dtype)
        
def md_semirelaxed_fused_gromov_wasserstein(C1:th.Tensor,
                                            A1:th.Tensor,
                                            p:th.Tensor,
                                            C2:th.Tensor,
                                            A2:th.Tensor,
                                            gamma_entropy:float,
                                            alpha:float,
                                            symmetry:bool=True, 
                                            init_mode:str='product',
                                            T_init:th.Tensor=None,
                                            use_log:bool=False,
                                            eps:float=10**(-5),
                                            max_iter:int=1000,
                                            seed:int=0,
                                            verbose:bool=False,
                                            device:str='cpu',
                                            dtype:type=th.float32):
    """ 
        Mirror descent algorithm for semi-relaxed fused gromov-wasserstein:
            
            \min_{T}   \alpha * <L(C_1, C_2) \otimes T, T> + (1-\alpha) * <D, T>
        
        The implementation corresponds to a generalization of the mirror-descent algorithm detailed
        in Section 3.2 of the main paper.
        This general form is discussed in Algorithm 4. of section 7.3.2 in the supplementary material.
        For srFGW, it comes down to consider:

            - linear_cost = (1-\alpha) * D 
            - alpha = \alpha
    """
    N1 = A1.shape[0]
    N2 = A2.shape[0]
    d = A1.shape[1]
    # Compute matrix of euclidean distances between features
    FS2 = (A1**2) @ th.ones((d, N2))
    FT2 = th.ones((N1, d)) @ (A2**2).T
    D = FS2 + FT2 - 2 * A1 @ A2.T
    
    return md_semirelaxed(C1, p, C2, gamma_entropy, alpha, (1 - alpha) * D,
                          init_mode, T_init, symmetry, use_log, 
                          eps, max_iter, seed, verbose, device, dtype)



#%% MM algorithm using concave sparsity promoting regularization using group lasso like penalty on 
def mm_lpl1_semirelaxed(C1:th.Tensor,
                        p:th.Tensor,
                        C2:th.Tensor,
                        gamma_entropy:float,
                        alpha:float=1.,
                        linear_cost:th.Tensor=0.,
                        T_init:th.Tensor=None,
                        init_mode:str='product',
                        symmetry:bool=True,
                        p_reg:float=1/2,
                        lambda_reg:float = 0.001,
                        use_log:bool = False,
                        use_warmstart:bool=False,
                        eps_inner:float=10**(-6),
                        eps_outer:float=10**(-6),
                        max_iter_inner:int =1000,
                        max_iter_outer:int =50,
                        seed:int=0,
                        verbose:bool=False,
                        inner_log:bool = False,
                        dtype:type=th.float64,
                        device:str='cpu'):
    r""" 
        Solver:
            sparse regularization: 
                \Omega(T) = + lambda_reg* \sum_j ( \sum_i T_ij)^{p_reg} with 0 < p_{reg} < 1.
            general problem:
                min_{T \geq 0, T1= h_1} \alpha * <L(C_1, C_2) \otimes T, T> + <M, T> + \Omega(T)
    """
    assert 0 < p_reg < 1
    assert gamma_entropy >= 0
    N1 = C1.shape[0]
    N2 = C2.shape[0]
    
    if T_init is None:
        T= initializer_semirelaxed_GW(init_mode, p, N1, N2, seed=seed, dtype=dtype, device=device)
        
        if use_warmstart:
            T_init = T.clone()
    else:
        assert T_init.shape == (N1,N2)  # shape constraints
        T = T_init.clone()
    
    if symmetry is None:
        symmetry = th.all( C1 == C1.T) and th.all(C2 == C2.T)
    if gamma_entropy == 0:
        inner_solver = (lambda total_linear_cost, T_init : cg_semirelaxed(C1, p, C2, alpha, total_linear_cost,
                                                                          init_mode, T_init, symmetry, inner_log,
                                                                          eps_inner, max_iter_inner, seed, verbose, device, dtype))
    else:
        inner_solver = (lambda total_linear_cost, T_init : md_semirelaxed(C1, p, C2, gamma_entropy, alpha, total_linear_cost,
                                                                          init_mode, T_init, symmetry, inner_log,
                                                                          eps_inner, max_iter_inner, seed, verbose, device, dtype))
    
    reg_linear_cost = 0.
    add_linear_cost = not (linear_cost is None)
    if add_linear_cost:
        total_linear_cost = linear_cost    
    else:
        total_linear_cost = None
    best_T = T.clone()
    ones_p = th.ones((N1,1), dtype=dtype, device=device)
    if use_log:
        log={}
        log['loss']=[]
        if inner_log:
            log['inner_loss']=[]
        #log['T']=[T.copy()]
    best_loss = th.tensor(np.inf, dtype=dtype, device=device)
    current_loss =  th.tensor(1e15, dtype=dtype, device=device)
    convergence_criterion = np.inf
    outer_count=0
    while (convergence_criterion > eps_outer) and (outer_count < max_iter_outer):
        previous_loss = current_loss.clone()
        # 1. Solve the generalized problem using CG (gamma_entropy = 0) or MD (gamma_entropy >0)
        if inner_log :
            T, majorization_loss, inner_log_ = inner_solver(total_linear_cost, T_init)
        else:
            T, majorization_loss = inner_solver(total_linear_cost, T_init)
        # majorization loss satisfies:
        # maj_loss = alpha * srgw_loss + linear_loss + linearized reg loss
        linearized_reg_loss = (reg_linear_cost * T).sum()
        if use_warmstart:
            T_init = T.clone()
        
        # 2. Update the regularization info
        #   - compute exact regularization loss
        #   - Update the regularization cost coming from the tangent approximation
        q = T.sum(axis=0)
        reg_loss = lambda_reg * th.sum((q + 1e-15) ** p_reg)
        current_loss = majorization_loss - linearized_reg_loss + reg_loss
        reg_linear_cost = lambda_reg * p_reg * ((ones_p @ q[None, :]) + 1e-15) ** (p_reg - 1.)
        if add_linear_cost:
            total_linear_cost = reg_linear_cost + linear_cost
        else:
            total_linear_cost = reg_linear_cost
        if verbose:
            print('---outer_count: %s / log : %s  / q : %s '%(outer_count, log['loss'], q))
            
        outer_count+=1
        if use_log:
            log['loss'].append(current_loss.item())
            if inner_log:
                log['inner_loss'].append(inner_log_)

        if previous_loss != 0:
            convergence_criterion = abs(previous_loss.item() - current_loss.item())/ abs(previous_loss.item())
        else:
            convergence_criterion = abs(previous_loss.item() - current_loss.item())/ abs(previous_loss.item() + 1e-15)
        if current_loss < best_loss:
            best_loss = current_loss.clone()
            best_T = T.clone()
    if use_log:
        return best_T, best_loss, log
    else:
        return best_T, best_loss    


def mm_lpl1_semirelaxed_gromov_wasserstein(C1:th.Tensor,
                                           p:th.Tensor,
                                           C2:th.Tensor,
                                           gamma_entropy:float,
                                           T_init:th.Tensor=None,
                                           init_mode:str='product',
                                           symmetry:bool=True,
                                           p_reg:float=1/2,
                                           lambda_reg:float = 0.001,
                                           use_log:bool = False,
                                           use_warmstart:bool=False,
                                           eps_inner:float=10**(-6),
                                           eps_outer:float=10**(-6),
                                           max_iter_inner:int =1000,
                                           max_iter_outer:int =50,
                                           seed:int=0,
                                           verbose:bool=False,
                                           inner_log:bool = False,
                                           dtype:type=th.float64,
                                           device:str='cpu'):
    r""" 
        Solver:
            sparse regularization: 
                \Omega(T) = + lambda_reg* \sum_j ( \sum_i T_ij)^{p_reg} with 0 < p_{reg} < 1.
            srGW problem:
                min_{T \geq 0, T1= h_1} <L(C_1, C_2) \otimes T, T> + \Omega(T)
    """    
    return mm_lpl1_semirelaxed(C1, p, C2, gamma_entropy, 1., None, T_init, init_mode,
                               symmetry, p_reg, lambda_reg, use_log, use_warmstart,
                               eps_inner, eps_outer, max_iter_inner, max_iter_outer,seed,
                               verbose, inner_log, dtype, device)

def mm_lpl1_semirelaxed_fused_gromov_wasserstein(C1:th.Tensor,
                                                 A1:th.Tensor,
                                                 p:th.Tensor,
                                                 C2:th.Tensor,
                                                 A2:th.Tensor,
                                                 alpha:float,
                                                 gamma_entropy:float,
                                                 T_init:th.Tensor=None,
                                                 init_mode:str='product',
                                                 symmetry:bool=True,
                                                 p_reg:float=1/2,
                                                 lambda_reg:float = 0.001,
                                                 use_log:bool = False,
                                                 use_warmstart:bool=False,
                                                 eps_inner:float=10**(-6),
                                                 eps_outer:float=10**(-6),
                                                 max_iter_inner:int =1000,
                                                 max_iter_outer:int =50,
                                                 seed:int=0,
                                                 verbose:bool=False,
                                                 inner_log:bool = False,
                                                 dtype:type=th.float64,
                                                 device:str='cpu'):
    r""" 
        Solver:
            sparse regularization: 
                \Omega(T) = + lambda_reg* \sum_j ( \sum_i T_ij)^{p_reg} with 0 < p_{reg} < 1.
            srFGW problem:
                min_{T \geq 0, T1= h_1} \alpha * <L(C_1, C_2) \otimes T, T> + (1-\alpha) <D, T> + \Omega(T)
    """
    N1 = A1.shape[0]
    N2 = A2.shape[0]
    d = A1.shape[1]
    # Compute matrix of euclidean distances between features
    FS2 = (A1**2) @ th.ones((d, N2), dtype=dtype, device=device)
    FT2 = th.ones((N1, d), dtype=dtype, device=device) @ (A2**2).T
    D = FS2 + FT2 - 2 * A1 @ A2.T
    return mm_lpl1_semirelaxed(C1, p, C2, gamma_entropy, alpha, (1 - alpha) * D, T_init, init_mode,
                               symmetry, p_reg, lambda_reg, use_log, use_warmstart,
                               eps_inner, eps_outer, max_iter_inner, max_iter_outer,seed,
                               verbose, inner_log, dtype, device)