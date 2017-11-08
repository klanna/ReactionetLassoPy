#!/usr/bin/env python3
import numpy as np
from scipy.sparse import coo_matrix
import progressbar

#==============================================================================
# class causal_graphs():
#     def __init__(self, causal = None, stoich = None, num_species = None, num_reactions = None):
#         if stoich == None:            
#             if causal_graph == None:
#                 self.causal
#             self.stoich_from_causal()
#         else:
#             self.stoich = stoich
#             if causal == None:
#                 self.causal_from_stoich()
#             else:
#                 self.causal = causal
#==============================================================================
                
def construct_moments(data):
#    mean_dynamics = data.groupby('time').mean()
    mean_dynamics = data.groupby('time').mean()
    cov_dynamics = data.groupby('time').cov().unstack()
    
    mean_dynamics = mean_dynamics.drop(mean_dynamics.index[0])
    cov_dynamics = cov_dynamics.drop(cov_dynamics.index[0])
    
    return mean_dynamics, cov_dynamics
    
def get_propensity(parents, species_names, mean_dynamics, cov_dynamics):
    if len(parents) == 1:
        propensity = mean_dynamics[species_names[parents[0]]]
    else:
        propensity = cov_dynamics[species_names[parents[0]]][species_names[parents[1]]] + mean_dynamics[species_names[parents[0]]]*mean_dynamics[species_names[parents[1]]]
    return(list(propensity))
        
def construct_design_mean(stoich, species_names, mean_dynamics, cov_dynamics):
#    # estimate structure of the design matrix    
    n_species, n_reactions = stoich.shape
    n_tp = len(mean_dynamics.index)
    n = n_species*n_tp
    
    row = []
    col = []
    data = []
    for re, stoich_vec in enumerate(stoich.transpose()):
        parents = np.where(stoich_vec == -1)[0]
        sp_idx = np.where(stoich_vec != 0)[0]
        
        col = col + [re]*(len(sp_idx)*n_tp)
        a_l = get_propensity(parents, species_names, mean_dynamics, cov_dynamics)
        for i in sp_idx:
            row = row + list(range(n_tp*i, n_tp*(i+1)))
            data = data + [stoich_vec[i]*j for j in a_l]

    A = coo_matrix((data, (row, col)), shape = (n, n_reactions)).toarray()
    return A
#
def construct_response(species_names, mean_dynamics, cov_dynamics, timepoints, gradientType = 'mean'):
#    # estimate gradients with smoothing splines
    dx = np.gradient(timepoints)
    b = np.empty([0, 0])
    for sp in species_names:
        dy = np.gradient(mean_dynamics[sp]).transpose()        
        b = np.append(b, dy/dx)
    
    if gradientType == 'all':
        for isp1 in range(len(species_names)):
            for isp2 in [j for j in range(len(species_names)) if j > isp1]:
                sp1 = species_names[isp1]
                sp2 = species_names[isp2]
                dy = np.gradient(cov_dynamics[sp1][sp2]).transpose()        
                b = np.append(b, dy/dx)                        
    return b

def construct_design_second(stoich, data, mean_dynamics, cov_dynamics, species_names):
    n_species, n_reactions = stoich.shape
    for re, stoich_vec in enumerate(stoich.transpose()):        
        parents = np.where(stoich_vec == -1)[0]
        p1 = species_names[parents[0]]
        if len(parents) > 1:            
            p2 = species_names[parents[1]]
            data['prop'] = data[p1]*data[p2]
            mean_prop = data.groupby('time')['prop'].mean()
#            mean_prop = data.groupby('time')['prop'].mean()
            
            cov_prop = data.groupby('time').cov().unstack()['prop']
            
            mean_prop = mean_prop.drop(mean_prop.index[0])
            cov_prop = cov_prop.drop(cov_prop.index[0])
        else:
            mean_prop = mean_dynamics[p1]
            cov_prop = cov_dynamics[p1]
        
        feature_vec = np.empty([0, 0])        
        for isp1 in range(len(species_names)):        
            sp1 = species_names[isp1]
            feature_vec = np.append(feature_vec, stoich_vec[isp1]*mean_prop)
            
        for isp1 in range(len(species_names)):
            for isp2 in [j for j in range(len(species_names)) if j > isp1]:
                sp1 = species_names[isp1]
                sp2 = species_names[isp2]                 
                feature_vec = np.append(feature_vec, stoich_vec[isp1]*stoich_vec[isp2]*mean_prop + stoich_vec[isp1]*cov_prop[sp2] + stoich_vec[isp2]*cov_prop[sp1])
        if re:
            design = np.column_stack((design, feature_vec))
        else:
            design = feature_vec
    return design

def generate_bootstrap_weights(raw_data, stoich, x, species_names, timepoints, n_boot = 100, eps = 1e-10):    
    residuals = np.empty([0,0])
    bar = progressbar.ProgressBar()
    for b in range(n_boot):
        boot_sample = raw_data.sample(frac=0.8, replace=True)
        boot_sample = boot_sample.sort_values('time')
        mean_dynamics, cov_dynamics = construct_moments(boot_sample)
        response = construct_response(species_names, mean_dynamics, cov_dynamics, timepoints, 'all')
        design = construct_design_second(stoich, boot_sample, mean_dynamics, cov_dynamics, species_names)
        res = response - np.dot(design, x)
        if b:
            residuals = np.column_stack((residuals, res))
        else:
            residuals = res        
        if b % 5 == 0:
            bar.update(b)
    weights = residuals.std(axis = 1)
    weights[np.where(weights < eps)] = 1
    print('\n')
    return(weights)

def generate_bootstrap_weights_lsq(raw_data, stoich, x, species_names, timepoints, n_boot = 100, eps = 1e-10):    
    residuals = np.empty([0,0])
    bar = progressbar.ProgressBar()
    for b in range(n_boot):
        boot_sample = raw_data.sample(frac=0.8, replace=True)
        boot_sample = boot_sample.sort_values('time')
        mean_dynamics, cov_dynamics = construct_moments(boot_sample)
        response = construct_response(species_names, mean_dynamics, cov_dynamics, timepoints, 'mean')
        design = construct_design_mean(stoich, species_names, mean_dynamics, cov_dynamics)
        res = response - np.dot(design, x)
        if b:
            residuals = np.column_stack((residuals, res))
        else:
            residuals = res        
        if b % 5 == 0:
            bar.update(b)
    weights = residuals.std(axis = 1)
    weights[np.where(weights < eps)] = 1
    print('\n')
    return(weights)