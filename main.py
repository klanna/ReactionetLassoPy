#!/usr/bin/env python3
import optparse
import os
import re
import numpy as np
import pandas as pd
import graphviz as gv

from reactionet_prepare_data import*
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import cross_validation
import scipy
from scipy import *  
from scipy.sparse import coo_matrix
from causal_topology import causal_graph
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import f_regression

def read_input_parameters():
    parser = optparse.OptionParser()
    parser.add_option('-p', '--path', dest='path', help='Path to the folder with your data.', default = None)
    parser.add_option('-m', '--model', dest='model_name', help='Input model: name of the folder with your data.', 
                      default = None)
    parser.add_option('-i', '--inputfile', dest='inputfile', help='Full path to the output folder.', 
                      default = 'data.csv')
    parser.add_option('-o', '--outfolder', dest='outfolder', help='Full path to the output folder.', 
                      default = None)
    parser.add_option('-s', '--skeleton', dest='skeleton', help='Full path to the skeleton (if available).', 
                      default = None)
    parser.add_option('-a', '--alpha', dest='alpha', help='Binomial noise level (probability of success).', 
                      default = 0.05, type = float)
    
    (options, args) = parser.parse_args()
    return options

class output_names(object):
    def __init__(self, options):
        if options.path is None:
            self.path = os.getcwd()
        else:
            self.path = options.path
        
        if options.model_name is None:
            print("Please, specify your input folder!")
            quit()
        else:                        
            self.data_folder = '{0}/{1}/'.format(self.path, options.model_name)

        self.data_file = '{0}/{1}'.format(self.data_folder, options.inputfile)

        if options.outfolder is None:
            self.output_folder = '{0}/results_reactionetlasso/{1}/'.format(self.path, options.model_name)
        else:
            self.output_folder = options.outfolder

        self.alpha = options.alpha
        self.prior_file = options.skeleton                
        self.show()
        self.create()

    def show(self):
        print('Input file:', self.data_file)
        print('Output folder:', self.output_folder)
        print('Prior skeleton:', self.prior_file)
        
    def create(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

sys_names = output_names(read_input_parameters())
 
raw_data = pd.read_csv(sys_names.data_file) / sys_names.alpha

if sys_names.prior_file is None:
    skeleton = None
else:
    skeleton = np.array(pd.read_csv(sys_names.prior_file))
    skeleton[skeleton < 0] = 0

species_names = list(raw_data.columns.values)
del species_names[-1]
timepoints = raw_data['time'].unique()
timepoints = np.delete(timepoints, 0)

mean_dynamics, cov_dynamics = construct_moments(raw_data)
 
plausible_graph = causal_graph(species_names, causal = skeleton, save_to_file = sys_names.output_folder, title_name = 'skeleton')
stoich_cur = plausible_graph.stoich
print('plausible_graph #reactions = ', stoich_cur.shape[1])

# construct linear regression formulation for the means
design = construct_design_mean(stoich_cur, species_names, mean_dynamics, cov_dynamics)
response = construct_response(species_names, mean_dynamics, cov_dynamics, timepoints)

# find OLS solution
OLSQfit = scipy.optimize.nnls(design, response)
k_lsq = OLSQfit[0]
idx_active = np.where(k_lsq > 0)[0]
stoich_lsq = causal_graph(species_names, stoich = stoich_cur[ :, idx_active], causal = None, save_to_file = sys_names.output_folder, title_name = 'OLS')
print('lsq_graph #reactions = ', stoich_lsq.stoich.shape[1])
cgraph = pd.DataFrame(stoich_lsq.causal, columns=species_names)
cgraph.to_csv(sys_names.output_folder + 'causal_lsq.csv', index = False)

stoich_cur = stoich_cur[ :, idx_active]
k_lsq = k_lsq[idx_active]

s = pd.DataFrame(stoich_cur)
s.to_csv(sys_names.output_folder + 'stoich_ols.csv', index = False)
weights = generate_bootstrap_weights(raw_data, stoich_cur, k_lsq, species_names, timepoints)
# add second order moments to improve identifiability
response = construct_response(species_names, mean_dynamics, cov_dynamics, timepoints, 'all')
design = construct_design_second(stoich_cur, raw_data, mean_dynamics, cov_dynamics, species_names)

# reweight the problem according to the noise distribution
for row in range(design.shape[0]):
    design[row] /= weights[row]
response /=  weights    
     
# feasible generallised least squares fit
FGfit = scipy.optimize.nnls(design, response)
k_FG = FGfit[0]
idx_active = np.where(k_FG > 1e-10)[0]

# shrink the feature space for relaxed lasso
stoich_cur = stoich_cur[ :, idx_active]
s = pd.DataFrame(stoich_cur)
s.to_csv(sys_names.output_folder + 'stoich_FG.csv', index = False)
k_FG = k_FG[idx_active]
design = design[:, idx_active]

stoich_FG = causal_graph(species_names, stoich = stoich_cur, causal = None, save_to_file = sys_names.output_folder, title_name = 'FG')
cgraph = pd.DataFrame(stoich_FG.causal, columns=species_names)
cgraph.to_csv(sys_names.output_folder + 'causal_FG.csv', index = False)
 
# adaptive lasso part 
for col in range(design.shape[1]):
    design[:, col] *= k_FG[col]

clf = sklearn.linear_model.ElasticNetCV(eps = 0.0001, cv = 5, l1_ratio=1, n_alphas=100, max_iter=1000000, tol=1e-10, fit_intercept=False, positive=True)
clf.fit(design, response)
k_lasso = clf.coef_ * k_FG

idx_active = np.where(k_lasso > 0)[0]
stoich_lasso = causal_graph(species_names, stoich = stoich_cur[ :, idx_active], causal = None, save_to_file = sys_names.output_folder, title_name = 'lasso')

# select the most meaningful features of lasso according to the means
stoich_cur = stoich_lasso.stoich
s = pd.DataFrame(stoich_cur)
s.to_csv(sys_names.output_folder + 'stoich_lasso.csv', index = False)

design = construct_design_mean(stoich_cur, species_names, mean_dynamics, cov_dynamics)
response = construct_response(species_names, mean_dynamics, cov_dynamics, timepoints)

F, pval = sklearn.feature_selection.f_regression(design, response, center=False)
idx_active = np.where(pval > 1e-2)[0]
stoich_F = causal_graph(species_names, stoich = stoich_cur[ :, idx_active], causal = None, save_to_file = sys_names.output_folder, title_name = 'Freg')
s = pd.DataFrame(stoich_F.stoich)
s.to_csv(sys_names.output_folder + 'stoich_Freg.csv', header = False, index = False)

cgraph = pd.DataFrame(stoich_F.causal, columns=species_names)
cgraph.to_csv(sys_names.output_folder + 'causal_Freg.csv', index = False)
