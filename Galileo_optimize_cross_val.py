"""
@author: Adapted from Maziar Raissi by George Wilkie & Enrico Camporeale

- Modified to accept Jupiter data from EPD 
- Optimization script to find optimal hyperparameters

Author: Maria Prado Rodriguez

"""

import argparse
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from pprint import pprint
import h5py
import os
import json
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, SparkTrials
from hyperopt.early_stop import no_progress_loss
import pyspark
print(tf.__version__)
print(tfp.__version__)
np.random.seed(None)
tf.compat.v1.set_random_seed(None)
tf.debugging.set_log_device_placement(False)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X, f, bc, layers, layers_sub, lb, ub, fmin0, fmax, sheetname, iteridx, outdir, rate, learn_rate, activation_func):

        # In this version tau is used as a convection term
        
        # Lower boundary in (L,t)
        self.lb = lb 
        # Upper boundary in (L,t)
        self.ub = ub

        # L-shell for training 
        self.x = X[:,0:1]
        # Time for training
        self.t = X[:,1:2]
        # Phase Space Density (Normalized and logged)
        self.f = f
        # Boundary conditions
        self.bc = bc
        self.fmin0 = fmin0 # training f min and max (not logged)
        self.fmax = fmax
        self.sheetname = sheetname
        self.iteridx = iteridx
        self.outdir = outdir

        self.layers = layers
        self.layers_sub = layers_sub
        self.rate = rate
        self.learn_rate = learn_rate
        self.activation_func = activation_func

        # Initialize NNs
        # For f
        self.weights, self.biases = self.initialize_NN(layers)
        
        # D_LL
        self.weights_DLL, self.biases_DLL = self.initialize_NN(layers_sub)

        # C here is tau
        self.weights_tau, self.biases_tau = self.initialize_NN(layers_sub)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.saver = tf.train.Saver()

        # Initialize parameters that will hold the real data
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.f_tf = tf.placeholder(tf.float32, shape=[None, self.f.shape[1]])
        self.bc_tf = tf.placeholder(tf.float32, shape=[None, self.bc.shape[1]])

        self.DLL = self.net_DLL(self.x_tf, self.t_tf)
        self.tau = self.net_tau(self.x_tf, self.t_tf)

        self.f_pred = self.net_f(self.x_tf, self.t_tf)

        self.g_f_pred = self.net_g(self.x_tf, self.t_tf,fmin0,fmax)

        # Data within Boundary, f_tf is the true f and f_pred is the predicted
        self.loss1 = tf.reduce_mean(tf.square(self.f_tf - self.f_pred) ) 
        # Boundary
        self.loss2 = 1.0*tf.reduce_mean(self.bc*tf.square(self.g_f_pred)) 

        # f_tf here will be normalized
        self.loss = tf.reduce_mean(tf.square(self.f_tf - self.f_pred)) + \
                1.0e0*tf.reduce_mean(self.bc*tf.square(self.g_f_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 100000,
                                                                           'maxfun': 5000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(self.learn_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
        # Size of weights matrix.
            W = self.xavier_init(size=[layers[l], layers[l+1]])
	
            # Initial biases are set to zero.
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases, rate, activation_func):
        # Weights is a list of the weight matrix per layer, so the length of that list 
        # gives us the number of layers (including the output layer).
        num_layers = len(weights) + 1

        # Fraction of phase space location with respect to the whole phase space (denoted by the upper and lower boundaries).
        # X is an array of all the L and t values, so H will take this same form.
        # Normalization here for (L,t) goes between [-1, 1] 
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        
        # If num_layers is 4, then loop only does indeces (0,1).
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            
            # matmul = matrix multiplication of the input data with the weights. Then add the biases.
            # Tanh is the activation function.
            # Looping over all the layers. Start out with input activations (first H) and end with second to last layer activations (last H). 
            if(activation_func=='relu'):
                H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
            elif(activation_func=='tanh'):
                H = tf.tanh(tf.add(tf.matmul(H, W), b))
            elif(activation_func=='leaky_relu'):
                H = tf.nn.leaky_relu(tf.add(tf.matmul(H, W), b))
            H = tf.nn.dropout(x=H, rate=rate, seed=l)
        
        # Final  weights and biases for the network (for the output layer).
        W = weights[-1]
        b = biases[-1]
        # Y is the predictions as it is using the last weights, biases, and H.
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_DLL(self, x, t):

        # When calling neural_net(), the output is a normalized version of that quantity. 
        # So we need to take an exponent or renormalize it to get it back to the regular form that we need to the equation.
        DLL = self.neural_net(tf.concat([x,t],1), self.weights_DLL, self.biases_DLL, self.rate, self.activation_func)
        # Returns e^D_LL
        return abs(DLL)

    def net_tau(self, x, t):
        tau = self.neural_net(tf.concat([x,t],1), self.weights_tau, self.biases_tau, self.rate, self.activation_func)
        # Returns e^C
        return tau

    def net_f(self, x, t):
        # What this network actually solves for is: 
        # f = 2.0*(tf.math.log(f) - tf.math.log(fmin)) / (tf.math.log(fmax) - tf.math.log(fmin)) +1e-3
        # The normalized AND logged f. So this is not f. It is log(f) normalized.
        f = self.neural_net(tf.concat([x,t],1), self.weights, self.biases, self.rate, self.activation_func)
        return f

    def net_g(self, x, t, fmin0, fmax):

        # g = log(f) But the rest is normalization for tanh. 
        # When you plug in the form of the normalized input f, you obtain that g=log(f) like you would expect.
        # By calling net_f(), we get a normalized f out since inside net_f() we don't unnormalize like for D_LL and C.
        f = 0.5*(self.net_f(x,t) - 1.0e-3)*(tf.math.log(fmax) - tf.math.log(fmin0)) +tf.math.log(fmin0)
        # Derivative of g with respect to t
        f_t = tf.gradients(f, t)[0]
        # Derivative of g with respect to L
        f_x = tf.gradients(f, x)[0]

        DLL = self.DLL
        tau = self.tau
	
        # Difference of g PDE
        g_f = f_t - tf.square(x)*tf.gradients(f_x*DLL/tf.square(x), x)[0]- DLL*tf.square(f_x)  + tau*tf.gradients(f,x)[0] + tf.gradients(tau,x)[0]

        return g_f

    def callback(self, loss, loss1, loss2 ):
        print('Dataset %d, Loss: %.3e, loss1: %.3e, loss2: %.3e' % (self.iteridx, loss, loss1, loss2))

    def train(self, nIter):

        print('Learning_Rate :', self.optimizer_Adam._lr)

        # Setting the training input dictionary
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.f_tf: self.f, self.bc_tf:self.bc}
        start_time = time.time()
        
        for it in range(nIter):
            # Evaluate self.train_op_Adam using tf_dict as inputs
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss1_value = self.sess.run(self.loss1, tf_dict)
                loss2_value = self.sess.run(self.loss2, tf_dict)
                print('Dataset %d -- It: %d/%d, Losses: %.3e, %.3e, Time: %.2f' %
                      (self.iteridx, it, nIter, loss1_value, loss2_value, elapsed))
                start_time = time.time()

        self.optimizer_Adam._lr *= 0.2
        print('Learning_Rate :',self.optimizer_Adam._lr)
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss1_value = self.sess.run(self.loss1, tf_dict)
                loss2_value = self.sess.run(self.loss2, tf_dict)
                print('Dataset %d -- It: %d/%d, Losses: %.3e, %.3e, Time: %.2f' %
                      (self.iteridx, it, nIter, loss1_value, loss2_value, elapsed))
                start_time = time.time()

        # Now the Scipy optimizer
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss, self.loss1, self.loss2],
                                loss_callback = self.callback)

        self.saver.save(self.sess, f'{self.outdir}/NNparams_EPD_{self.sheetname}_cd_{self.iteridx}.dat')

        loss_tot = self.loss.eval(session=self.sess, feed_dict=tf_dict)
        lossf = self.loss1.eval(session=self.sess, feed_dict=tf_dict)
        lossg = self.loss2.eval(session=self.sess, feed_dict=tf_dict)

        return loss_tot, lossf, lossg

    def predict(self, X_star):

        # Setting the prediction input dictionary
        # Here you are not training, so f is not needed for the prediction. Already a best fit f has been calculate for (L,t) values.
        tf_dict = {self.x_tf: X_star[:,0:1], self.t_tf: X_star[:,1:2]}

        self.saver.restore(self.sess, f'{self.outdir}/NNparams_EPD_{self.sheetname}_cd_{self.iteridx}.dat')

        # Evaluate self.f_pred using tf_dict as inputs
        f_star = self.sess.run(self.f_pred, tf_dict)
        g_f_star = self.sess.run(self.g_f_pred, tf_dict)
        DLL_star = self.sess.run(self.DLL, tf_dict)
        tau_star = self.sess.run(self.tau, tf_dict)

        return f_star, g_f_star, DLL_star, tau_star

def new_dir(fulldir, newdirs, maindir):
    # Makes new directory if there is none.
    if not os.path.exists(fulldir):
        for folder in newdirs:
            try:
                os.mkdir(os.path.join(maindir,folder))
            except FileExistsError:
                pass
            
            maindir = os.path.join(maindir,folder)

def objective(search_space):
    err_train = []
    err_test = []
    
    layers = [2, int(search_space['layers1']), int(search_space['layers2']), 1]
    layers_sub = [2, int(search_space['layers_sub1']), int(search_space['layers_sub2']), 1]

    for iteridx in range(len(orbListTrain)):
        # List of lists of random 4 test orbits: 
        # [4orbitlist_trial0, 4orbitlist_trial1, 4orbitlist_trial2, 4orbitlist_trial3, 4orbitlist_trial4, 4orbitlist_trial5]
        orbUseList = orbListTrain[iteridx]

        # Any of these orbits in the data will be taken as the test sample and anything not in these orbits will be the training sample.
        # This will be done for one set of 4 orbits at a time.
        df_jup_test = df_jup_full.loc[df_jup_full['Orbit'].isin(orbUseList)]     
        df_jup_train = df_jup_full.loc[~df_jup_full['Orbit'].isin(orbUseList)]
        df_jup_train = df_jup_train.sort_values(by=['Boundary'])
        x_data = df_jup_train['L-shell'].values
        t_data = df_jup_train['Time (sec)'].values
        f_data = df_jup_train['PSD'].values
        bc_data = df_jup_train['Boundary'].values # Only care about BC for training data
        logf_data = np.log(f_data)
        
        #  Normalize training data. Normalizing over all values: Training and Test
        fhat_data = 2.0*(logf_data - fdata_min) / (fdata_max - fdata_min) +1e-3

        #  Load test data, also predict and test
        x_data_test = df_jup_test['L-shell'].values
        t_data_test = df_jup_test['Time (sec)'].values
        f_data_test = df_jup_test['PSD'].values
        logf_data_test = np.log(f_data_test)
        
        # Normalize test data Normalizing over all values: Training and Test
        fhat_data_test = 2.0*(logf_data_test - fdata_min) / (fdata_max - fdata_min) +1e-3 

        # Training values only
        fmin0 = float(f_data.min()) # NON-LOG UNNORMALIZED values
        fmax = float(f_data.max()) # NON-LOG UNNORMALIZED values
        Ndata = x_data.size
        N_train = Ndata
        N_test = x_data_test.size

        # Prediction values for both train and test data
        X_star = np.zeros((Ndata,2))
        f_star = np.zeros((Ndata,1))
        bc_wg = np.zeros((Ndata,1))

        X_star[:,0] = x_data # L-shell
        X_star[:,1] = t_data # Time in days
        f_star[:,0] = fhat_data # Normalized PSD
        bc_wg[:,0] = bc_data # BC points

        X_star_test = np.zeros((N_test,2))
        f_star_test = np.zeros((N_test,1))

        X_star_test[:,0] = x_data_test # L-shell
        X_star_test[:,1] = t_data_test # Time in days
        f_star_test[:,0] = fhat_data_test # Normalized PSD

        # Bounds of L-shell and time for training sample.
        lb = X_star.min(0)
        ub = X_star.max(0)
        print("Lower and Upper bounds of L-shell and Time")
        print(lb, ", ", ub)

        # Training Data -- changes the indices every time
        # Shuffles the order of the (L,t) input to the Neural Network. 
        idx = np.random.choice(X_star.shape[0], N_train, replace=False)
        x_train = X_star[idx,:]
        f_train = f_star[idx,:]
        bc_wg_train = bc_wg[idx,:]

        # Train the model with shuffled input
        model = PhysicsInformedNN(x_train, f_train, bc_wg_train, layers, layers_sub, lb, ub, fmin0, fmax, sheetname, iteridx, outdir, search_space['dropout_rate'], search_space['learn_rate'], search_space['activation_func'])
        loss_tot, lossf, lossg = model.train(int(search_space['epochs']))
        print('Loss Total:', loss_tot, 'Loss F:', lossf, 'Loss G:', lossg)

        # Make predictions of Training data -- then save. Doing this to later compare training error with test error to avoid overtraining.
        fhat_pred, g_f_pred, DLL_pred, C_pred = model.predict(X_star)
        # Normalized AND logged F
        fhat_pred = fhat_pred[:,0]
        # Unnormalized LOG OF F (back to the F we know and love)
        logf_pred = np.log(fmin0) + 0.5*(fhat_pred - 1.0e-3)*(np.log(fmax) - np.log(fmin0))
        # JUST F. No more logs!
        f_pred = np.exp(logf_pred)

        # Error on training data. np.linalg.norm is a vector norm 
        error_f = np.linalg.norm(logf_data-logf_pred,2)/np.linalg.norm(logf_data,2)

        #  Do the same with the test data
        fhat_pred_test, g_f_pred_test, DLL_pred_test, C_pred_test = model.predict(X_star_test)
        fhat_pred_test = fhat_pred_test[:,0]
        logf_pred_test = np.log(fmin0) + 0.5*(fhat_pred_test - 1.0e-3)*(np.log(fmax) - np.log(fmin0))
        f_pred_test = np.exp(logf_pred_test)
        
        # Error on test data
        error_test_f = np.linalg.norm(logf_data_test - logf_pred_test, 2)/np.linalg.norm(logf_data_test,2)

        err_train.append(error_f)
        err_test.append(error_test_f)

    avg_error_test = np.average(err_test)
    avg_error_train = np.average(err_train)
    avg_error_diff = avg_error_test - avg_error_train

    min_quant = avg_error_diff + avg_error_test

    print('Average Error diff:', avg_error_diff)
    print('Average Error Test:', avg_error_test)
    print('Average Error Train:', avg_error_train)
    print('Average Error diff + Average Error Test:', min_quant)

    return {'loss': min_quant, 'status': STATUS_OK}

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='idx', default=0, type=int, help='Index, controls the save directory')
parser.add_argument('-s', dest='sheet', default='10MeVG', type=str, help='Controls the input data-sheet')
parser.add_argument('-m', dest='model_num', default=0, type=int, help='NN model architecture number to keep track of output files for different models.')

args = parser.parse_args()
save_idx = args.idx
sheetname = args.sheet
model_num = args.model_num

print('RUNS: ', args.idx)
print('MAGNETIC MOMENT: ', args.sheet)
print('MODEL: ', args.model_num)

## Create Data
main_folder = '/Users/mvprado/Documents/research_presentations/JPL/research/'
indir = main_folder + 'data'
outdir = main_folder + f'PINN_output/moment{sheetname}/model{model_num}/Run{save_idx}'
datafile = outdir + '/' + f'optimized_params_cross_val.json'

folders_outdir = ['PINN_output', f'moment{sheetname}', f'model{model_num}', f'Run{save_idx}']

# Create new directories if not already there.
new_dir(outdir, folders_outdir, main_folder)

# The good orbit list -- list of orbits with more data
orbList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 16, 17, 18, 19, 21, 23, 24, 25, 26, 28, 29, 30]
orbListTrain = []
num_grid_evals = 150 # 150 without parallelism ; 970 with parallelism
early_stop_num = 10

# Randomly select 4 orbits each iteration
while(len(orbList) > 0):
    tempList = np.random.choice(orbList, size=4, replace=False)
    orbList = [x for x in orbList if x not in tempList]
    orbListTrain.append(tempList)

# Load Galileo data
df_jup_full = pd.read_csv(f'{indir}/PINN_EPD_{sheetname}_v2.csv')
df_jup_full = df_jup_full.drop('Unnamed: 0', axis=1)

# Save training and test errors for each iteration
err_train = []
err_test = []
losstot_train = []
lossf_train = []
lossg_train = []

# Refers to the integral that needs to be performed to obtain f which needs to be integrated 
# from fmin to fmax to get a numerical value for f.
fdata_min = np.log(df_jup_full['PSD'].values).min() # LOG f values
fdata_max = np.log(df_jup_full['PSD'].values).max() # LOG f values

# Hyperparameter space that it will optimize
search_space = {'epochs':hp.quniform('epochs',10000,30000,1),
                'layers1': hp.quniform('layers1',1,50,1),
                'layers2': hp.quniform('layers2',1,25,1),
                'layers_sub1':hp.quniform('layers_sub1',1,50,1),
                'layers_sub2':hp.quniform('layers_sub2',1,25,1),
                'dropout_rate':hp.uniform('dropout_rate',0.,0.9),
                'learn_rate':hp.loguniform('learn_rate',-8.0,0.),
                'activation_func':hp.choice('activation_func',['relu','tanh','leaky_relu'])
               }

# Algorithm used for optimization
algorithm = tpe.suggest
best_params = fmin(fn=objective, space=search_space, algo=algorithm, max_evals=num_grid_evals, early_stop_fn=no_progress_loss(early_stop_num))

best_fit = space_eval(search_space, best_params)
print(best_fit)

# Store interger values in json file
intvals = ['epochs', 'layers1', 'layers2', 'layers_sub1', 'layers_sub2']
for cat in intvals:
    best_fit[cat] = int(best_fit[cat])

with open(datafile,'w') as f:
    json.dump(best_fit, f, indent=2, sort_keys=True)
