"""
@author: Adapted from Maziar Raissi by George Wilkie & Enrico Camporeale

- Modified to accept Jupiter data from EPD 
- Implemented 6-fold cross validation for model selection:
    1) Out of 30 orbits, randomly select groups of 4
    2) Keep orbits [10, 13, 15, 20, 22, 27] always in training data set as they are shorter

Note: Currently this works for tensorflow v1, using the versions:
Tensorflow v1.15.0
Tensorflow probability v0.8.0

Currently only runs on the CPU cluster!

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
print(tf.__version__)
print(tfp.__version__)
np.random.seed(None)
tf.compat.v1.set_random_seed(None)
tf.debugging.set_log_device_placement(False)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X, f, bc, layers, layers_sub, lb, ub, fmin, fmax, sheetname, iteridx, outdir):

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
        # Training f min and max (not logged)
        self.fmin = fmin 
        self.fmax = fmax
        self.sheetname = sheetname
        self.iteridx = iteridx
        self.outdir = outdir

        self.layers = layers

        self.layers_sub = layers_sub

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

        self.g_f_pred = self.net_g(self.x_tf, self.t_tf,fmin,fmax)

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

        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(2e-3)
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

    def neural_net(self, X, weights, biases):
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
            # No backpropagation here as it is not actually training!
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        # Final  weights and biases for the network (for the output layer).
        # Usually use Sigmoid for last layer activation function.
        W = weights[-1]
        b = biases[-1]
        # Y is the predictions as it is using the last weights, biases, and H.
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_DLL(self, x, t):

        # When calling neural_net(), the output is a normalized version of that quantity. 
        # So we need to take an exponent or renormalize it to get it back to the regular form that we need to the equation.
        DLL = self.neural_net(tf.concat([x,t],1), self.weights_DLL, self.biases_DLL)
        # Returns e^D_LL
        return DLL

    def net_tau(self, x, t):
        tau = self.neural_net(tf.concat([x,t],1), self.weights_tau, self.biases_tau)
        # Returns e^C
        return tau

    def net_f(self, x, t):
        # What this network actually solves for is: 
        # f = 2.0*(tf.math.log(f) - tf.math.log(fmin)) / (tf.math.log(fmax) - tf.math.log(fmin)) +1e-3
        # The normalized AND logged f. So this is not f. It is log(f) normalized.
        f = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return f

    def net_g(self, x, t,fmin,fmax):

        # g = log(f) But the rest is normalization for tanh. 
        # When you plug in the form of the normalized input f, you obtain that g=log(f) like you would expect.
        # By calling net_f(), we get a normalized f out since inside net_f() we don't unnormalize like for D_LL and C.
        f = 0.5*(self.net_f(x,t) - 1.0e-3)*(tf.math.log(fmax) - tf.math.log(fmin)) +tf.math.log(fmin)
        # Derivative of g with respect to t
        f_t = tf.gradients(f, t)[0]
        # Derivative of g with respect to L
        f_x = tf.gradients(f, x)[0]

        DLL = self.DLL
        tau = self.tau
	
        # Difference of g PDE
        g_f = f_t - tf.square(x)*tf.gradients(f_x*abs(DLL)/tf.square(x), x)[0]- abs(DLL)*tf.square(f_x)  + tau*tf.gradients(f,x)[0] + tf.gradients(tau,x)[0]

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

def main():
    # Original layers from Earth data model
    # layers = [2, 30, 20, 20, 20, 20, 20, 1]
    # layers_sub = [2, 30, 20, 10, 1]

    layers = [2, 10, 5, 1]
    layers_sub = [2, 5, 5, 1]

    # For easier bash-scripting
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

    # Create Data
    main_folder = '/Users/mvprado/Documents/research_presentations/JPL/research/'
    indir = main_folder + 'data'
    outdir = main_folder + f'PINN_output/moment{sheetname}/model{model_num}/Run{save_idx}'
    datafile = outdir + '/' + f'Run{save_idx}.json'

    folders_outdir = ['PINN_output', f'moment{sheetname}', f'model{model_num}', f'Run{save_idx}']

    # Create new directories if not already there.
    new_dir(outdir, folders_outdir, main_folder)

    # The good orbit list -- list of orbits with more data
    orbList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 16, 17, 18, 19, 21, 23, 24, 25, 26, 28, 29, 30]
    orbListTrain = []

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
    # Taking the global minimum and maximum values for normalizing the data
    # Refers to the integral that needs to be performed to obtain f which needs to be integrated 
    # from fmin to fmax to get a numerical value for f.
    fdata_min = np.log(df_jup_full['PSD'].values).min() # LOG f values
    fdata_max = np.log(df_jup_full['PSD'].values).max() # LOG f values

    for iteridx in range(len(orbListTrain)):
        # List of lists of random 4 test orbits: [4orbitlist1, 4orbitlist2, 4orbitlist3, 4orbitlist4, 4orbitlist5, 4orbitlist6]
        orbUseList = orbListTrain[iteridx]

        # Any of these orbits in the data will be taken as the test sample and anything not in these orbits will be the training sample.
        # This will be done for one set of 4 orbits at a time.
        df_jup_test = df_jup_full.loc[df_jup_full['Orbit'].isin(orbUseList)]     
        df_jup_train = df_jup_full.loc[~df_jup_full['Orbit'].isin(orbUseList)]
        df_jup_train = df_jup_train.sort_values(by=['Boundary'])
        x_data = df_jup_train['L-shell'].values
        t_data = df_jup_train['Time (sec)'].values
        f_data = df_jup_train['PSD'].values
        # Only care about BC for training data
        bc_data = df_jup_train['Boundary'].values 
        logf_data = np.log(f_data)
        #  Normalize training data. Normalizing over all values: Training and Test
        fhat_data = 2.0*(logf_data - fdata_min) / (fdata_max - fdata_min) +1e-3

        #  Load test data, also predict and test
        x_data_test = df_jup_test['L-shell'].values
        t_data_test = df_jup_test['Time (sec)'].values
        f_data_test = df_jup_test['PSD'].values
        logf_data_test = np.log(f_data_test)
        # Normalize test data normalizing over all values: Training and Test
        fhat_data_test = 2.0*(logf_data_test - fdata_min) / (fdata_max - fdata_min) +1e-3 

        # Training values only
        fmin = float(f_data.min()) # NON-LOG UNNORMALIZED values
        fmax = float(f_data.max()) # NON-LOG UNNORMALIZED values
        Ndata = x_data.size
        N_train = Ndata
        N_test = x_data_test.size

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

        # Shuffles the order of the (L,t) input to the Neural Network. 
        # idx is an ndarray of the same size as the data but randomly ordered.
        idx = np.random.choice(X_star.shape[0], N_train, replace=False)
        x_train = X_star[idx,:]
        f_train = f_star[idx,:]
        bc_wg_train = bc_wg[idx,:]

        # Train the model with shuffled input
        model = PhysicsInformedNN(x_train, f_train, bc_wg_train, layers, layers_sub, lb, ub, fmin, fmax, sheetname, iteridx, outdir)
        loss_tot, lossf, lossg = model.train(10000)
        print('Loss Total:', loss_tot, 'Loss F:', lossf, 'Loss G:', lossg)

        # Make predictions of Training data -- then save. Doing this to later compare training error with test error to avoid overtraining.
        fhat_pred, g_f_pred, DLL_pred, C_pred = model.predict(X_star)
        # Normalized AND logged F
        fhat_pred = fhat_pred[:,0]
        # Unnormalized LOG OF F (back to the F we know and love)
        logf_pred = np.log(fmin) + 0.5*(fhat_pred - 1.0e-3)*(np.log(fmax) - np.log(fmin))
        # JUST F. No more logs!
        f_pred = np.exp(logf_pred)

        # All data saved to text files
        # Load with np.loadtxt
        np.savetxt(f'{outdir}/f_NN_EPD_{sheetname}_train_cd_{iteridx}.txt', f_pred)
        np.savetxt(f'{outdir}/L_EPD_{sheetname}_train_t_{iteridx}.txt', X_star) # Saves L as a function of time
        np.savetxt(f'{outdir}/DLL_NN_EPD_{sheetname}_train_cd_{iteridx}.txt', DLL_pred)
        np.savetxt(f'{outdir}/C_NN_EPD_{sheetname}_train_cd_{iteridx}.txt', C_pred)
        np.savetxt(f'{outdir}/g_NN_EPD_{sheetname}_train_cd_{iteridx}.txt', g_f_pred)

        # Error on training data. 
        error_f = np.linalg.norm(logf_data-logf_pred,2)/np.linalg.norm(logf_data,2)

        #  Do the same with the test data
        fhat_pred_test, g_f_pred_test, DLL_pred_test, C_pred_test = model.predict(X_star_test)
        fhat_pred_test = fhat_pred_test[:,0]
        logf_pred_test = np.log(fmin) + 0.5*(fhat_pred_test - 1.0e-3)*(np.log(fmax) - np.log(fmin))
        f_pred_test = np.exp(logf_pred_test)
        np.savetxt(f'{outdir}/f_NN_EPD_{sheetname}_test_cd_{iteridx}.txt', f_pred_test)
        np.savetxt(f'{outdir}/L_EPD_{sheetname}_test_t_{iteridx}.txt', X_star_test) # Saves L as a function of time
        np.savetxt(f'{outdir}/DLL_NN_EPD_{sheetname}_test_cd_{iteridx}.txt', DLL_pred_test)
        np.savetxt(f'{outdir}/C_NN_EPD_{sheetname}_test_cd_{iteridx}.txt', C_pred_test)
        np.savetxt(f'{outdir}/g_NN_EPD_{sheetname}_test_cd_{iteridx}.txt', g_f_pred_test)

        # Error on test data
        error_test_f = np.linalg.norm(logf_data_test - logf_pred_test, 2)/np.linalg.norm(logf_data_test,2)

        err_train.append(error_f)
        err_test.append(error_test_f)
        losstot_train.append(loss_tot)
        lossf_train.append(lossf)
        lossg_train.append(lossg)
        print(f'Error f (training): {error_f:.5e}')
        print(f'Error f (test): {error_test_f:.5e}')

    # After everything is done running, print out errors over all iterations and take averages
    for idx in range(len(err_train)):
        print(f'Iteration {idx}, orbits used for testing: ', orbListTrain[idx])
        print(f'Training Err: {err_train[idx]:.5e} -- Testing Err: {err_test[idx]:.5e}')

    results_dict = {}
    results_dict.update( { "avg_err_train" : np.average(err_train) } )
    results_dict.update( { "avg_err_test" : np.average(err_test) } )

    with open(datafile,'w') as f:
        json.dump(results_dict, f, indent=2, sort_keys=True)

    print('Average training error:', np.average(err_train))
    print('Average testing error:', np.average(err_test))
    print('Average training total loss:', np.average(losstot_train))
    print('Average training loss F:', np.average(lossf_train))
    print('Average training loss G:', np.average(lossg_train))

    print('Layers for f:', layers)
    print('Layers for D_LL and C:', layers_sub)

if __name__ == "__main__":
    main()
