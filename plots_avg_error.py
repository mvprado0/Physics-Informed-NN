"""

Plotting script for PINN average train and test errors. TODO: NEED TO ADD COMMENTS / CLEAN UP 

Author: Maria Prado Rodriguez

"""

import os, datetime, collections, copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import json
import argparse
import pandas as pd
import random
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

def fitfunc(x, a, b):
    return np.log10(abs(a)) + b*np.log10(x)

def forward_box(x):
    return 1 - (x + 1)**(-1.2)

def inverse_box(x):
    return np.power(1/(1-x), (1/1.2))-1

def forward_sc(x):
    return 1 - (x + 1)**(-2.0)

def inverse_sc(x):
    return np.power(1/(1-x), (1/2.0))-1

def save_image(filename):
    
    p = PdfPages(filename)
      
    fig_nums = plt.get_fignums()  
    figs = [plt.figure(n) for n in fig_nums]
      
    # iterating over the numbers in list
    for fig in figs: 
        
        # and saving the files
        fig.savefig(p, format='pdf') 
      
    p.close() 

def plot_final_model(xval, model_num, label, typ=0):
    
    # Histogram of final model

    if(typ==2):
        hist_bins = np.logspace(np.log10(1e-14),np.log10(1e-2), 20)
    else:
        hist_bins = 20
    x_fit = np.linspace(min(xval), max(xval), num=50)
    xval = [100*val for val in xval]

    fig, ax = plt.subplots(figsize=(5,5))

    mean = np.mean(np.array(xval))
    median = np.median(np.array(xval))

    if(typ==2):
        ax.hist(xval, bins=hist_bins, histtype="stepfilled", color='orange', label='Median: %5.2e' % median) 
    else:
        ax.hist(xval, bins=hist_bins, histtype="stepfilled", color='orange', label='Median: ' + str(round(median,2)))

    ax.axvline(x=median, color='gray', linestyle='dashed')

    if(typ==2):
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel(label, fontsize='large')
    ax.set_ylabel(r'Counts', fontsize='large')
    ax.legend(loc='upper right', fontsize='large', ncol=1)

    plt.title("Mag. Moment: " + str(sheetname) + ", Model: " + str(model_num) + " (1000 trials)", fontsize='large')

def plot_DLL_C_coeff(L_train, val_train, ptype):
    
    # Plot distribution of either DLL or C coefficients
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

    a = []
    b = []

    numbinsa = np.logspace(np.log10(1e-10),np.log10(1e-1), 20)
    numbinsb = 20

    for r in range(len(L_train)):
        fopt, fcov = curve_fit(fitfunc, L_train[r], val_train[r])
        a.append(fopt[0])
        b.append(fopt[1])
    
    mediana = np.median(np.array(a))
    medianb = np.median(np.array(b))
    
    ax1.hist(a, bins=numbinsa, histtype="stepfilled", color='lightseagreen', label='Median: %5.2e' % mediana)
    ax2.hist(b, bins=numbinsb, histtype="stepfilled", color='lightseagreen', label='Median: %5.2e' % medianb)
    ax1.axvline(x=mediana, color='black', linestyle='dashed')
    ax2.axvline(x=medianb, color='black', linestyle='dashed')
    
    ax1.set_ylabel('Counts', fontsize=15)
    ax2.set_ylabel('Counts', fontsize=15)
    ax1.set_xscale('log')
    
    if (ptype=='DLL'):
        ax1.set_xlabel(r'Log($D_{LL}$) Coefficient: a', fontsize=15)
        ax2.set_xlabel(r'Log($D_{LL}$) Coefficient: b', fontsize=15)
    elif (ptype=='C'):
        ax1.set_xlabel(r'Log(C) Coefficient: a', fontsize=15)
        ax2.set_xlabel(r'Log(C) Coefficient: b', fontsize=15)

    fig.suptitle("Magnetic Moment: " + sheetname + ", Model: " + str(main_model), fontsize=15)

    ax1.legend(loc='best', fontsize='large', ncol=1)
    ax2.legend(loc='best', fontsize='large', ncol=1)

def plot_DLL_C(L_train, L_test, val_train, val_test, avgerr, ptype):
    
    # Plot either DLL or C
    
    fig, ax = plt.subplots(figsize=(6,6))

    fopt, fcov = curve_fit(fitfunc, L_train, val_train)

    print('LOOK HERE',fopt[0])
    ax.scatter(L_train, val_train, c='blue', marker="o", label='Train')
    ax.scatter(L_test, val_test, c='orange', marker="o", label='Test')
    ax.plot(L_train, fitfunc(L_train, *fopt), 'r-', label='fit: a=%5.2e, b=%5.1f' % tuple(fopt), linewidth=2.5)
    ax.set_xlabel(r'L-Shell', fontsize='large')
    
    if (ptype=='DLL'):
        ax.set_ylabel(r'Log($D_{LL}$)', fontsize='large')
    elif (ptype=='C'):
        ax.set_ylabel(r'Log(C)', fontsize='large')

    plt.title("Magnetic Moment: " + sheetname + ", Model: " + str(main_model) + ", Run: " + str(rand_run) + ", Trial: " + str(rand_idx) + ', Avg Error Diff: ' + str(round(avgerr,4)), fontsize='medium')

    ax.legend(loc='best', fontsize='large', ncol=1)
    
def plot_avg_error_diff(xval, avg_diff, labels, palette):

    # Box plot 
    
    fig, ax = plt.subplots(figsize=(7,5))

    max_val_diff = 1.3*np.nanmax(avg_diff)

    plt.boxplot(avg_diff, labels=labels, showfliers=False)

    for x, val, c in zip(xval, avg_diff, palette):
        plt.scatter(x, val, alpha=0.4, color=c)

    ax.set_ylim(0., max_val_diff)
    ax.set_yscale('function', functions=(forward_box, inverse_box))
    ax.set_yticks([0, 0.2, 0.5, 1, 2, 8])
    ax.set_xlabel(r'Neural Network Models', fontsize='large')
    ax.set_ylabel(r'Average Error Difference (Test - Train) ', fontsize='large')

    plt.title("Magnetic Moment: " + sheetname + ", " + "(100 trials)", fontsize='large')

def plot_train_test_errors(avg_err_train, avg_err_test, sheetname, palette, labels):
    
    # Diagonal scatter plot
    
    fig, ax = plt.subplots(figsize=(6,6))

    min_val = 0.3 
    max_val = 1.3*max( np.nanmax(avg_err_train), np.nanmax(avg_err_test) )

    ax.plot([min_val, max_val], [min_val, max_val], color="grey", lw=2)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_yscale('function', functions=(forward_sc, inverse_sc))
    ax.set_xscale('function', functions=(forward_sc, inverse_sc))
    ax.set_yticks([0.5, 1.0, 2.0, max_val])
    ax.set_xticks([0.5, 1.0, 2.0, max_val])

    for i in range(len(avg_err_train)):
        if(i==3):
            ax.scatter(avg_err_train[i], avg_err_test[i], c=palette[i], marker="o", label=labels[i], zorder=10)
        else:
            ax.scatter(avg_err_train[i], avg_err_test[i], c=palette[i], marker="o", label=labels[i])

    ax.set_xlabel(r'Average Train Error', fontsize='large')
    ax.set_ylabel(r'Average Test Error', fontsize='large')

    plt.title("Magnetic Moment: " + sheetname + ", " + "(100 trials)", fontsize='large')

    ax.legend(loc='lower right', fontsize='large', ncol=1)

# Main

parser = argparse.ArgumentParser(description='PINN Plots.')
parser.add_argument('-s', dest='sheet', default='10MeVG', type=str, help='Controls the input data-sheet')
parser.add_argument('-m', dest='model_num', nargs='+', help='NN model architecture number to keep track of output files for different models.')
parser.add_argument('-f', dest='finalmodel_num', default=4, type=int, help='Final model number.')
parser.add_argument('-n', dest='main_model', default=4, type=int, help='Best NN model architecture that you will want to test performance of.')

args = parser.parse_args()

sheetname = args.sheet
model_num = args.model_num
fmodel_num = args.finalmodel_num
main_model = args.main_model

numruns = 100
avg_err_test = []
avg_err_train = []
avg_diff = []
xval = []
trials = [0, 1, 2, 3, 4, 5]
labels = ['Model 0', 'Model 1', 'Model 2', 'Model 4', 'Model 5', 'Model 6', 'Model 7']
palette = ['b', 'g', 'r', 'm', 'lightseagreen', 'orange', 'gold']
runs = [int(i) for i in np.linspace(0, numruns-1, numruns).tolist()]
rand_run = random.choice(runs)
rand_idx = random.choice(trials)
main_folder = '/Users/mvprado/Documents/research_presentations/JPL/research/'

for model in model_num:
    err_test = []
    err_train = []
    diff = []
    for run in runs:
        file0 = main_folder + f'PINN_output/moment{sheetname}/model{model}/Run' + str(run) + '/Run' + str(run) + '.json'

        with open(file0) as f:
            results = json.load(f)

        err_test.append(results['avg_err_test'])
        err_train.append(results['avg_err_train'])
        diff.append(results['avg_err_test']-results['avg_err_train'])
    avg_err_test.append(err_test)
    avg_err_train.append(err_train)
    avg_diff.append(diff)

m = np.array(avg_diff)
m = np.stack((m[0],m[1], m[2], m[3], m[4], m[5], m[6]), axis=1)
df = pd.DataFrame(m, columns=labels)
for i, col in enumerate(df.columns):
    xval.append(np.random.normal(i + 1, 0.05, df[col].values.shape[0]))
xval = [xval[i].tolist() for i in range(len(xval))]

# Scatter plot avg error comparison
plot_train_test_errors(avg_err_train, avg_err_test, sheetname, palette, labels)

# Box plot
plot_avg_error_diff(xval, avg_diff, labels, palette)

# Scatter plot D_LL and C plots as functions of L-shell

L = []
DLL = []
C = []

for run in runs:
    for trial in trials:
        quant_folder = main_folder + f'PINN_output/moment{sheetname}/model{main_model}/Run' + str(run)
        L_train = np.loadtxt(quant_folder + f'/L_EPD_{sheetname}_train_t_{trial}.txt') 
        C_pred_train = np.loadtxt(quant_folder + f'/C_NN_EPD_{sheetname}_train_cd_{trial}.txt')
        L_train = list(L_train[:,0])
        C_pred_train = [np.log10(abs(i)) for i in list(C_pred_train)]
        index = np.argsort(L_train)
        L_train = np.array(L_train)[index]
        L.append(L_train)
        C_pred_train = np.array(C_pred_train)[index]
        C.append(C_pred_train)

        if(main_model!=7):
            DLL_pred_train = np.loadtxt(quant_folder + f'/DLL_NN_EPD_{sheetname}_train_cd_{trial}.txt')
            DLL_pred_train = [np.log10(1e-7) if abs(i)==0. else np.log10(abs(i)) for i in list(DLL_pred_train)]
            DLL_pred_train = np.array(DLL_pred_train)[index]
            DLL.append(DLL_pred_train)

plot_DLL_C_coeff(L, C, 'C')
if(main_model!=7):
    plot_DLL_C_coeff(L, DLL, 'DLL')

quant_folder = main_folder + f'PINN_output/moment{sheetname}/model{main_model}/Run' + str(rand_run)
L_train = np.loadtxt(quant_folder + f'/L_EPD_{sheetname}_train_t_{rand_idx}.txt') 
C_pred_train = np.loadtxt(quant_folder + f'/C_NN_EPD_{sheetname}_train_cd_{rand_idx}.txt')
L_test = np.loadtxt(quant_folder + f'/L_EPD_{sheetname}_test_t_{rand_idx}.txt') 
C_pred_test = np.loadtxt(quant_folder + f'/C_NN_EPD_{sheetname}_test_cd_{rand_idx}.txt')
err_diff_DLLC = quant_folder + '/Run' + str(rand_run) + '.json'
L_train = list(L_train[:,0])
L_test = list(L_test[:,0])
C_pred_train = [np.log10(abs(i)) for i in list(C_pred_train)]
C_pred_test = [np.log10(abs(i)) for i in list(C_pred_test)]

index = np.argsort(L_train)
L_train = np.array(L_train)[index]
C_pred_train = np.array(C_pred_train)[index]

if(main_model!=7):
    DLL_pred_train = np.loadtxt(quant_folder + f'/DLL_NN_EPD_{sheetname}_train_cd_{rand_idx}.txt')
    DLL_pred_test = np.loadtxt(quant_folder + f'/DLL_NN_EPD_{sheetname}_test_cd_{rand_idx}.txt')
    DLL_pred_train = [np.log10(abs(i)) for i in list(DLL_pred_train)]
    DLL_pred_test = [np.log10(abs(i)) for i in list(DLL_pred_test)]
    DLL_pred_train = np.array(DLL_pred_train)[index]

with open(err_diff_DLLC) as f:
    dataerr = json.load(f)

avgerr = dataerr['avg_err_test']-dataerr['avg_err_train']

plot_DLL_C(L_train, L_test, C_pred_train, C_pred_test, avgerr, 'C')
if(main_model!=7):
    plot_DLL_C(L_train, L_test, DLL_pred_train, DLL_pred_test, avgerr, 'DLL')

druns = 1000
err_d1 = []
err_d2 = []
lossf = []
lossg = []

for run in range(druns):
    file1 = main_folder + f'PINN_output/moment{sheetname}/final_model{fmodel_num}/Run' + str(run) + '/Run' + str(run) + '.json'

    with open(file1) as f:
        results = json.load(f)

    err_d1.append(results['err_whole_data1'])
    err_d2.append(results['err_whole_data2'])
    lossf.append(results['loss_f'])
    lossg.append(results['loss_g'])

plot_final_model(err_d2, fmodel_num, 'Percent Error Log(PSD)', typ=0)
plot_final_model(lossf, fmodel_num, r'Loss: $(f_{Predicted} - f_{True})^{2}$', typ=1)
plot_final_model(lossg, fmodel_num, r'Loss: (Diffusion Equation Residual)$^{2}$', typ=2)

# Name Pdf file
filename = main_folder + "plots/PINN_error_plots.pdf"

# Call the function
save_image(filename)
