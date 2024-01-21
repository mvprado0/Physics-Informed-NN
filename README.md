
# Jupiter's Radiation Environment using a Physics-Informed Neural Network

A project to improve the model for the radiation environment in Jupiter using a Physics-Informed Neural Network (PINN) in preparation for the Europa Clipper mission. There is limited amount of data from the Galileo mission that orbited Jupiter, so the main idea of this project is to add the physics diffusion equation (which is the equation for radiation belt dynamics that we know the data should follow at least to first order) as an extra term in the loss function to help guide the neural network without the need for more data.

## Running the scripts

First, we need to set the environment. 

```
source setup_script.sh
```
Before running the training script, we want to optimize the model hyperparameters. We would run the optimization by running a version of

```
python generate_optimize_job.py
``` 

Depending on what quantity you want Hyperopt to minimize (ex. average difference loss, average test loss, or both) and how you want to minimize it (ex. with or without cross-validation), there are different versions of this script. These will call the corresponding `Galileo_optimize_cross_val.py` script. 

The optimized parameters will be found in the file `optimized_params_cross_val.json` inside the directory `PINN_output`.

Once you have found the optimized parameters, you can run the training. First, swap out the old parameters for the optimized ones. Then, run any form of the script `Galileo_PINN_Cluster_CrossValidation.py` by replacing the script name inside `generate_sim_jobs.py`. Currently, there are forms of this script for Models 2, 4, 5, 6, and 7. To train many trials in parallel, change the number of trials inside `generate_sim_jobs.py`. Then run it doing 

```
python generate_sim_jobs.py
```

All of the outputs will be found inside of the directory `PINN_output`. This directory and any other directory inside of it will be automatically created by the scripts and do not need to be manually created.

Similarly, to train the final model, we need to run `generate_final_model_jobs.py` and this will run `Galileo_PINN_Cluster_finaltraining.py` for as many trials as the user inputs. Again, the outputs will be found inside `PINN_output`.

```
python generate_final_model_jobs.py
```

Finally, to make the plots, we need to run 

```
python plots_avg_error.py -s <magnetic moment> -m <all models to compare> -f <final training model number> -n <main model>
``` 

* `-s` : Magnetic moment should be of form `10MeVG`
* `-m` : All models to compare should be of this form `-m 0 1 2 4 5 6 7`
* `-f` : Final training model should be the model number for having trained on the full dataset (merging train + test samples and re-training on the whole set) 
* `-n` : Main model should be the model user wants for the a and b distributions for DLL and C as well as the fitted plots of the predicted DLL (or C) vs L-shell
