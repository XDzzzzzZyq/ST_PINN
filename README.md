# Sample Reconstruction of Spatial Transcriptomics Data through Reversed Fluid Neural ODE Process

ST-PINN is a deep learning model that will reconstruct the sparse Poisson samples through a resampling process based on **Reversed Fluid Neural ODE Process**. 
It generally comprises a reverse Neural operator that could accurately send a diffused count field back to the original state. With perturbation, 

## Configuration
Currently, there are several configurations for demo purposes. 
- ```default_configs.py```: contains most of the basic parameters and setups.
- ```large_configs.py```: will use the large **U-net** architecture implemented in the paper **Score-Based Generative Modeling through Stochastic Differential Equations**
- ```simulate_configs.py```: will use a simulated dataset with a real gene expression profile.
- ```optimizer_configs.py```: could be used for experiments with training hyperparameters.

## Pretraining
A short, supervised pretraining that directly fits the time derivative has been proven to be a method that can accelerate loss convergence. The following command performs the pretraining stage according to the specified configuration file. 
```
python main.py --config config/large_configs.py --mode train --workdir workdir/large
```

## Training
At this stage, the following command will run a semi-supervised training method following the **Adjoint Neural ODE** approach.
```
python main.py --config config/large_configs.py --mode train --workdir workdir/large
```

## Inference
This command will load the trained model and perform the Poisson Resampling method, and will give an estimation of the density field based on the count data.
```
python main.py --config config/large_configs.py --mode deblur --workdir workdir/large
```
