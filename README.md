# Conditional Diffusion model for Colour Fundus Photography

## Setup environment

```
python3 -m venv generative_discriminative
source generative_discriminative/bin/activate
pip3 install -r requirements.txt
```

## Setup hyperparameters

Change the values in [config file](conf.yaml). 


## Code for training

To initiate the training process use the following command
```
python run.py --conf sample_config/sample_conf.yaml
```
## Code for sampling

```
python sampler.py --conf conf.yaml
```
## Code for denoising
```
python denoiser.py --conf conf.yaml
```
## Check logs
```
tensorboard --logdir=logs --bind_all
```