# CycleAugmetGAN

## CycleGAN for data augmentation - Introduction
This repository uses cycleGAN for the augmentation of mammography samples. The GANs used for the augmentation were pretrained on an in-house dataset (*UKE dataset*) subdivided into three visually different domains. The idea is to use the cycleGAN generators for the augmentation of training data, as shown in this figure:
![Idea](https://github.com/AmirMaEl/CycleAugmentGAN/blob/main/overview.png)
 1. use a cycleGAN to train the translation between `BRIGHT`, `NORMAL` and `DARK` subdomains of the *UKE datasaet*. The cycleGAN model architectures were modified with various cyclic (black loss) and acyclic (orange loss) loss functions.
2. cycleGAN generators can now be extracted and reused for the augmentation of training data, thereby imporving robustness and generalizability of the input data (e.g. a *YOLO* breast lesion detector)



## Usage

### Directory structure
```
- Output
	- BRIGHT_NORMAL
	- NORMAL_BRIGHT
	- NORMAL_DARK
	- DARK_NORMAL
- Input 		# copy input files here
	- BRIGHT
	- NORMAL
	- DARK
```

### General

**NOTE**: this code does not require any GPUs!


1. clone the repository 
2. download `model.zip` from [here](https://faubox.rrze.uni-erlangen.de/getlink/fiRBM55TXSUgXzJgRftve6EV/), extract the file and put it in the models directory
3. install `requirements.txt` either with `pip install -r requirements.txt` or simply use conda 
4. choose a model out of the listed models with flag `-md`/`--model`:
    - `UNet_acyc_geo`
    - `UNet_acyc_perc`
    - `UNet_adversarial` - default model 
    - `UNet_cyc_geoqq:`
    - `UNet_cyc_perc`

5. put your input files in the respective `BRIGHT`,`NORMAL` and `DARK` folders in the `Input`
6. generate images using `python3 generate.py -md UNet_acyc_geo`, generated images are saved into the respective subdirectory `Output/BRIGHT_NORMAL`, `Output/NORMAL_BRIGHT`, `Output/DARK_NORMAL` and `Output/NORMAL_DARK`

## Flags

- `-m`/`--model`: choose the generator model out of the list; Default is `UNet_adversarial`
- `-d`/`--delete_input`: delete all previous input files in `BRIGHT`, `NORMAL` and `DARK`; Default is `False`
- `-s`/`--size`: set the generated image size; Default is `512`
 
