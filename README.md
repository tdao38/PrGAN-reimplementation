# PrGAN-scratch

## GET SETUP LOCALLY
1. Clone this repo locally
2. Create a virtual environment called `venv` using python 3.7
```
# FIRST, make sure python3 is python 3.7
✿ 12:09♡ python3
Python 3.7.4 (default, Sep  7 2019, 18:27:02)
[Clang 10.0.1 (clang-1001.0.46.4)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> quit()

# IF, virtualenv module is not already install, use pip3 to install
✿ 12:09♡ pip3 install virtualenv

# THEN, create virtualenv
✿ 12:09♡ virtualenv venv --python $(which python3)

# NEXT, activate virtualenv - notice that (venv) appears before your prompt now
✿ 12:10♡ source venv/bin/activate
(venv) ✿ 12:10♡ 

# FINALLY, install all requirements from requirements.txt
(venv) ✿ 12:11♡ pip install -r requirements.txt
```

## RUNNING STUFF
- To train model from scratch 
```
python3 train.py 
```
    it should automatically create the following directory
        1. checkpoints/ 
        2. results/


- To retrain model from a checkpoint
```
python3 train.py -r 
```
- To evaluate the model's performance by generating 3d object and its 2d images 
```
python3 train.py -e
```
## Contribution 
The orginal code was based on TensorFlow https://github.com/matheusgadelha/PrGAN, but we reimplement it in PyTorch
- Implement prgan_generator.py, prgan_discriminator.py and train.py based on the paper 
    - Thu Dao implemented `prgan_generator.py`
    - Shiqi Gao implemented `prgan_discriminator.py`
    - Together implemented `train.py`
- Reuse several functions in `ops.py` from the original repe  https://github.com/matheusgadelha/PrGAN/blob/master/src/ops.py and reimplement them in PyTorch. These functions are reused and rewritten in PyTorch in line 8-147 in `prgan_generator.py`. 
- We used  functions from provided homework templates in the course to create directories, these are saved in `utils.py` 
