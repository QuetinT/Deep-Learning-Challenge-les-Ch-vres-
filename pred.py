### Load various libraries that we need
from pathlib import Path
import pandas as pd
import joblib
from utilitary_functions import *

import seaborn as sns
import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift
import matplotlib.pyplot as plt


import sklearn as skl
from sklearn import decomposition, metrics, preprocessing, utils
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from tqdm import tqdm


pd.set_option('display.max_columns' , None)
Projet = Path('')


model, scaler_X, scaler_y, best_val_loss = joblib.load(Projet / "Models/gMLP_multi.pkl")
#best results with 30 fourier features selected from 15*3*2 fourier features and
#train_model_MLP_multi(
#    batch_size = 64,
#    epochs = 200,
#    lr=1e-4,
#    scaler_X = preprocessing.MinMaxScaler(),
#    #scaler_y = Target_MinMaxScaler(),
#    dropout = [0.2, 0.1, 0.1, 0.1],
#    patience = 40,
#    hidden_shapes = [120, 105, 90, 60 , 40, 30, 30],
#    batchnorm = True,
#    output_tanh = True
test_MLP_multi = joblib.load(Projet / "Data/test_MLP_multi.pkl")
submit_multi(model, scaler_X, scaler_y, test_MLP_multi, submit_name = "")