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


### Neural Networks architectures and Datasets definitions

class Target_MinMaxScaler():
    """MinMax Scaler that scales data globally (instead column wise as in preprocessing.MinMax()).
    """
    def __init__(self):
        self.min = 0
        self.max = 1

    def fit(self, y):
        self.min = y.min()
        self.max = y.max()

    def transform(self, y):
        return((y - self.min)/(self.max - self.min))

    def fit_transform(self, y):
        self.fit(y)
        return(self.transform(y))

    def inverse_transform(self, y):
        return(y*(self.max - self.min) + self.min)

class ConsumptionDataset(Dataset):
    """Create a Dataset object from dataframes with features and consumption data.

    Args:
        Dataset (DataFrame): dataframe containing features and the consumption data
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return(len(self.y))

    def __getitem__(self, idx):
        return(self.X[idx,], self.y[idx])

class gMLP(nn.Module):
    
    def __init__(
                self,
                input_shape,
                hidden_shape : list,
                output_shape,
                activation =  nn.ReLU(),
                dropout = [],
                batchnorm = False,
                output_tanh = False,
                ):

            super().__init__()

            # Here our Linear hidden layers:
            self.W = nn.ModuleList()
            self.Dropout = nn.ModuleList()
            self.batchnorm = batchnorm

            i = input_shape
            for h in hidden_shape:
                    self.W.append(torch.nn.Linear(in_features = i, out_features = h))
                    i = h
            self.W.append(torch.nn.Linear(in_features = i, out_features = output_shape))


            # If dropout added
            if len(dropout) >0:
                self.Dropout = nn.ModuleList()
                for p in dropout:
                        self.Dropout.append(nn.Dropout(p))

            # If batchnorm added

            if batchnorm:
                self.Bn = nn.ModuleList()
                for h in hidden_shape:
                        self.Bn.append(nn.BatchNorm1d(h))
                self.Bn.append(nn.BatchNorm1d(output_shape))

            self.activation = activation
            self.output_tanh = output_tanh
            if output_tanh:
                self.tanh = nn.Tanh()



    def _initialize_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x):
            N = len(self.W)
            output = x
            for k in range(N - 1):
                    output = self.W[k](output)
                    if self.batchnorm:
                            output = self.Bn[k](output)
                    if k < len(self.Dropout):
                            output = self.Dropout[k](output)
                    output = self.activation(output)
            output = self.W[N-1](output)
            if self.batchnorm:
                    output = self.Bn[N-1](output)
            if self.output_tanh:
                    output = 1.5*self.tanh(output)
            return output


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed


# Custom Dataset for LSTM
class TimeSeriesDataset(Dataset):
    def __init__(self, data, target, seq_length):
        self.data = data
        self.target = target
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        X = self.data[index : index + self.seq_length]
        y = self.target[index + self.seq_length]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Take last time step's output
        return out

# preprocessing and training functions

### Weather data :
weather = pd.read_parquet(Projet / "Data/meteo.parquet")

### Calendar data:
train = pd.read_csv(Projet / "Data/train.csv")

### test data:
test = pd.read_csv(Projet / "Data/test.csv")

### Droping rows with missing electricity consumptions
train.dropna(how = 'any', axis = 0, inplace=True)


### Keeping informations about locations in list and dictionaries
locations = list(train.columns)
locations.remove('date')
regions = locations.copy()
regions.remove('France')
metropoles = []

for location in locations:
    if 'Métropole' in location:
        metropoles.append(location)
        regions.remove(location)

dict_locations = {} # Dictionary with locations as keys and list of stations belonging to each location as values
dict_locations['France'] = weather['numer_sta'].unique()
for region in regions:
    dict_locations[region] = weather[weather['nom_reg'] == region]['numer_sta'].unique()

dict_metropoles = {} # Dictionary with metropoles as keys and regions they belong to as values
dict_metropoles['Montpellier Méditerranée Métropole'] = 'Occitanie'
dict_metropoles['Métropole Européenne de Lille'] = 'Hauts-de-France'
dict_metropoles['Métropole Grenoble-Alpes-Métropole'] = "Provence-Alpes-Côte d'Azur"
dict_metropoles["Métropole Nice Côte d'Azur"] = "Provence-Alpes-Côte d'Azur"
dict_metropoles['Métropole Rennes Métropole'] = 'Bretagne'
dict_metropoles['Métropole Rouen Normandie'] = 'Normandie'
dict_metropoles["Métropole d'Aix-Marseille-Provence"] = "Provence-Alpes-Côte d'Azur"
dict_metropoles['Métropole de Lyon'] = 'Auvergne-Rhône-Alpes'
dict_metropoles['Métropole du Grand Nancy'] = 'Grand Est'
dict_metropoles['Métropole du Grand Paris'] = 'Île-de-France'
dict_metropoles['Nantes Métropole'] = 'Pays de la Loire'
dict_metropoles['Toulouse Métropole'] = 'Occitanie'


train = time_zone_convertion(train)
weather = time_zone_convertion(weather)
test = time_zone_convertion(test)




####### LOCAL MLPs #######

### Preprocessing for location wise models
def preprocessing_for_location_wise_models(weather_features = ['t', 'pmer', 'tend', 'u', 'ff', 'pres']):
    # For regions and metropoles
    weather = pd.read_parquet(Projet / "Data/meteo.parquet")
    weather = time_zone_convertion(weather)

    weather['nbas'] = weather['nbas'].astype(np.float64, errors='ignore')
    weather['geop'] = weather['geop'].astype(np.float64, errors='ignore')

    weather.drop(['mois_de_l_annee', 'coordonnees', 'nom', 'cod_tend', 'ww', 'etat_sol', 'type_de_tendance_barometrique', 'temps_present', 'libgeo', 'codegeo', 'nom_epci', 'code_epci', 'nom_dept', 'code_dep', 'nom_reg', 'code_reg'], axis = 1, inplace = True)

    weather = drop_NaN_columns(weather, 0.02)
    
    features_to_flatten = list(weather.columns)
    features_to_flatten.remove('date')
    features_to_flatten.remove('numer_sta')
    weather = flatten_features(weather, features = features_to_flatten)
    
    weather = interpolate_df(weather)
    weather.dropna(how = 'any', axis = 0, inplace=True) # We delete first rows, (which couldn't be linearly interpolated, because there wasn't any value before)
    
    joblib.dump(weather, Projet / "Data/weather_flatten.pkl")
    
    
    test_1 = test.merge(weather, on = ['date'], how = 'inner')
    train_1 = train.merge(weather, on = ['date'], how = 'inner')

    train_1 = add_time_features(train_1)
    test_1 = add_time_features(test_1)
    train_1 = add_fourier_features_classique(train_1)
    test_1 = add_fourier_features_classique(test_1)

    for region in regions:
        train_1_region = selection_region(train_1, region)
        joblib.dump(train_1_region, Projet / ("Data/train_1_"+region+".pkl"))
        test_1_region = selection_region(test_1, region)
        joblib.dump(test_1_region, Projet / ("Data/test_1_"+region+".pkl"))

    
    ### For the France column
    weather = pd.read_parquet(Projet / "Data/meteo.parquet")
    weather['nbas'] = weather['nbas'].astype(np.float64, errors='ignore')
    weather['geop'] = weather['geop'].astype(np.float64, errors='ignore')

    weather.drop(['mois_de_l_annee', 'coordonnees', 'nom', 'cod_tend', 'ww', 'etat_sol', 'type_de_tendance_barometrique', 'temps_present', 'libgeo', 'codegeo', 'nom_epci', 'code_epci', 'nom_dept', 'code_dep', 'nom_reg', 'code_reg'], axis = 1, inplace = True)

    weather = weather[['date', 'numer_sta'] + weather_features]

    weather = drop_NaN_columns(weather, 0.3)

    weather = time_zone_convertion(weather)

    features_to_flatten = list(weather.columns).copy()
    features_to_flatten.remove('date')
    features_to_flatten.remove('numer_sta')
    weather = flatten_features(weather, features_to_flatten)

    weather = interpolate_df(weather)

    columns = list(weather.columns).copy()
    for col in columns:
        if weather[col].isna().sum() > 5:
            weather.drop(col, axis = 1, inplace = True)

    weather.dropna(how = 'any', axis = 0, inplace=True)
    
    df = weather.copy()
    weather_means = dict()
    for region in regions:
        print(region)
        for feature in weather_features:
            l = [feature+'_'+station for station in dict_locations[region]]
            l = [col for col in l if col in list(weather.columns)]
            if l != []:
                weather_means[region+ '_' + feature] = weather[l].mean(axis = 1)
                print("     ", feature, "<--- done")
            else:
                print("     ", feature, "<--- fail")
    
    weather_means = pd.DataFrame(weather_means)
    weather_means.insert(0, 'date', df['date'])
    
    train_fr = train.merge(weather_means, on = ['date'], how = 'inner')
    test_fr = test.merge(weather_means, on = ['date'], how = 'inner')

    train_fr.drop(regions+metropoles, axis = 1, inplace = True)

    train_fr = add_time_features(train_fr)
    test_fr = add_time_features(test_fr)

    train_fr = add_fourier_features_classique(train_fr)
    test_fr = add_fourier_features_classique(test_fr)

    joblib.dump(train_fr, Projet / "Data/train_fr.pkl")
    joblib.dump(test_fr, Projet / "Data/test_fr.pkl")
    return()

### Training functions for location wise models
def train_model_local(location, weather_features,
                batch_size = 32,
                epochs = 200,
                lr = 1e-3,
                dropout = [],
                patience = 15,
                hidden_shapes = None,
                batchnorm = False):
    """train a model for a single location

    Args:
        location (str): location for which we want to train the model
        weather_features (list): list of weather features we want to keep
        batch_size (int, optional): batch size. Defaults to 32.
        epochs (int, optional): number of epochs. Defaults to 200.
        lr (_type_, optional): learning rate. Defaults to 1e-3.
        dropout (list, optional): list of dropouts proportions. Defaults to [].
        patience (int, optional): patience for early stopping. Defaults to 15.
        hidden_shapes (_type_, optional): list of hidden layers shapes. If None is specified, it is chosen accordingly to the number of features. Defaults to None.
        batchnorm (bool, optional): Set to True if you want batchnorm. Defaults to False.
    """
    if location in regions:
        region = location
    else:
        region = dict_metropoles[location]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_1_region = joblib.load(Projet / ("Data/train_1_"+region+".pkl"))

    features = list(train_1_region.columns).copy()
    for feature in features:
        if feature in locations and feature != location:
            train_1_region.drop(feature, axis = 1, inplace = True)

    train_1_region = weather_features_selection(train_1_region, weather_features)

    X = train_1_region.copy()
    y = train_1_region[location]

    time_split = 0.8
    time_split = int(len(X)*time_split/2)
    X_train = pd.concat([X.iloc[:time_split], X.iloc[-time_split:]])
    X_val = X.iloc[time_split:-time_split]
    y_train = pd.concat([y.iloc[:time_split], y.iloc[-time_split:]])
    y_val = y.iloc[time_split:-time_split]

    date_train = X_train['date']
    date_val = X_val['date']
    X_train = X_train.drop(['date', location], axis=1)
    X_val = X_val.drop(['date', location], axis=1)

    #Applying a MinMax on both features and targets
    scaler_X = preprocessing.MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)

    scaler_y = preprocessing.MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_val = scaler_y.transform(y_val.values.reshape(-1, 1))

    #Creating Datasets and Dataloaders
    train_dataset = ConsumptionDataset(X_train, y_train)
    val_dataset = ConsumptionDataset(X_val, y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle = True)

    # Instenciate the model:
    input_shape = train_dataset[0][0].shape[0]
    n_hidden_layers = int(np.log2(input_shape/8).round()+1)
    if hidden_shapes is None:
        hidden_shapes = [round(input_shape/(2**(n+1))) for n in range(n_hidden_layers)]
    #hidden_shapes = [128,64,32,16,8]
    output_shape = 1

    model = gMLP(input_shape, hidden_shapes, output_shape ,dropout = dropout, batchnorm = batchnorm)
    model = model.to(device)

    model._initialize_weights()

    # define the optimizer and the criterion
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training loop :
    # Training Loop
    best_val_loss = np.inf
    best_weights = None
    patience_counter = 0
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Initialize the running loss
        val_loss = 0.0
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for inputs, targets in tqdm_bar:
            # Move data to device (GPU or CPU)
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        model.eval()

        # Disable gradient computation during testing
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Calculate the loss
                outputs_np = outputs.cpu().detach().numpy()
                targets_np = targets.cpu().detach().numpy()

                outputs_tensor = torch.tensor(scaler_y.inverse_transform(outputs_np), dtype=torch.float32)
                targets_tensor = torch.tensor(scaler_y.inverse_transform(targets_np), dtype=torch.float32)

                loss = criterion(outputs_tensor, targets_tensor)

                # Accumulate the validation loss
                val_loss += loss.item()

        # Square root of average validation loss
        avg_val_loss = np.sqrt(val_loss / len(val_loader))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_weights = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        # Average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    model.load_state_dict(best_weights)
    joblib.dump((model, scaler_X, scaler_y, best_val_loss), Projet / ("Models/gMLP_"+location+".pkl"))
    return(model, scaler_X, scaler_y, train_dataset, val_dataset, date_train, date_val)

def train_model_france(batch_size = 64,
                       epochs = 200,
                       lr = 1e-3,
                       dropout = [],
                       patience = 15,
                       hidden_shapes = None,
                       batchnorm = False):
    """train a model for the France column

    Args:
        batch_size (int, optional): batch size. Defaults to 64.
        epochs (int, optional): number of epochs. Defaults to 200.
        lr (float, optional): learning rate. Defaults to 1e-3.
        dropout (list, optional): list of dropouts proportions. Defaults to [].
        patience (int, optional): patience for early stopping. Defaults to 15.
        hidden_shapes (list, optional): list of hidden layer shapes. Defaults to None.
        batchnorm (bool, optional): Set to True to have batch normalization. Defaults to False.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_fr = joblib.load(Projet / "Data/train_fr.pkl")

    #X = train_1_region.drop(['date', region], axis=1)
    X = train_fr.copy()
    y = X['France']

    time_split = 0.8
    time_split = int(len(X)*time_split/2)
    X_train = pd.concat([X.iloc[:time_split], X.iloc[-time_split:]])
    X_val = X.iloc[time_split:-time_split]
    y_train = pd.concat([y.iloc[:time_split], y.iloc[-time_split:]])
    y_val = y.iloc[time_split:-time_split]

    date_train = X_train['date']
    date_val = X_val['date']
    X_train = X_train.drop(['date', 'France'], axis=1)
    X_val = X_val.drop(['date', 'France'], axis=1)

    #Applying a MinMax on both features and targets
    scaler_X = preprocessing.MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)

    scaler_y = preprocessing.MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train.values.reshape(-1,1))
    y_val = scaler_y.transform(y_val.values.reshape(-1,1))

    #Creating Datasets and Dataloaders
    train_dataset = ConsumptionDataset(X_train, y_train)
    val_dataset = ConsumptionDataset(X_val, y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle = True)

    # Instenciate the model:
    input_shape = train_dataset[0][0].shape[0]
    n_hidden_layers = int(np.log2(input_shape/8).round()+1)
    if hidden_shapes == None:
        hidden_shapes = [round(input_shape/(2**(n+1))) for n in range(n_hidden_layers)]
    #hidden_shapes = [128,64,32,16,8]
    output_shape = 1

    model = gMLP(input_shape, hidden_shapes, output_shape , dropout = dropout, batchnorm = batchnorm)
    model = model.to(device)

    model._initialize_weights()

    # define the optimizer and the criterion
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training loop :
    # Training Loop
    best_val_loss = np.inf
    best_weights = None
    patience_counter = 0
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Initialize the running loss
        val_loss = 0.0
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for inputs, targets in tqdm_bar:
            # Move data to device (GPU or CPU)
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        model.eval()

        # Disable gradient computation during testing
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Calculate the loss
                outputs_np = outputs.cpu().detach().numpy()
                targets_np = targets.cpu().detach().numpy()

                outputs_tensor = torch.tensor(scaler_y.inverse_transform(outputs_np), dtype=torch.float32)
                targets_tensor = torch.tensor(scaler_y.inverse_transform(targets_np), dtype=torch.float32)

                loss = criterion(outputs_tensor, targets_tensor)

                # Accumulate the validation loss
                val_loss += loss.item()

        # Square root of average validation loss
        avg_val_loss = np.sqrt(val_loss / len(val_loader))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_weights = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        # Average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    model.load_state_dict(best_weights)
    joblib.dump((model, scaler_X, scaler_y, best_val_loss), Projet / "Models/gMLP_France.pkl")
    return(model, scaler_X, scaler_y, train_dataset, val_dataset, date_train, date_val)

def train_model_location_wise(plot = True):
    model, scaler_X, scaler_y, train_dataset, val_dataset, date_train, date_val = train_model_france(batch_size = 64,
                                                                                                 epochs = 200,
                                                                                                 lr = 1e-3,
                                                                                                 dropout = [0.1,0.1,0.1],
                                                                                                 patience = 15,
                                                                                                 hidden_shapes = [120, 100, 80, 60, 40, 20, 10, 5],
                                                                                                 batchnorm = True)
    if plot:
        print("\n\n\n###### France ######\n\n")
        plot_model_pred_local(model, scaler_X, scaler_y, train_dataset, val_dataset, date_train, date_val)
        
    weather_features = ['t', 'pmer', 'tend', 'u', 'ff', 'n', 'pres']

    for location in locations:
        if location != 'France':
            model, scaler_X, scaler_y, train_dataset, val_dataset, date_train, date_val = train_model_local(location, weather_features, patience = 20)
            if plot:
                print("\n\n\n######"+location+"######\n\n")
                plot_model_pred_local(model, scaler_X, scaler_y, train_dataset, val_dataset, date_train, date_val)
    return()




####### GLOBAL MLP #######

### Preprocessing for the global MLPs

def preprocessing_for_global_models(weather_features = ['t', 'pmer', 'tend', 'u', 'ff', 'pres']):

    weather = pd.read_parquet(Projet / "Data/meteo.parquet")
    weather['nbas'] = weather['nbas'].astype(np.float64, errors='ignore')
    weather['geop'] = weather['geop'].astype(np.float64, errors='ignore')

    weather.drop(['mois_de_l_annee', 'coordonnees', 'nom', 'cod_tend', 'ww', 'etat_sol', 'type_de_tendance_barometrique', 'temps_present', 'libgeo', 'codegeo', 'nom_epci', 'code_epci', 'nom_dept', 'code_dep', 'nom_reg', 'code_reg'], axis = 1, inplace = True)

    weather = weather[['date', 'numer_sta'] + weather_features]

    weather = drop_NaN_columns(weather, 0.3)

    weather = time_zone_convertion(weather)

    features_to_flatten = list(weather.columns).copy()
    features_to_flatten.remove('date')
    features_to_flatten.remove('numer_sta')
    weather = flatten_features(weather, features_to_flatten)

    weather = interpolate_df(weather)

    columns = list(weather.columns).copy()
    for col in columns:
        if weather[col].isna().sum() > 5:
            weather.drop(col, axis = 1, inplace = True)

    weather.dropna(how = 'any', axis = 0, inplace=True)

    df = weather.copy()

    weather_means = dict()
    for region in regions:
        print(region)
        for feature in weather_features:
            l = [feature+'_'+station for station in dict_locations[region]]
            l = [col for col in l if col in list(weather.columns)]
            if l != []:
                weather_means[region+ '_' + feature] = weather[l].mean(axis = 1)
                print("     ", feature, "<--- done")
            else:
                print("     ", feature, "<--- fail")

    weather_means.insert(0, 'date', df['date'])

    train_MLP_multi = train.merge(weather_means, on = ['date'], how = 'inner')
    test_MLP_multi = test.merge(weather_means, on = ['date'], how = 'inner')

    train_MLP_multi = add_time_features(train_MLP_multi)
    test_MLP_multi = add_time_features(test_MLP_multi)

    train_MLP_multi = add_fourier_features_classique(train_MLP_multi,15)
    test_MLP_multi = add_fourier_features_classique(test_MLP_multi,15)


    # Here we drop time features to only keep Fourier features and weather features
    time_features_to_drop = ['year', 'month', 'day', 'day_of_year', 'weekday', 'week', 'is_weekend', 'hour', 'minutes', 'time_index']
    train_MLP_multi.drop(time_features_to_drop, axis = 1, inplace = True)
    test_MLP_multi.drop(time_features_to_drop, axis = 1, inplace = True)


    # We perform a linear regression on our Fourier features to have a clue about which frequencies are important, and drop others
    fourier_features = [e for e in list(train_MLP_multi.columns) if ('cos' in e or 'sin' in e)]
    X = train_MLP_multi[fourier_features]
    y = train_MLP_multi['France'] - train_MLP_multi['France'].mean()

    linear_reg = LinearRegression()
    linear_reg.fit(X, y)

    Coefficients = pd.DataFrame({'Feature': fourier_features, 'Coefficient': linear_reg.coef_, 'Coefficient_abs': np.abs(linear_reg.coef_)})
    Coefficients = Coefficients.sort_values(by = 'Coefficient_abs', ascending = False)
    
    fourier_features_to_drop = list(Coefficients['Feature'].iloc[30:])

    train_MLP_multi.drop(fourier_features_to_drop, axis = 1, inplace = True)
    test_MLP_multi.drop(fourier_features_to_drop, axis = 1, inplace = True)

    joblib.dump(train_MLP_multi, Projet / "Data/train_MLP_multi.pkl")
    joblib.dump(test_MLP_multi, Projet / "Data/test_MLP_multi.pkl")

    return()

### Training functions for the global MLP

def train_model_MLP_multi(batch_size = 32,
                          epochs = 200,
                          lr = 1e-3,
                          scaler_X = preprocessing.MinMaxScaler(),
                          scaler_y = preprocessing.MinMaxScaler(),
                          dropout = [],
                          patience = 15,
                          hidden_shapes = None,
                          batchnorm = False,
                          output_tanh = False):
    """train a multi-output MLP model

    Args:
        batch_size (int, optional): batch size. Defaults to 32.
        epochs (int, optional): number of epochs. Defaults to 200.
        lr (float, optional): learning rate. Defaults to 1e-3.
        scaler_X (optional): transformation applied to the inputs. Defaults to preprocessing.MinMaxScaler().
        scaler_y (optional): transformation applied to the outputs. Defaults to preprocessing.MinMaxScaler().
        dropout (list, optional): list of dropout proportions. Defaults to [].
        patience (int, optional): patience for early stopping. Defaults to 15.
        hidden_shapes (list, optional): list of hidden layer shapes. If None is specified, choose shapes to decrease by 2 each time. Defaults to None.
        batchnorm (bool, optional): Set to True to put batch normalizations. Defaults to False.
        output_tanh (bool, optional): Set to True to put a Tanh activation at the end of the model (1.5*Tanh, to allow greater values than in the training set). Defaults to False.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_MLP_multi = joblib.load(Projet / ("Data/train_MLP_multi.pkl"))

    #X = train_1_region.drop(['date', region], axis=1)
    X = train_MLP_multi.copy()
    y = X[locations]
    #y = X.copy()

    time_split = 0.8
    time_split = int(len(X)*time_split/2)
    X_train = pd.concat([X.iloc[:time_split], X.iloc[-time_split:]])
    X_val = X.iloc[time_split:-time_split]
    y_train = pd.concat([y.iloc[:time_split], y.iloc[-time_split:]])
    y_val = y.iloc[time_split:-time_split]


    date_train = X_train['date']
    date_val = X_val['date']
    X_train = X_train.drop(['date']+locations, axis=1)
    X_val = X_val.drop(['date']+locations, axis=1)

    #Applying a MinMax on both features and targets
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)

    y_train = scaler_y.fit_transform(y_train.values)
    y_val = scaler_y.transform(y_val.values)

    #Creating Datasets and Dataloaders
    train_dataset = ConsumptionDataset(X_train, y_train)
    val_dataset = ConsumptionDataset(X_val, y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle = True)

    # Instenciate the model:
    input_shape = train_dataset[0][0].shape[0]
    n_hidden_layers = int(np.log2(input_shape/25).round()+1)
    if hidden_shapes == None:
        hidden_shapes = [round(input_shape/(2**(n-1))) for n in range(n_hidden_layers)]
    output_shape = train_dataset[0][1].shape[0]

    model = gMLP(input_shape, hidden_shapes, output_shape ,dropout = dropout, batchnorm = batchnorm, output_tanh = output_tanh)
    model = model.to(device)

    model._initialize_weights()

    # define the optimizer and the criterion
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # Training Loop
    best_val_loss = np.inf
    best_weights = None
    patience_counter = 0
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Initialize the running loss
        val_loss = 0.0
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for inputs, targets in tqdm_bar:
            # Move data to device (GPU or CPU)
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        model.eval()

        # Disable gradient computation during testing
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Calculate the loss
                outputs_np = outputs.cpu().detach().numpy()
                targets_np = targets.cpu().detach().numpy()

                outputs_tensor = torch.tensor(scaler_y.inverse_transform(outputs_np), dtype=torch.float32)
                targets_tensor = torch.tensor(scaler_y.inverse_transform(targets_np), dtype=torch.float32)

                loss = criterion(outputs_tensor, targets_tensor)

                # Accumulate the validation loss
                val_loss += loss.item()

        # Square root of average validation loss
        avg_val_loss = np.sqrt(val_loss / len(val_loader))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_weights = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        # Average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    model.load_state_dict(best_weights)
    joblib.dump((model, scaler_X, scaler_y, best_val_loss), Projet / ("Models/gMLP_multi.pkl"))
    return(model, scaler_X, scaler_y, train_dataset, val_dataset, date_train, date_val)




####### MLP on the residuals of a linear regression over fourier features #######

### Training MLP on residuals of linear regression over Fourier features
def train_model_fourier_MLP_residuals(batch_size = 32,
                          epochs = 200,
                          lr = 1e-3,
                          scaler_X = preprocessing.MinMaxScaler(),
                          scaler_y = preprocessing.MinMaxScaler(),
                          dropout = [],
                          patience = 15,
                          hidden_shapes = None,
                          batchnorm = False,
                          output_tanh = False):

    """train linear regressions over Fourier features and then a multi-output MLP model on the residuals
    
    Args:
        batch_size (int, optional): batch size. Defaults to 32.
        epochs (int, optional): number of epochs. Defaults to 200.
        lr (float, optional): learning rate. Defaults to 1e-3.
        scaler_X (optional): transformation applied to the inputs. Defaults to preprocessing.MinMaxScaler().
        scaler_y (optional): transformation applied to the outputs. Defaults to preprocessing.MinMaxScaler().
        dropout (list, optional): list of dropout proportions. Defaults to [].
        patience (int, optional): patience for early stopping. Defaults to 15.
        hidden_shapes (list, optional): list of hidden layer shapes. If None is specified, choose shapes to decrease by 2 each time. Defaults to None.
        batchnorm (bool, optional): Set to True to put batch normalizations. Defaults to False.
        output_tanh (bool, optional): Set to True to put a Tanh activation at the end of the model (1.5*Tanh, to allow greater values than in the training set). Defaults to False.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_MLP_multi = joblib.load(Projet / ("Data/train_MLP_multi.pkl"))

    fourier_features = [e for e in list(train_MLP_multi.columns) if ('cos' in e or 'sin' in e)]
    non_fourier_features = [e for e in list(train_MLP_multi.columns) if e not in fourier_features+['date']+locations]

    X_fourier = train_MLP_multi[fourier_features]
    X = train_MLP_multi[['date']+non_fourier_features].copy()
    y = train_MLP_multi[locations].copy()

    linear_regressions = dict()
    for location in locations:
        y_fourier = train_MLP_multi[location]
        linear_regressions[location] = LinearRegression()
        linear_regressions[location].fit(X_fourier, y_fourier)
        y[location] = y_fourier - linear_regressions[location].predict(X_fourier)


    time_split = 0.8
    time_split = int(len(X)*time_split/2)
    X_train = pd.concat([X.iloc[:time_split], X.iloc[-time_split:]])
    X_val = X.iloc[time_split:-time_split]
    y_train = pd.concat([y.iloc[:time_split], y.iloc[-time_split:]])
    y_val = y.iloc[time_split:-time_split]


    date_train = X_train['date']
    date_val = X_val['date']
    X_train = X_train.drop(['date'], axis=1)
    X_val = X_val.drop(['date'], axis=1)

    #Applying a MinMax on both features and targets
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)

    y_train = scaler_y.fit_transform(y_train.values)
    y_val = scaler_y.transform(y_val.values)

    #Creating Datasets and Dataloaders
    train_dataset = ConsumptionDataset(X_train, y_train)
    val_dataset = ConsumptionDataset(X_val, y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle = True)

    # Instenciate the model:
    input_shape = train_dataset[0][0].shape[0]
    n_hidden_layers = int(np.log2(input_shape/25).round()+1)
    if hidden_shapes == None:
        hidden_shapes = [round(input_shape/(2**(n-1))) for n in range(n_hidden_layers)]
    output_shape = train_dataset[0][1].shape[0]

    model = gMLP(input_shape, hidden_shapes, output_shape ,dropout = dropout, batchnorm = batchnorm, output_tanh = output_tanh)
    model = model.to(device)

    model._initialize_weights()

    # define the optimizer and the criterion
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # Training Loop
    best_val_loss = np.inf
    best_weights = None
    patience_counter = 0
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Initialize the running loss
        val_loss = 0.0
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for inputs, targets in tqdm_bar:
            # Move data to device (GPU or CPU)
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        model.eval()

        # Disable gradient computation during testing
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Calculate the loss
                outputs_np = outputs.cpu().detach().numpy()
                targets_np = targets.cpu().detach().numpy()

                outputs_tensor = torch.tensor(scaler_y.inverse_transform(outputs_np), dtype=torch.float32)
                targets_tensor = torch.tensor(scaler_y.inverse_transform(targets_np), dtype=torch.float32)

                loss = criterion(outputs_tensor, targets_tensor)

                # Accumulate the validation loss
                val_loss += loss.item()

        # Square root of average validation loss
        avg_val_loss = np.sqrt(val_loss / len(val_loader))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_weights = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        # Average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    model.load_state_dict(best_weights)
    joblib.dump((model, scaler_X, scaler_y, best_val_loss, linear_regressions), Projet / ("Models/fourier_gMLP_residuals.pkl"))
    return(model, scaler_X, scaler_y, train_dataset, val_dataset, date_train, date_val, linear_regressions)


####### linear regression with Fourier features on the residuals of a MLP on weather features #######

### Training linear regression on residuals of MLP on weather features
def train_model_MLP_weather_fourier_residuals(batch_size = 32,
                          epochs = 200,
                          lr = 1e-3,
                          scaler_X = preprocessing.MinMaxScaler(),
                          scaler_y = preprocessing.MinMaxScaler(),
                          dropout = [],
                          patience = 15,
                          hidden_shapes = None,
                          batchnorm = False,
                          output_tanh = False):
    """train a multi output MLP on weather features, and then linear regressions on the residuals, over Fourier features
    
    Args:
        batch_size (int, optional): batch size. Defaults to 32.
        epochs (int, optional): number of epochs. Defaults to 200.
        lr (float, optional): learning rate. Defaults to 1e-3.
        scaler_X (optional): transformation applied to the inputs. Defaults to preprocessing.MinMaxScaler().
        scaler_y (optional): transformation applied to the outputs. Defaults to preprocessing.MinMaxScaler().
        dropout (list, optional): list of dropout proportions. Defaults to [].
        patience (int, optional): patience for early stopping. Defaults to 15.
        hidden_shapes (list, optional): list of hidden layer shapes. If None is specified, choose shapes to decrease by 2 each time. Defaults to None.
        batchnorm (bool, optional): Set to True to put batch normalizations. Defaults to False.
        output_tanh (bool, optional): Set to True to put a Tanh activation at the end of the model (1.5*Tanh, to allow greater values than in the training set). Defaults to False.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_MLP_multi = joblib.load(Projet / ("Data/train_MLP_multi.pkl"))

    fourier_features = [e for e in list(train_MLP_multi.columns) if ('cos' in e or 'sin' in e)]
    non_fourier_features = [e for e in list(train_MLP_multi.columns) if e not in fourier_features+['date']+locations]


    X = train_MLP_multi[['date']+non_fourier_features].copy()
    y = train_MLP_multi[locations].copy()


    time_split = 0.8
    time_split = int(len(X)*time_split/2)
    X_train = pd.concat([X.iloc[:time_split], X.iloc[-time_split:]])
    X_val = X.iloc[time_split:-time_split]
    y_train = pd.concat([y.iloc[:time_split], y.iloc[-time_split:]])
    y_val = y.iloc[time_split:-time_split]


    date_train = X_train['date']
    date_val = X_val['date']
    X_train = X_train.drop(['date'], axis=1)
    X_val = X_val.drop(['date'], axis=1)

    #Applying a MinMax on both features and targets
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)

    y_train = scaler_y.fit_transform(y_train.values)
    y_val = scaler_y.transform(y_val.values)

    #Creating Datasets and Dataloaders
    train_dataset = ConsumptionDataset(X_train, y_train)
    val_dataset = ConsumptionDataset(X_val, y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle = True)

    # Instenciate the model:
    input_shape = train_dataset[0][0].shape[0]
    n_hidden_layers = int(np.log2(input_shape/25).round()+1)
    if hidden_shapes == None:
        hidden_shapes = [round(input_shape/(2**(n-1))) for n in range(n_hidden_layers)]
    output_shape = train_dataset[0][1].shape[0]

    model = gMLP(input_shape, hidden_shapes, output_shape ,dropout = dropout, batchnorm = batchnorm, output_tanh = output_tanh)
    model = model.to(device)

    model._initialize_weights()

    # define the optimizer and the criterion
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # Training Loop
    best_val_loss = np.inf
    best_weights = None
    patience_counter = 0
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Initialize the running loss
        val_loss = 0.0
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for inputs, targets in tqdm_bar:
            # Move data to device (GPU or CPU)
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        model.eval()

        # Disable gradient computation during testing
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Calculate the loss
                outputs_np = outputs.cpu().detach().numpy()
                targets_np = targets.cpu().detach().numpy()

                outputs_tensor = torch.tensor(scaler_y.inverse_transform(outputs_np), dtype=torch.float32)
                targets_tensor = torch.tensor(scaler_y.inverse_transform(targets_np), dtype=torch.float32)

                loss = criterion(outputs_tensor, targets_tensor)

                # Accumulate the validation loss
                val_loss += loss.item()

        # Square root of average validation loss
        avg_val_loss = np.sqrt(val_loss / len(val_loader))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_weights = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        # Average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    model.load_state_dict(best_weights)

    linear_regressions = dict()
    X_fourier = train_MLP_multi[fourier_features]
    X_weather = scaler_X.transform(train_MLP_multi[non_fourier_features])
    X_weather = torch.tensor(X_weather, dtype=torch.float32).to(device)
    y_weather = model(X_weather)
    y_weather = scaler_y.inverse_transform(y_weather.detach().cpu().numpy())
    for i in range(len(locations)):
        y = train_MLP_multi[locations[i]] - y_weather[:,i]
        linear_regressions[locations[i]] = LinearRegression()
        linear_regressions[locations[i]].fit(X_fourier, y)
    joblib.dump((model, scaler_X, scaler_y, best_val_loss, linear_regressions), Projet / ("Models/gMLP_weather.pkl"))
    return(model, scaler_X, scaler_y, train_dataset, val_dataset, date_train, date_val, linear_regressions)



####### LSTM #######

### Preprocessing for the LSTM
def preprocessing_for_LSTM(weather_features = ['t', 'pmer', 'tend', 'u']):
    weather = pd.read_parquet(Projet / "Data/meteo.parquet")
    weather['nbas'] = weather['nbas'].astype(np.float64, errors='ignore')
    weather['geop'] = weather['geop'].astype(np.float64, errors='ignore')

    weather.drop(['mois_de_l_annee', 'coordonnees', 'nom', 'cod_tend', 'ww', 'etat_sol', 'type_de_tendance_barometrique', 'temps_present', 'libgeo', 'codegeo', 'nom_epci', 'code_epci', 'nom_dept', 'code_dep', 'nom_reg', 'code_reg'], axis = 1, inplace = True)

    weather = weather[['date', 'numer_sta'] + weather_features]

    weather = drop_NaN_columns(weather, 0.3)

    weather = time_zone_convertion(weather)

    features_to_flatten = list(weather.columns).copy()
    features_to_flatten.remove('date')
    features_to_flatten.remove('numer_sta')
    weather = flatten_features(weather, features_to_flatten)

    weather = interpolate_df(weather)

    columns = list(weather.columns).copy()
    for col in columns:
        if weather[col].isna().sum() > 5:
            weather.drop(col, axis = 1, inplace = True)

    weather.dropna(how = 'any', axis = 0, inplace=True)

    weather_means = dict()
    for region in regions:
        print(region)
        for feature in weather_features:
            l = [feature+'_'+station for station in dict_locations[region]]
            if set(l).issubset(list(weather.columns)):
                weather_means[region+ '_' + feature] = weather[l].mean(axis = 1)
                print("     ", feature, "<--- done")
            else:
                print("     ", feature, "<--- fail")
                
    weather_means = pd.DataFrame(weather_means)

    weather_means.insert(0, 'date', weather['date'])
    weather_means['date']

    train_LSTM = train.merge(weather_means, on = ['date'], how = 'inner')
    test_LSTM = test.merge(weather_means, on = ['date'], how = 'inner')

    train_LSTM = add_time_features(train_LSTM)
    test_LSTM = add_time_features(test_LSTM)

    train_LSTM = add_fourier_features_classique(train_LSTM, 3)
    test_LSTM = add_fourier_features_classique(test_LSTM, 3)


    # Here we drop time features to only keep Fourier features and weather features
    train_LSTM.drop(['year', 'month', 'day', 'day_of_year', 'weekday', 'week',  'hour', 'minutes', 'time_index'], axis = 1, inplace = True)
    test_LSTM.drop(['year', 'month', 'day', 'day_of_year', 'weekday', 'week',  'hour', 'minutes', 'time_index'], axis = 1, inplace = True)

    joblib.dump(train_LSTM, Projet / "Data/train_LSTM.pkl")
    joblib.dump(test_LSTM, Projet / "Data/test_LSTM.pkl")

    return()


### Training of the LSTM
def train_lstm(seq_length=336, hidden_dim=128, num_layers=2, batch_size=32,
               epochs=50, lr=1e-3, patience=5, val_split=0.2):
    """train an LSTM model on a time series dataset

    Args:
        df (DataFrame): dataset to train the model on
        features (list): list of features to use for training
        target_variable (str): target variable name
        seq_length (int, optional): sequence length. Defaults to 336.
        hidden_dim (int, optional): dimension of hidden layer. Defaults to 128.
        num_layers (int, optional): number of hidden layers. Defaults to 2.
        batch_size (int, optional): batch size. Defaults to 32.
        epochs (int, optional): number of epochs. Defaults to 50.
        lr (float, optional): learning rate. Defaults to 1e-3.
        patience (int, optional): patience for early stopping. Defaults to 5.
        val_split (float, optional): proportion of the dataset to use for validation. Defaults to 0.2.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_LSTM = joblib.load(Projet / "Data/train_LSTM.pkl")

    X = train_LSTM.copy()
    y = X[locations]
    X.drop(['date']+locations, axis=1, inplace=True)
    
    # Scale features
    scaler_X = preprocessing.MinMaxScaler()
    X = scaler_X.fit_transform(X)

    scaler_y = preprocessing.MinMaxScaler()
    y = scaler_y.fit_transform(y)

    # Split Data into Train & Validation Sets
    split_idx = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split_idx], X[split_idx - seq_length:]  # Keep sequence continuity in val set
    y_train, y_val = y[:split_idx], y[split_idx - seq_length:]

    # Create Datasets & Dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_length)
    val_dataset = TimeSeriesDataset(X_val, y_val, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate model
    input_dim = len(list(X.columns))
    output_dim = len(locations)
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Early Stopping
    best_val_loss = float("inf")
    patience_counter = 0

    # Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for X_batch, y_batch in tqdm_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.view(y_pred.shape))  # Ensure correct shape
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)

                loss = criterion(y_pred, y_batch.view(y_pred.shape))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_wts = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                model.load_state_dict(best_model_wts)  # Restore best model
                break
    joblib.dump((model, scaler_X, scaler_y, best_val_loss), Projet / "Models/LSTM.pkl")
    return model, scaler_X, scaler_y


####### MLP with autoencoder #######

### Training of the autoencoder
def train_autoencoder(latent_dim, epochs=100, batch_size=32, lr=1e-5, patience=10):
    """Train an autoencoder model on a dataset.

    Args:
        df (DataFrame): dataset to embed
        features (list): list of features to embed
        latent_dim (int): dimension of latent space
        epochs (int, optional): number of epochs. Defaults to 100.
        batch_size (int, optional): batch size. Defaults to 32.
        lr (float, optional): learning rate. Defaults to 1e-5.
        patience (int, optional): patience for early stopping. Defaults to 10.
    """

    train_autoencoder = joblib.load(Projet / "Data/train_LSTM.pkl")
    X = train_autoencoder.copy()
    X.drop(['date']+locations, axis=1, inplace=True)
    input_dim = len(list(X.columns))

    # Data scaling
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)

    # Model creation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim, latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Conversion
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Early stopping
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None  # Sauvegarde du meilleur modèle

    # Training
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        # Vérification early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # Sauvegarde du meilleur modèle
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏳ Early stopping triggered")
                break  # Arrêt de l'entraînement

    # Charger le meilleur modèle
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Encodage des features
    model.eval()
    with torch.no_grad():
        encoded_features = model.encoder(X_tensor).cpu().numpy()

    # Retourner les features encodées sous forme de DataFrame
    encoded_df = pd.DataFrame(encoded_features, index=df.index, columns=[f"latent_{i}" for i in range(latent_dim)])
    joblib.dump((model, encoded_df, scaler), Projet / "Models/autoencoder.pkl")

    return model, encoded_df, scaler


def train_MLP_auto(df, encoded_df, target, batch_size = 32, epochs = 200, lr = 1e-4, dropout = [.2], patience = 15, hidden_shapes = None, batchnorm = False):
    """train a multi-output MLP model on the encoded features

    Args:
        df (DataFrame): dataset to train the model on
        encoded_df (DataFrame): encoded features
        target (str): name of the target variable
        batch_size (int, optional): batch size. Defaults to 32.
        epochs (int, optional): number of epochs. Defaults to 200.
        lr (float, optional): learning rate. Defaults to 1e-4.
        dropout (list, optional): list of dropout proportions. Defaults to [.2].
        patience (int, optional): patience for early stopping. Defaults to 15.
        hidden_shapes (list, optional): list of hidden layer shapes. Defaults to None.
        batchnorm (bool, optional): set to True for batch normalization. Defaults to False.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    X = encoded_df.copy()
    y = df[target]

    time_split = 0.8
    time_split = int(len(X)*time_split/2)
    X_train = pd.concat([X.iloc[:time_split], X.iloc[-time_split:]])
    X_val = X.iloc[time_split:-time_split]
    y_train = pd.concat([y.iloc[:time_split], y.iloc[-time_split:]])
    y_val = y.iloc[time_split:-time_split]



    scaler_y = preprocessing.MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train.values)
    y_val = scaler_y.transform(y_val.values)

    #Creating Datasets and Dataloaders
    train_dataset = ConsumptionDataset(X_train.values, y_train)
    val_dataset = ConsumptionDataset(X_val.values, y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle = True)

    # Instenciate the model:
    input_shape = train_dataset[0][0].shape[0]
    n_hidden_layers = int(np.log2(input_shape/25).round()+1)
    if hidden_shapes == None:
        hidden_shapes = [round(input_shape/(2**(n-1))) for n in range(n_hidden_layers)]
    output_shape = train_dataset[0][1].shape[0]

    model = gMLP(input_shape, hidden_shapes, output_shape ,dropout = dropout, batchnorm = batchnorm)
    model = model.to(device)

    model._initialize_weights()

    # define the optimizer and the criterion
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training loop :
    # Training Loop
    best_val_loss = np.inf
    best_weights = None
    patience_counter = 0
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Initialize the running loss
        val_loss = 0.0
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for inputs, targets in tqdm_bar:
            # Move data to device (GPU or CPU)
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        model.eval()

        # Disable gradient computation during testing
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Calculate the loss
                outputs_np = outputs.cpu().detach().numpy()
                targets_np = targets.cpu().detach().numpy()

                outputs_tensor = torch.tensor(scaler_y.inverse_transform(outputs_np), dtype=torch.float32)
                targets_tensor = torch.tensor(scaler_y.inverse_transform(targets_np), dtype=torch.float32)

                loss = criterion(outputs_tensor, targets_tensor)

                # Accumulate the validation loss
                val_loss += loss.item()

        # Square root of average validation loss
        avg_val_loss = np.sqrt(val_loss / len(val_loader))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_weights = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        # Average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    model.load_state_dict(best_weights)
    
    joblib.dump((model, scaler_y, best_val_loss), Projet / ("Models/gMLP_auto.pkl"))
    return {'model' : model,
            'scaler_y' : scaler_y,
            'train_dataset':train_dataset,
            'val_dataset' : val_dataset}

def preprocess_new_data(new_df, features, scaler_X):
    # Apply the same normalization used during training
    X_new = new_df[features].values

    print("Training Features:", len(scaler_X.feature_names_in_))  # Check training feature names
    print("New Data Features:", len(new_df.columns))

    X_new = scaler_X.transform(X_new)

    # Convert to tensor
    X_tensor = torch.tensor(X_new, dtype=torch.float32)

    return X_tensor

def encode_new_data(autoencoder, X_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)
    X_tensor = X_tensor.to(device)

    autoencoder.eval()
    with torch.no_grad():
        latent_features = autoencoder.encoder(X_tensor).cpu().numpy()

    # Convert to DataFrame
    encoded_df = pd.DataFrame(latent_features, columns=[f"latent_{i}" for i in range(latent_features.shape[1])])

    return encoded_df

def predict_with_gMLP(gmlp_model, encoded_df, scaler_y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gmlp_model.to(device)

    X_tensor = torch.tensor(encoded_df.values, dtype=torch.float32).to(device)


    gmlp_model.eval()
    with torch.no_grad():
        predictions = gmlp_model(X_tensor).cpu().numpy()

    # Inverse transform to get the original scale
    predictions = scaler_y.inverse_transform(predictions)

    return predictions


#################################################### MAIN ####################################################

def main():
    ####### local MLPs #######
    preprocessing_for_location_wise_models()
    train_model_location_wise()
    
    ###### global MLP ######
    preprocessing_for_global_models()
    
    model, scaler_X, scaler_y, train_dataset, val_dataset, date_train, date_val = train_model_MLP_multi(
        batch_size = 64,
        epochs = 200,
        lr=1e-4,
        scaler_X = preprocessing.MinMaxScaler(),
        #scaler_y = Target_MinMaxScaler(),
        dropout = [0.2, 0.1, 0.1, 0.1],
        patience = 40,
        hidden_shapes = [120, 105, 90, 60 , 40, 30, 30],
        batchnorm = True,
        output_tanh = True
    )

    plot_model_pred_MLP_multi(model, scaler_X, scaler_y, train_dataset, val_dataset, date_train, date_val)
    
    ###### MLP on the residuals of a linear regression over fourier features ######
    model, scaler_X, scaler_y, train_dataset, val_dataset, date_train, date_val, linear_regressions = train_model_fourier_MLP_residuals(
        batch_size = 64,
        epochs = 200,
        lr=1e-4,
        scaler_X = preprocessing.MinMaxScaler(),
        #scaler_y = Target_MinMaxScaler(),
        dropout = [0.1, 0.1, 0.1],
        patience = 40,
        hidden_shapes = [100, 80, 60 , 40, 30, 30],
        batchnorm = True,
        output_tanh = True)
    
    ####### linear regression with Fourier features on the residuals of a MLP on weather features ######
    model, scaler_X, scaler_y, train_dataset, val_dataset, date_train, date_val, linear_regressions = train_model_MLP_weather_fourier_residuals(
        batch_size = 64,
        epochs = 200,
        lr=1e-4,
        scaler_X = preprocessing.MinMaxScaler(),
        #scaler_y = Target_MinMaxScaler(),
        dropout = [0.1, 0.1, 0.1],
        patience = 40,
        hidden_shapes = [100, 80, 60 , 40, 30, 30],
        batchnorm = True,
        output_tanh = True)
    
    ####### LSTM #######
    preprocessing_for_LSTM()
    seq_length = 48*7  # Using 10 past time steps (5 hours) for prediction
    lstm_model, scaler_X, scaler_y = train_lstm(seq_length)
    
    ####### autoencoder #######
    model_auto_gMLP, encoded_df, X_scaler_auto = train_autoencoder(latent_dim=64, epochs=100, batch_size=32, lr=1e-5, patience=10)
    df = joblib.load(Projet / "Data/train_LSTM.pkl")
    gMLP_auto = train_MLP_auto(df, encoded_df, locations, batch_size = 32, epochs = 200, lr = 1e-4, dropout = [.2], patience = 15, hidden_shapes = None, batchnorm = False)
    
    return()