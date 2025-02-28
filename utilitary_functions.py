def time_zone_convertion(df):
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'], utc = True)
    df_copy.set_index('date', inplace = True)
    df_copy = df_copy.tz_convert(tz = 'UTC+01:00')
    df_copy.reset_index(inplace=True)
    return(df_copy)


def drop_NaN_columns(df, alpha = 0.5):
    columns_to_drop = []
    for c in df.columns:
        if df[c].isna().sum()/len(df[c]) > alpha:
            columns_to_drop.append(c)
    return(df.drop(columns_to_drop, axis= 1))


def flatten_features(old_df, features, date_range = pd.concat([train[['date']], test[['date']]])):
    """Reshape our weather dataset to have a single row at each time, and a column for each station and each feature

    Args:
        old_df (DataFrame): our old dataset, unflattened
        features (list): the list of features to flatten (our weather features)
        date_range (DataFrame, optional): DataFrame containing a column with the dates we want to keep. Defaults to pd.concat([train[['date']], test[['date']]]).
    """
    df_filtered = old_df.copy()
    df_flat = date_range
    for sta in df_filtered['numer_sta'].unique():
        for feature in features:
            col = pd.DataFrame()
            col[['date',feature+'_'+str(sta)]] = df_filtered.loc[df_filtered['numer_sta'] == sta][['date']+[feature]]
            df_flat = pd.merge(df_flat, col, on='date', how='left')
    return(df_flat)


def interpolate_df(df = weather):
    """Interpolate the df dataset columns to have a value for each 30-minute interval

    Args:
        df (DataFrame, optional): dataframe to interpolate. Defaults to weather.
    """
    expanded_weather = pd.DataFrame()

    # Create a new DataFrame with 30-minute intervals
    date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='30min')
    new_df = pd.DataFrame({'date': date_range})

    # Merge the new DataFrame with the weather dataframe
    new_df = pd.merge(new_df, df, on='date', how='left')

    # Interpolate for all numerical values
    new_df.set_index('date', inplace=True)
    new_df.interpolate(method='time', inplace=True)
    new_df.reset_index(inplace=True)
    return(new_df)


def add_time_features(df):
    """Add time features to the dataset df based on its date column

    Args:
        df (DataFrame): df dataset to which we want to add time features
    """
    df_time = df.copy()
    df_time['year'] = df_time['date'].dt.year.astype(np.float64)
    df_time['month'] = df_time['date'].dt.month.astype(np.float64)
    df_time['day'] = df_time['date'].dt.day.astype(np.float64)
    df_time['day_of_year'] = ((df_time['date'].dt.dayofyear - 1)/365).astype(np.float64)
    df_time['weekday'] = df_time['date'].dt.weekday.astype(np.float64)
    df_time['week'] = df_time['date'].dt.isocalendar().week.astype(np.float64)
    df_time['is_weekend'] = (df_time['weekday'] >= 5).astype(np.float64)
    ### Handle the hours and minutes:
    df_time['hour'] = df_time['date'].dt.hour.astype(np.float64)
    df_time['minutes'] = df_time['date'].dt.minute.astype(np.float64)
    df_time['time_index'] = df_time['hour'] * 2 + (df_time['minutes'] // 30).astype(np.float64)
    return(df_time)


def add_fourier_features(df, column, T, harmonics = 5):
    """add fourier features to the dataset df for the column column, with period T and harmonics

    Args:
        df (DataFrame): dataset to which we want to add fourier features
        column (str): column to which we want to add fourier features
        T (float): period of the fourier features
        harmonics (list or int, optional): list of frequencies, or max frequency. Defaults to 5.

    Returns:
        DataFrame: dataset with the fourier features added on the column 'column'
    """
    if type(harmonics) == int:
        harmonics = range(1, harmonics + 1)
    df_fourier = df.copy()
    for k in harmonics:
        sin_f = column + '_sin_{}'.format(k)
        cos_f = column + '_cos_{}'.format(k)
        df_fourier[sin_f] = np.sin(2 * np.pi * k * df_fourier[column] / T)
        df_fourier[cos_f] = np.cos(2 * np.pi * k *  df_fourier[column] / T)
    return df_fourier


def add_fourier_features_classique(df, nb_harmonics = 5):
    """add fourier features to the dataset df for the columns 'time_index', 'weekday', 'day' and 'day_of_year'

    Args:
        df (DataFrame): dataset to which we want to add fourier features
        nb_harmonics (int, optional): number of harmonics to add per feature. Defaults to 5.
    """
    #***********************Daily Cycle (each half hour)****************************
    df_copy = df.copy()
    period = 48 # There are 48 half-hours in day
    df_copy = add_fourier_features(df_copy, column= 'time_index', T=period, harmonics=nb_harmonics)

    #****************************Weekly cycles**************************************

    df_copy['weekday'] = (df_copy['weekday'] * 48) + df_copy['time_index']
    period = 48 * 7
    df_copy = add_fourier_features(df_copy, 'weekday', T=period, harmonics= nb_harmonics)


    #***********************Monthly Cycle*******************************************

    df_copy['day'] = ((df_copy['day'] - 1) * 48) + df_copy['time_index']
    period = 48 * 30
    df_copy = add_fourier_features(df_copy, 'day', T=period, harmonics=nb_harmonics)

    #**********************Day of year Cycle****************************************

    df_copy = add_fourier_features(df_copy, 'day_of_year', T=1, harmonics=nb_harmonics)
    return(df_copy)


def selection_region(df, region):
    """Allows to drop every weather feature not coming from a station outside a specified region

    Args:
        df (DataFrame): dataset to which we want to apply the selection
        region (str): region we want to keep in the dataset
    """
    df_copy = df.copy()
    list_sta_region = dict_locations[region]
    list_sta_France = dict_locations['France']
    features = list(df_copy.columns).copy()
    for feature in features:
        if len(feature) >= 5:
            if feature[-5:] not in list_sta_region and feature[-5:] in list_sta_France:
                df_copy.drop(feature, axis = 1, inplace = True)
    return(df_copy)


def weather_features_selection(df, weather_features):
    """Drop every weather feature not in the list weather_features

    Args:
        df (DataFrame): dataset to which we want to apply the selection
        weather_features (list): list of weather features we want to keep
    """
    df_copy = df.copy()
    list_sta_France = dict_locations['France']
    features = list(df_copy.columns).copy()
    for feature in features:
        if len(feature)>=5 and feature[-5:] in list_sta_France and feature[:-6] not in weather_features:
            df_copy.drop(feature, axis = 1, inplace = True)
    return(df_copy)


def plot_model_pred_local(model, scaler_X, scaler_y, train_dataset, val_dataset, date_train, date_val):
    """plot the predictions of a single output model (for example a model for a single region) on the training and validation datasets

    Args:
        model : torch model with a single output
        scaler_X : transformation applied to the input data
        scaler_y : transformation applied to the output data
        train_dataset (Dataset): training dataset
        val_dataset (Dataset): validation dataset
        date_train (pd.Serie): dates of the training dataset
        date_val (pd.Serie): dates of the validation dataset
    """
    plot_dataframe_train = pd.DataFrame()
    plot_dataframe_train['date_train'] = date_train
    plot_dataframe_train['target'] = scaler_y.inverse_transform(train_dataset.y.detach().cpu().numpy())
    plot_dataframe_train['prediction'] = scaler_y.inverse_transform(model(train_dataset.X).detach().cpu().numpy())
    plot_dataframe_train['residuals'] = plot_dataframe_train['target'] - plot_dataframe_train['prediction']
    plot_dataframe_train.sort_values(by=['date_train'], inplace = True)

    plt.figure()
    plot_dataframe_train.plot(x = 'date_train', y = ['target', 'prediction'])
    plt.show()

    plt.figure()
    plot_dataframe_train.plot(x = 'date_train', y = 'residuals')
    plt.show()

    plot_dataframe_val = pd.DataFrame()
    plot_dataframe_val['date_val'] = date_val
    plot_dataframe_val['target'] = scaler_y.inverse_transform(val_dataset.y.detach().cpu().numpy())
    plot_dataframe_val['prediction'] = scaler_y.inverse_transform(model(val_dataset.X).detach().cpu().numpy())
    plot_dataframe_val['residuals'] = plot_dataframe_val['target'] - plot_dataframe_val['prediction']
    plot_dataframe_val.sort_values(by=['date_val'], inplace = True)

    plt.figure()
    plot_dataframe_val.plot(x = 'date_val', y = ['target', 'prediction'])
    plt.show()

    plt.figure()
    plot_dataframe_val.plot(x = 'date_val', y = 'residuals')
    plt.show()

    return()


def submit_by_locations():
    """submit the predictions for the model trained on each location
    """
    submit = pd.read_csv(Projet / "Data/test.csv")
    for location in locations:
        if location != 'France':
            model, scaler_X, scaler_y, best_val_loss = joblib.load(Projet / ("Models/gMLP_"+location+".pkl"))
            if location in regions:
                region = location
            else:
                region = dict_metropoles[location]
        test_1_region = joblib.load(Projet / ("Data/test_1_"+region+".pkl"))
        test_1_region = weather_features_selection(test_1_region, weather_features)
        X = test_1_region.drop(['date'], axis=1)
        X = scaler_X.transform(X)
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = model(X)
        y = scaler_y.inverse_transform(y.detach().cpu().numpy())
        submit["pred_"+location] = y

    #submit['pred_France'] = submit[["pred_"+region for region in regions]].sum(axis=1)

    model, scaler_X, scaler_y, best_val_loss = joblib.load(Projet / "Models/gMLP_France.pkl")

    test_fr = joblib.load(Projet / ("Data/test_fr.pkl"))

    X = test_fr.drop(['date'], axis=1)
    X = scaler_X.transform(X)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = model(X)
    y = scaler_y.inverse_transform(y.detach().cpu().numpy())

    submit["pred_France"] = y

    submit = submit[['date']+["pred_"+ location for location in locations]]
    submit.to_csv(Projet / 'Submit/pred_regions.csv', index=False)
    return()


def plot_model_pred_MLP_multi(model, scaler_X, scaler_y, train_dataset, val_dataset, date_train, date_val):
    """plot the predictions of a multi-output model on the training and validation datasets

    Args:
        model : torch model with multiple outputs
        scaler_X : transformation applied to the input data
        scaler_y : transformation applied to the output data
        train_dataset (Dataset): training dataset
        val_dataset (Dataset): validation dataset
        date_train (pd.Serie): dates of the training dataset
        date_val (pd.Serie): dates of the validation dataset
    """
    plot_dataframe_train = pd.DataFrame()
    plot_dataframe_train['date_train'] = date_train
    target = scaler_y.inverse_transform(train_dataset.y.detach().cpu().numpy())
    for i in range(len(locations)):
        plot_dataframe_train[locations[i]] = target[:,i]
    prediction = scaler_y.inverse_transform(model(train_dataset.X).detach().cpu().numpy())
    for i in range(len(locations)):
        plot_dataframe_train['prediction_'+locations[i]] = prediction[:,i]
        plot_dataframe_train['residuals_'+locations[i]] = plot_dataframe_train[locations[i]] - plot_dataframe_train['prediction_'+locations[i]]
    plot_dataframe_train.sort_values(by=['date_train'], inplace = True)


    plot_dataframe_val = pd.DataFrame()
    plot_dataframe_val['date_val'] = date_val
    target = scaler_y.inverse_transform(val_dataset.y.detach().cpu().numpy())
    for i in range(len(locations)):
        plot_dataframe_val[locations[i]] = target[:,i]
    prediction = scaler_y.inverse_transform(model(val_dataset.X).detach().cpu().numpy())
    for i in range(len(locations)):
        plot_dataframe_val['prediction_'+locations[i]] = prediction[:,i]
        plot_dataframe_val['residuals_'+locations[i]] = plot_dataframe_val[locations[i]] - plot_dataframe_val['prediction_'+locations[i]]
    plot_dataframe_val.sort_values(by=['date_val'], inplace = True)


    for location in locations:
        fig, axs = plt.subplots(2, 2, figsize = (15,10))
        plot_dataframe_train.plot(x = 'date_train', y = [location, 'prediction_'+location], ax = axs[0,0], title = 'train')
        plot_dataframe_val.plot(x = 'date_val', y = [location, 'prediction_'+location], ax = axs[0,1], title = 'val')
        plot_dataframe_train.plot(x = 'date_train', y = ['residuals_'+location], ax = axs[1,0], title = 'train')
        plot_dataframe_val.plot(x = 'date_val', y = ['residuals_'+location], ax = axs[1,1], title = 'val')
        plt.show()
    return()


def submit_multi(model, scaler_X, scaler_y, test_1, submit_name = ""):
    """submit the predictions for a multi-output model

    Args:
        model : torch model with multiple outputs
        scaler_X : transformation applied to the inputs
        scaler_y : transformation applied to the outputs
        test_1 (DataFrame): test dataset, with the same features as the training dataset
        submit_name (str, optional): submition name to add after "pred" in the file name. Defaults to "".
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test = pd.read_csv(Projet / "Data/test.csv")
    date = test['date']

    col = ['pred_'+location for location in locations]

    X = test_1.drop(['date'], axis=1)
    X = scaler_X.transform(X)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = model(X)
    y = scaler_y.inverse_transform(y.detach().cpu().numpy())

    y_pred = pd.DataFrame(y, columns=col)
    y_pred.columns = col
    y_pred.insert(0, 'date', date)
    y_pred.to_csv(Projet / ('Submit/pred'+submit_name+'.csv'), index = False)
    return()


def submit_fourier_MLP_residual(model, scaler_X, scaler_y, linear_regressions, test_1, submit_name = ""):
    """submit the predictions for a model trained on the residuals of the linear regressions over Fourier features (or on the other way)

    Args:
        model : torch model with multiple outputs
        scaler_X : transformation applied to the inputs of the MLP
        scaler_y : transformation applied to the outputs of the MLP
        linear_regressions (dict): dictionary of linear regressions for each location
        test_1 (DataFrame): test dataset, with the same features as the training dataset
        submit_name (str, optional): submit name, to add after "pred" in the file name. Defaults to "".
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test = pd.read_csv(Projet / "Data/test.csv")
    date = test['date']

    col = ['pred_'+location for location in locations]

    X = test_1.drop(['date'], axis=1)
    fourier_features = [e for e in list(X.columns) if ('cos' in e or 'sin' in e)]
    X_fourier = X[fourier_features]
    X.drop(fourier_features, axis = 1, inplace = True)
    X = scaler_X.transform(X)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = model(X)
    y = scaler_y.inverse_transform(y.detach().cpu().numpy())

    y_pred = pd.DataFrame(y, columns=col)
    y_pred.columns = col
    y_pred.insert(0, 'date', date)
    for location in locations:
        y_pred['pred_'+location] = y_pred['pred_'+location] + linear_regressions[location].predict(X_fourier)
    y_pred.to_csv(Projet / ('Submit/pred'+submit_name+'.csv'), index = False)
    return()


def predict_lstm(model, df, features, scaler_X, scaler_y, seq_length=10, batch_size=32):
    """Predicts over the entire test dataset using a rolling window approach.
    
    Args:
        model : Trained LSTM model
        df (DataFrame): Test dataframe
        features (list) : List of input features
        scaler_X : Scaler used for inputs
        scaler_y : Scaler used for target variable (25 outputs)
        seq_length (int, optional): Number of time steps (default 10 for 5-hour window)
        batch_size (int, optional): Not used (single-step inference)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set to evaluation mode

    date = df['date']
    col = ['pred_'+location for location in locations]


    # Normalize test data using training scaler
    X_test = scaler_X.transform(df[features])

    # Store predictions
    predictions = []

    # Sliding window prediction
    with torch.no_grad():
        for i in range(len(X_test) - seq_length):
            X_input = X_test[i : i + seq_length]  # Get sequence
            X_tensor = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dim

            y_pred = model(X_tensor)
            y_pred = y_pred.cpu().numpy().flatten()  # Flatten to match (25,)

            predictions.append(y_pred)

    # Convert to numpy and reshape for inverse scaling
    predictions = np.array(predictions)  # Shape: (num_predictions, 25)
    predictions = scaler_y.inverse_transform(predictions)  # Now correctly shaped (num_predictions, 25)

    y_pred = pd.DataFrame(predictions, columns=col)
    y_pred.columns = col
    y_pred.insert(0, 'date', date)

    return y_pred

    
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