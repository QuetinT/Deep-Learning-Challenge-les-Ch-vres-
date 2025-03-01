The goal of this project is to predict electricity consumption in France during 2022, given weather data.

DLProject_notebook.ipynb contains explanations about our experiences, and plots of the results.

In train.py, you'll find functions to train various models.
By launching main(), you'll train every model.

Every model is stored in 'Models'

executing pred.py afterward will produce a file named pred.csv, containing the prediction of our best model, which was a MLP, with weather data and selected Fourier features.

In the file 'utilitary_functions.py', you'll find functions beginning by 'submit' or 'plot', that would submit or plot the results of the training of various models (even if they have docstrings, you can find examples of utilisations in the notebook)
