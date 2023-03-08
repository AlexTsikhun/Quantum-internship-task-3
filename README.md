# Quantum-internship-task-3
## Regression on the tabular data

Task is to build model that predicts a target based on the proposed features. Please provide predictions for internship_hidden_test.csv file. Target metric is **RMSE**.
### Project Structure:
```
├───README.md                           <- The top-level README for developers using this project
|
├───data                                
|   |───raw                             <- The original, immutable data dump.
│   |───predictions.csv                 <- Final Predictions
|   └───predictions_dtree_pycaret.csv   <- Predictions with pycaret
|
├───notebooks                           <- EDA
│   └───eda_regres_tabular_data.ipynb   <- EDA notebook
|
├───models                              
|   └───dtree.pkl
|
├───src
|   |───models
|   |   |───predict_model.py            <- Model inference results                   
|   |   └───train_model.py              <- Model training results
|   |
|   ├───constants.py                    <- Constant variables
|   └───utils.py                        <- Useful and repeated func
|
├───.gitignore                          <- Ignore files
|
└───requirements.txt                    <- The requirements file for reproducing the analysis environment, e.g. generated with `pip freeze > requirements.txt`
```
### Solution
The task is to resolve the regressions problem. First of all, the EDA was made (see in notebook). Correlation was found in the dataset. Unnecessary data was removed. After that, the regression result was good (an experiment was conducted, and when the correlated feature was not removed - the results were bad. Column 6 is important). Checked distributions. Using pycaret for selected the best regression model. We made predictions with him (saved the results) and with scikitlearn (final version). The process is described in more detail in the notebook. The model was trained, the prediction results are shown in graphs. RMSE=0.007.

<center>Reuslt of predictions (enlarged image)</center>

![](res.jpg)