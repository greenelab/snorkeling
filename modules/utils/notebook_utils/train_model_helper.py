from sklearn.model_selection import GridSearchCV
from snorkel.learning import GenerativeModel

def train_generative_model(data_matrix, burn_in=10, epochs=100, reg_param=1e-6, 
    step_size=0.001, deps=[], lf_propensity=False):
    """
    This function is desgned to train the generative model
    
    data_matrix - the label function matrix which contains the output of all label functions
    burnin - number of burn in iterations
    epochs - number of epochs to train the model
    reg_param - how much regularization is needed for the model
    step_size - how much of the gradient will be used during training
    deps - add dependencey structure if necessary
    lf_propensity - boolean variable to determine if model should model the likelihood of a label function
    
    return a fully trained model
    """
    model = GenerativeModel(lf_propensity=lf_propensity)
    model.train(
        data_matrix, epochs=epochs,
        burn_in=burn_in, reg_param=reg_param, 
        step_size=step_size, reg_type=2
    )
    return model

def run_grid_search(model, data,  grid, labels):
    """
    This function is designed to find the best hyperparameters for a machine learning model.

    model - Sklearn model to be optimized
    data - the data to train the model
    grid - the search grid for each model
    labels - binary training labels for optimization criteria
    """

    searcher = GridSearchCV(model, param_grid=grid, cv=10, return_train_score=True, scoring='roc_auc')
    return searcher.fit(data, labels)