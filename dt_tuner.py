from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Decision Tree Tuner Module

# Create Decision Tree Model
model = DecisionTreeRegressor()

# RESULTS
# Run dtA
best_model_dta_h, dta_param_grid = dt_tunerA(model, X_train888, X_test_h, Y_train888, Y_test_h)
# dtA#1 Best score: 0.915601902864449  #1st tuner run
# dtA#1 Best hyperparameters: {'max_depth': 3, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10}
# dtA#2 Best score: 0.9961605376307513  2nd tuner run
# dtA#2 Best hyperparameters: {'max_depth': 7, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 2}
# dtA#3 Best score: 0.9910901056338818
# dtA#3 Best hyperparameters: {'max_depth': 7, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 2}

# Run dtB
best_model_hcap_dtb, dtb_param_grid_hcap = dt_tunerB(model, X_train_h,  X_test_h, Y_train_h, Y_test_h)
# dtB#1 Best score: -1.0305959300120142 #1st Run
# dtB#1 Best hyperparameters: {'max_depth': 9, 'min_samples_leaf': 2, 'min_samples_split': 2}
# dtB#2 Best score: -1.0190935914339612
# dtB#2 Best hyperparameters: {'max_depth': 9, 'min_samples_leaf': 2, 'min_samples_split': 2}
# dtB#3 Best score: -1.0205678088243584
# dtB#3 Best hyperparameters: {'max_depth': 9, 'min_samples_leaf': 1, 'min_samples_split': 8}


# END RESULTS-------------------------------------------------------------------------------]


# dt_tunerA
# defines a hyperparameter space for decision tree regressor 
# performs hyperparameter tuning using grid search
# evaluates the best model on test data
# prints the model score, best hyperparameters,& model type
# returns the best decision tree regressor object with the hyperparameters dict as dt_param_grid

def dt_tunerA(model, X_train, X_test, Y_train, Y_test):
    # Define the hyperparameter grid to search over
    param_grid = {
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Perform hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, Y_train)

    # Evaluate the best model on the test data
    best_model = grid_search.best_estimator_
    Y_pred = best_model.predict(X_test)
    score = best_model.score(X_test, Y_test)

    # Print the best hyperparameters and score
    print("dtA Best score:", score)
    print("dtA Best hyperparameters:", grid_search.best_params_)
    print("Model type:", type(best_model))

    # Return the best model and hyperparameters
    dt_param_grid = grid_search.best_params_
    return best_model, dt_param_grid




# dt_tunerB 
# scales the features of the train test data using StandardScaler
# defines a hyperparameter space for decision tree regressor
# performs grid search cross-validation finding the best hyperparameters
# evaluates the best model on test data
# prints best hyperparameters, score, and model type
# returns the best model and hyperparameters dict

def dt_tunerB(model, X_train_h, X_test_h, Y_train_h, Y_test_h):
    # Scale the features of training and testing data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_h)
    X_test_scaled = scaler.transform(X_test_h)

    # Define the hyperparameter space
    dt_param_grid = {
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 3, 4]
    }

    # Perform grid search cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=dt_param_grid, cv=5)
    grid_search.fit(X_train_scaled, Y_train_h)

    # Evaluate the best model on the test data
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    score = best_model.score(X_test_scaled, Y_test_h)

    # Print the results
    print('dtB Best score:', grid_search.best_score_)
    print('dtB Best hyperparameters:', grid_search.best_params_)
    print('Model type:', type(best_model))

    # Return the best model and hyperparameters
    return best_model, grid_search.best_params_
