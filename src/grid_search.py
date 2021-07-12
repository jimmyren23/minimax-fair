# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from os import error
import numpy as np
import pandas as pd
from src.minmaxML import rescale_feature_matrix, create_index_array, compute_model_errors, 
from src.train_test_split import create_fold_split, create_validation_split
from itertools import product
import warnings
from src.torch_wrapper import MLPRegressor

# class GridSearchResult:
class GridSearch():
    def __init__(self, param_grid = None, cv = 5):
        """
        :param param_grid: dictionary with keys being parameters and values being a list of possible parameter values
        :param cv: number of folds for cross-validation, set to 5 as default
        :param error_metric: list of types of error
        :
        """
        self.param_grid = param_grid # dictionary of parameters to a list of possible parameter values
        self.allCombos = self.paramCombos(param_grid) # creates a list of all possible parameter metrics
        self.cv = cv # number of folds for cross validation
        self.score_error = [] # initialize an array containing avg population error
        self.min_error_index = 0 # index of parameter settings that produced the minimum error for EACH metric

    def paramCombos(self, param_grid):
        allCombos = []
        for params in param_grid:
            items = sorted(params.items())
            if not items:
                print("WARNING: No parameters were listed.")
            else:
                keys, values = zip(*items) # unpacks the entries and puts them into key, values tuples
                for v in product(*values): # product gets the cartesian product of the values
                    params = dict(zip(keys, v)) # zips the values and keys back up into a dict for our model to later use
                    allCombos.append(params) # stores the local variable states of the params we want to use for our models
        return allCombos   

    def fit(self, X, y, numsteps, grouplabels, a=1, b=0.5, scale_eta_by_label_range=True, rescale_features=True,
                model_type='LinearRegression', error_type='Total',
                extra_error_types=set(), pop_error_type='Total',
                max_logi_iters=100, tol=1e-8, fit_intercept=True, logistic_solver='lbfgs',
                lr=0.01, momentum=0.9, weight_decay=0, n_epochs=10000, hidden_sizes=(2 / 3,), 
                batchsize=32, minibatch=False,
                group_names=(), group_types=(), 
                display_plots=True, verbose=False, use_input_commands=True,
                normalize_labels = False, test_size = 0.2, random_split_seed=0):

        # initially splits the data for validation
        X_train, _ , y_train, _ , grouplabels_train, _ = \
            create_validation_split(X, y, grouplabels, test_size, random_seed=random_split_seed)

        # Fr each parameter combination, we'll run cross_validation
        for param_combination in self.allCombos:
            for param, value in param_combination.items():
                # Set parameters for the model
                if "lr" == param:
                    lr = value
                elif "n_epochs" == param:
                    n_epochs = value
                elif "numsteps" == param:
                    numsteps = value
                elif "momentum" == param:
                    momentum = value
                elif "weight_decay" == param:
                    weight_decay =weight_decay
                elif "hidden_sizes" == param:
                    hidden_sizes = hidden_sizes
                else:
                    print("ERROR: Invalid parameters inserted into grid search.")
                    quit()
            ## Performs cross_validation on the current set of hyperparameters. This will return a list of error for each metric
            avg_pop_error = self.cross_validation(X_train, y_train, numsteps, grouplabels_train, a, b, scale_eta_by_label_range=scale_eta_by_label_range,
                model_type=model_type, error_type=error_type,
                extra_error_types=extra_error_types, pop_error_type=pop_error_type,
                max_logi_iters=max_logi_iters, tol=tol, fit_intercept=fit_intercept, logistic_solver=logistic_solver,
                lr=lr, momentum=momentum, weight_decay=weight_decay, n_epochs=n_epochs, hidden_sizes=hidden_sizes, 
                batchsize=batchsize, minibatch=minibatch,
                group_names=group_names, group_types=group_types, 
                display_plots=display_plots, verbose=verbose, use_input_commands=use_input_commands,
                normalize_labels=normalize_labels) # an nx1 list, where n is the number of types of error

            self.score_error.append(avg_pop_error) # appends the error to the appropriate list

        self.min_error_index = np.nanargmin(self.score_error) # gets the first index with the minimum error and returns its parameters, possibly change to gets min, breaks ties with pop error
        
    def cross_validation(self, X, y, numsteps, grouplabels, a, b=0.5, scale_eta_by_label_range=True,
                model_type='LinearRegression', error_type='Total',
                extra_error_types=set(), pop_error_type='Total',
                max_logi_iters=100, tol=1e-8, fit_intercept=True, logistic_solver='lbfgs', 
                lr=0.01, momentum=0.9, weight_decay=0, n_epochs=10000, hidden_sizes=(2 / 3,), 
                batchsize=32, minibatch=False,
                group_names=(), group_types=(),
                display_plots=True, verbose=False, use_input_commands=True,
                normalize_labels = False):
        """
        :param X:  numpy matrix of features with dimensions numsamples x numdims
        :param y:  numpy array of labels with length numsamples. Should be numeric (0/1 labels binary classification)
        :param estimator: model (do_learning?)
        :param k:  number of folds
        returns: the average error for each fold for each type of error
        """
        # list of the error for each type of error for each fold
        fold_error_list = []
        
        for X_train, X_test, y_train, y_test, grouplabels_train, grouplabels_test in create_fold_split(X, y, grouplabels, self.cv):
            model_pop_error = self.compute_model(X_train, y_train, X_test, y_test, numsteps, grouplabels, grouplabels_train, grouplabels_test, a, b, scale_eta_by_label_range=scale_eta_by_label_range,
                model_type=model_type, error_type=error_type,
                extra_error_types=extra_error_types, pop_error_type=pop_error_type,
                max_logi_iters=max_logi_iters, tol=tol, fit_intercept=fit_intercept, logistic_solver=logistic_solver,
                lr=lr, momentum=momentum, weight_decay=weight_decay, n_epochs=n_epochs, hidden_sizes=hidden_sizes, 
                batchsize=batchsize, minibatch=minibatch,
                group_names=group_names, group_types=group_types, 
                display_plots=display_plots, verbose=verbose, use_input_commands=use_input_commands,
                normalize_labels=normalize_labels)
            fold_error_list.append(model_pop_error)

        return np.sum(fold_error_list) / 5 # return the average error for this experiment for each type of error, which is an 2x1 array ("max group error", "pop_group_error")

    def compute_model(self, X_train, y_train, X_test, y_test, numsteps, 
                grouplabels, grouplabels_train, grouplabels_test, a=1, b=0.5, 
                scale_eta_by_label_range=True, rescale_features=True,
                model_type='LinearRegression', error_type='Total',
                extra_error_types=set(), pop_error_type='Total',
                max_logi_iters=100, tol=1e-8, fit_intercept=True, logistic_solver='lbfgs',
                lr=0.01, momentum=0.9, weight_decay=0, n_epochs=10000, hidden_sizes=(2 / 3,), 
                batchsize=32, minibatch=False,
                group_names=(), group_types=(),
                display_plots=True, verbose=False, use_input_commands=True,
                normalize_labels = False):
        """
        :param X:  numpy matrix of features with dimensions numsamples x numdims
        :param y:  numpy array of labels with length numsamples. Should be numeric (0/1 labels binary classification)
        :param numsteps:  number of rounds to run the game
        :param a, b:  parameters for eta = a * t ^ (-b)
        :param scale_eta_by_label_range: if the inputted a value should be scaled by the max absolute label value squared
        :param rescale_features: Whether or not feature values should be rescaled for numerical stability
        :param grouplabels:  numpy array of numsamples numbers in the range [0, numgroups) denoting groups membership
        :param group_names:  list of groups names in relation to underlying data (e.g. [male, female])
        :param extra_error_types: set of error types which we want to plot
        :param pop_error_type: error type to use on population e.g. Total for FP/FN
        :param logistic_solver: Which underlying solver to use for logistic regression
        :param fit_intercept: Whether or not we should fit an additional intercept
        :param max_logi_iters: max number of logistic regression iterations
        :param tol: tolerance of convergence for logistic regression
        :param lr: learning rate of gradient descent for MLP
        :param n_epochs: number of epochs per individual MLP model
        :param hidden_sizes: list of sizes for hidden layers of MLP - fractions (and 1) treated as proportions of numdims
        :param batchsize: if the model uses minibatch, then this will be the size of the minibatches
        :param minibatch: denotes if the model uses minibatch gradient descent or batch gradient descent
        :param normalize_labels: denotes if the y labels should be normalized 
        """
        
        # normalizes labels if necessary
        if normalize_labels:
            y_train = y_train / max(y_train)
            y_test = y_test / max(y_test)

        if not use_input_commands and display_plots:
            warnings.warn('WARNING: use_input_commands is set to False. '
                        'This may cause plots to appear and immediately dissappear when running code from the command '
                        'line.')

        # Rescales features to be within [-100, 100] for numerical stability
        if rescale_features:
            X_test = rescale_feature_matrix(X_test)
            X_train = rescale_feature_matrix(X_train)

        # Divide eta (via scaling a) by the max label value squared. Equivalent to linearly scaling labels to range [-1, 1]
        if scale_eta_by_label_range:
            a /= max(abs(y_train)) ** 2

        # Hacky way to adjust for the fact that numsteps is 1 fewer than we want it to be because of 1 indexing
        numsteps += 1


        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)


        # Denotes whether or not each person belongs to multiple groups
        num_group_types = grouplabels.shape[0]

        # Array of 'numgroups' arrays, one for each groups category
        numgroups = np.array([np.size(np.unique(grouplabels[i])) for i in range(num_group_types)])

        # set grouplabels to train and test
        grouplabels_train, grouplabels_test = grouplabels_train, grouplabels_test

        # Compute features about the data
        numsamples, numdims = X_train.shape
        val_numsamples, _ = X_test.shape

        # Setup arrays storing the indices of each individual groups for ease of use later
        groupsize = [np.array([]) for _ in range(num_group_types)]
        index = [[] for _ in range(num_group_types)]  # List of lists of slices
        for i in range(num_group_types):
            # index[i] is a "list" of length numgroups[i] whose elements are slices for (np.where label == g)
            groupsize[i], index[i] = create_index_array(numgroups[i], grouplabels_train[i])

        # Repeat the above for validation
        val_groupsize = [np.array([]) for _ in range(num_group_types)]
        val_index = [[] for _ in range(num_group_types)]
        for i in range(num_group_types):
            val_groupsize[i], val_index[i] = create_index_array(numgroups[i], grouplabels_test[i])

        # Instatiate all error arrays
        errors = np.zeros((numsteps, numsamples))  # Stores error for each member of pop for each round

        # Do for validation (val)
        val_errors = np.zeros((numsteps, val_numsamples))


        error_type = pop_error_type = 'MSE'  # Rename the 'total' error type to MSE in regression case

        # Dictionary of arrays storing the errors of each type, can use other functions to compute over rounds
        # Instantiate dictionaries with the error type we are reweighting on
        specific_errors = {error_type: errors}
        val_specific_errors = {error_type: val_errors}

        # Converting empty dictionary to set makes it easier to use set literals in main_drivers
        if extra_error_types == {}:
            extra_error_types = set()

        # Ensure we do not duplicate/overwrite the main error type as an extra error type
        if error_type in extra_error_types:
            extra_error_types.remove(error_type)

        # If pop_error_type is unspecified and not caught in the above cases, let it be the regular error type
        if pop_error_type == '':
            pop_error_type = error_type

        # Create a new array to store the errors of each type
        for extra_err_type in extra_error_types:
            specific_errors[extra_err_type] = np.zeros((numsteps, numsamples))
            val_specific_errors[extra_err_type] = np.zeros((numsteps, val_numsamples))

        if verbose:
            # print(f'Group labels are: {grouplabels}')
            print(f'Group names are:   {group_names}')
            print(f'Group types are:   {group_types}')
            print(f'Group sizes (train): {groupsize}')
            print(f'Group sizes (val):   {val_groupsize}')

        # Initialize sample weights and groups weights for Regulator
        groupweights = [np.zeros((numsteps, numgroups[i])) for i in range(num_group_types)]
        p = [np.array([]) for _ in range(num_group_types)]
        sampleweights = [np.array([]) for _ in range(num_group_types)]
        # Fill the weight arrays as necessary
        for i in range(num_group_types):
            p[i] = groupsize[i] / numsamples  # Compute each groups proportion of the population
            groupweights[i][0] = p[i]
            sampleweights[i] = np.ones(numsamples) / numsamples  # Initialize sample weights array to uniform
        # Convert sampleweights to numpy array since it's rectangular,
        sampleweights = np.array(sampleweights)
        avg_sampleweights = np.squeeze(np.sum(sampleweights, axis=0) / num_group_types)

        if verbose:
            print(f'Starting simulation with the following paramters: \n' +
                f'model_type: {model_type} \n' +
                f'numsamples: {numsamples} \n' +
                f'numdims: {numdims} \n' +
                f'numgroups: {numgroups} \n' +
                f'numsteps: {numsteps - 1} \n' +
                f'a: {a} \n' +
                f'b: {b} \n')
            if model_type == 'LogisticRegression':
                print('fit_intercept:', fit_intercept)
                print('solver', logistic_solver)
                print('max_iterations:', max_logi_iters)
                print('tol:', tol)

        hidden_sizes = [numdims] + \
                        list(map(lambda x: x if np.floor(x) == x else int(np.floor(x * numdims)), hidden_sizes))
        
        # Create unweighted model
        modelhat = MLPRegressor(hidden_sizes, lr=lr, momentum=momentum, weight_decay=weight_decay). \
            fit(X_train, y_train, avg_sampleweights, batchsize, minibatch, n_epochs=n_epochs)

        # Compute the errors of the model according to the specified loss function
        # Updates errors array with the round-specific errors for each person for round t
        compute_model_errors(modelhat, X_train, y_train, 1, errors, 'MSE')
        compute_model_errors(modelhat, X_test, y_test, 1, val_errors, 'MSE')

        # # Compute groups error rates for each groups this round across each type of groups
        # for i in range(num_group_types):
        #     update_group_errors(numgroups[i], 1, errors, grouperrs[i], agg_grouperrs[i], index[i],
        #                         groupsize_err_type[i])
        #     update_group_errors(numgroups[i], 1, val_errors, val_grouperrs[i], val_agg_grouperrs[i],
        #                         val_index[i], val_groupsize_err_type[i])
                                
        # returns the maximum group error and the average population error
        # return [max(val_grouperrs[0][1]), np.sum(val_errors[1]) / numsamples] 
        return np.sum(val_errors[1]) / numsamples

    def get_best_model(self):
        min_error = self.score_error[self.min_error_index]
        min_param_combo = self.allCombos[self.min_error_index]
        print(f'Minimum Average Population Error was {min_error}.')
        print(f'The best parameter combination was ({min_param_combo}).')



