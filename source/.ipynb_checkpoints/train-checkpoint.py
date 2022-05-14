
from __future__ import print_function
import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

def model_fn(model_dir):
    
    # Load model from the model_dir; this is the same model that is saved in the main if statement
    
    print("Loading model.")
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    return model


if __name__ == '__main__':
    
    # During a training job, all of the model parameters and training parameters are sent as arguments when this script is executed
    
    # Set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SM_OUTPUT_DATA_DIR: path to write output artifacts to (checkpoints, graphs, and other files to save, not including model artifacts):
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    # SM_OUTPUT_DATA_DIR: path to write model artifacts to:
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # SM_CHANNEL_TRAIN: path to the directory containing data in the ‘train’ channel
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## Model hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--max_leaf_nodes', type=int, default=None)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    
    ## args holds all passed-in arguments
    #args = parser.parse_args()
    args, _ = parser.parse_known_args()
    
    ## Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    ## Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    ## Define model
    model = RandomForestClassifier(n_estimators = args.n_estimators, 
                                   max_depth = args.max_depth,
                                   max_leaf_nodes = args.max_leaf_nodes,
                                   min_samples_split = args.min_samples_split,
                                   min_samples_leaf = args.min_samples_leaf)
    
    ## Train the model
    model.fit(train_x, train_y)
    
    ## Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
