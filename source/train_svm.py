
from __future__ import print_function
import argparse
import os
import pandas as pd
import joblib
from sklearn.svm import SVC

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
    parser.add_argument('--gamma', default='scale')
    parser.add_argument('--C', type=float, default=1.0)
    
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
    model_obj = SVC(class_weight='balanced',
                    gamma = args.gamma,
                    C = args.C)
    
    ## Train the model
    model_obj.fit(train_x, train_y)
    
    ## Save the trained model
    joblib.dump(model_obj, os.path.join(args.model_dir, "model.joblib"))
