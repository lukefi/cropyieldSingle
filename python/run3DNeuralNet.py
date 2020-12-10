"""
2020-10-22 MY
2020-11-19 MY added RNNsuperb
2020-11-24 MY added LSTMsuperb

# First merging farmID and crop yields to make target set y:
python run3DNeuralNet.py -i /Users/myliheik/Documents/myCROPYIELD/dataStack -j train1110 \
-k /Users/myliheik/Documents/myCROPYIELD/data -m 

# Then run classifier for train sets and test sets:
python run3DNeuralNet.py -i /Users/myliheik/Documents/myCROPYIELD/dataStack -j train1110 -n

OR full data aka RNNfull where all train sets (all crops) are combined:
python run3DNeuralNet.py -i /Users/myliheik/Documents/myCROPYIELD/dataStack -n -f

OR for production uses previous years and predicts the new year, RNNsuperb and LSTM superb:
python run3DNeuralNet.py -i /Users/myliheik/Documents/myCROPYIELD/dataStack -j train1110 \
-p -r /Users/myliheik/Documents/myCROPYIELD/dataStack2020

"""
import glob
import pandas as pd
import numpy as np
import pickle
import os.path
from pathlib import Path
import argparse
import textwrap
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt

# FUNCTIONS:

def load_intensities(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def save_intensities(filename, arrayvalues):
    with open(filename, 'wb+') as outputfile:
        pickle.dump(arrayvalues, outputfile)

def makeTarget(stack_dir, setti, refe_dir):
    # read farmIDs:
    fp = 'farmID_' + setti + '.pkl'
    ids = load_intensities(os.path.join(stack_dir, fp))
    idsdf = pd.DataFrame(ids)
    idsdf.columns = ['farmID']
    # read crop yields (target):
    alku = setti[:-4]
    loppu = setti[-4:]
    fp = alku + 'y' + loppu + '.csv'
    targets = pd.read_csv(os.path.join(refe_dir, fp))
    # merge:
    df = idsdf.merge(targets)
    df['y'] = df[df.columns[df.columns.str.endswith('ha')][0]].astype(float).round(0)
    dfnew = df[['farmID', 'y']]
    # save:
    fp = setti + 'y.pkl'
    print(f'Saving target y to {os.path.join(stack_dir, fp)}.')
    save_intensities(os.path.join(stack_dir, fp), df['y'])

    
def readData(stack_dir, setti):
    trainsetti = 'train' + setti[-4:] 
    testsetti = 'test' + setti[-4:] 
    
    fp = trainsetti + '.pkl'
    print(f'Read in train set {fp}')
    Xtrain = load_intensities(os.path.join(stack_dir, fp))
    fp = trainsetti + 'y.pkl'
    ytrain = load_intensities(os.path.join(stack_dir, fp))
    fp = testsetti + '.pkl'
    print(f'Read in test set {fp}')
    Xtest = load_intensities(os.path.join(stack_dir, fp))
    fp = testsetti + 'y.pkl'
    ytest = load_intensities(os.path.join(stack_dir, fp))

    print(f"Shape of training set: {Xtrain.shape, ytrain.shape}")
    print(f"Shape of testing set: {Xtest.shape, ytest.shape}")
    
    return Xtrain, ytrain, Xtest, ytest
    
    
def classifier(Xtrain, ytrain, Xtest, ytest, outputdir):
    # SimpleRNN for train sets and test sets
    print("\nTraining fully-connected RNN...")
    model = Sequential()
    model.add(SimpleRNN(128, return_sequences=True, input_shape = (Xtrain.shape[1],
        Xtrain.shape[2]), activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))

    model.compile(loss = 'mean_squared_error',
                  optimizer = 'adam',
                  metrics = ['mse'])
    
    # Model summary:
    print('\nNetwork architecture:')
    print(model.summary())
    
    fname = os.path.join(outputdir, 'networkArchitecture.pdf')
    print(f'\nSaving network architecture into file {fname}')
    plot_model(model, to_file=fname, show_shapes=True, rankdir='LR')

    # monitor validation progress
    early = EarlyStopping(monitor = "val_loss", mode = "min", patience = 20)
    callbacks_list = [early]
    
    # and train the model
    history = model.fit(Xtrain, ytrain,
        epochs=50, batch_size=25, verbose=0,
        validation_split = 0.20,
        callbacks = callbacks_list)


    test_predictions = model.predict(Xtest)

    # Evaluate the model on the test data
    print('\nEvaluating the model on the test set:')
    results = model.evaluate(Xtest, ytest, batch_size=128)
    print("\nTest set loss, test set acc:", results)
    print("RMSE: ", math.sqrt(results[1]))
    
    # Saving training history into file...
    fname = os.path.join(outputdir, 'trainingHistory.pdf')
    print(f'\nSaving training history into file {fname}...\n')
    fig = plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.savefig(fname, dpi=300)

    dfPreds = pd.DataFrame(test_predictions[:, :, 0])
    # starts 10.5. eli DOY 130
    # Predictions to compare with forecasts: 15.6. eli DOY 166, that is pythonic 165.
    # and 15.7. eli DOY 196 
    # and 15.8. eli DOY 227
    # and the last DOY 243 -> the final state
    dfPredsFinal = dfPreds.iloc[:,[166-129, 196-129, 227-129, 243-129]]
    
    predfile = os.path.join(outputdir, 'RNNPreds.pkl')
    print(f'Saving predictions on test set into {predfile}...\n')
    save_intensities(predfile, dfPredsFinal)

def readFullData(stack_dir):
    # concatenate all sets into one training set:
    

    for setti in ['train1110', 'train1120', 'train1230', 'train1310', 'train1320', 'train1400']:
        trainsetti = 'train' + setti[-4:] 
        fp = trainsetti + '.pkl'
        print(f'Read in train set {fp}')
        if setti == 'train1110':
            Xtrain = load_intensities(os.path.join(stack_dir, fp))
            fp = trainsetti + 'y.pkl'
            ytrain = load_intensities(os.path.join(stack_dir, fp))
        
        if not setti == 'train1110':
            Xtrain1 = load_intensities(os.path.join(stack_dir, fp))
            fp = trainsetti + 'y.pkl'
            ytrain1 = load_intensities(os.path.join(stack_dir, fp))
            
            Xtrain = np.concatenate((Xtrain, Xtrain1), axis = 0)
            ytrain = np.concatenate((ytrain, ytrain1), axis = 0)
    print(f"Shape of the full training set: {Xtrain.shape, ytrain.shape}")
    return Xtrain, ytrain
    
def fullclassifier(Xtrain, ytrain, outputdir, mname):
    # SimpleRNN for combined train sets (all crops):
    print("\nTraining fully-connected RNN on full data set...")

    model = Sequential()
    model.add(SimpleRNN(128, return_sequences=True, input_shape = (Xtrain.shape[1],
        Xtrain.shape[2]), activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))

    model.compile(loss = 'mean_squared_error',
                  optimizer = 'adam',
                  metrics = ['mse'])
    
    # Model summary:
    print('\nNetwork architecture:')
    print(model.summary())
    
    
    outputdirmodel = os.path.join(outputdir, mname)
    Path(outputdirmodel).mkdir(parents=True, exist_ok=True)

    fname = os.path.join(outputdirmodel, 'networkArchitectureFullDataset.pdf')
    print(f'\nSaving network architecture into file {fname}')
    plot_model(model, to_file=fname, show_shapes=True, rankdir='LR')

    # monitor validation progress
    early = EarlyStopping(monitor = "val_loss", mode = "min", patience = 20)
    callbacks_list = [early]
    
    # and train the model
    history = model.fit(Xtrain, ytrain,
        epochs=50, batch_size=25, verbose=0,
        validation_split = 0.20,
        callbacks = callbacks_list)

    # Saving the model:
    print(f'\nSaving the model into {outputdirmodel}')
    model.save(outputdirmodel)
    
    # Saving training history into file...
    fname = os.path.join(outputdir, 'trainingHistoryFulldata.pdf')
    print(f'\nSaving training history into file {fname}...\n')
    fig = plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.savefig(fname, dpi=300)
    
    return model

def partialPredictions(model, stack_dir, outputdir):
    
    for setti in ['train1110', 'train1120', 'train1230', 'train1310', 'train1320', 'train1400']:

        testsetti = 'test' + setti[-4:] 

        fp = testsetti + '.pkl'
        print(f'Read in test set {fp}')
        Xtest = load_intensities(os.path.join(stack_dir, fp))
        fp = testsetti + 'y.pkl'
        ytest = load_intensities(os.path.join(stack_dir, fp))

        print(f"Shape of testing set: {Xtest.shape, ytest.shape}")

        # Evaluate the model on the test data
        results = model.evaluate(Xtest, ytest, batch_size=128)
        print("Setti: ", setti)
        print("RMSE: ", math.sqrt(results[1]))

        test_predictions = model.predict(Xtest)
    
        dfPreds = pd.DataFrame(test_predictions[:, :, 0])
        # starts 10.5. eli DOY 130
        # Predictions to compare with forecasts: 15.6. eli DOY 166, that is pythonic 165.
        # and 15.7. eli DOY 196 
        # and 15.8. eli DOY 227
        # and the last DOY 243 -> the final state
        dfPredsFinal = dfPreds.iloc[:,[166-129, 196-129, 227-129, 243-129]]
       
        preddir = os.path.join(outputdir, testsetti)
        Path(preddir).mkdir(parents=True, exist_ok=True)
        predfile = os.path.join(preddir, 'RNNPredsFull.pkl')
        print(f'Saving predictions on test set into {predfile}...\n')
        save_intensities(predfile, dfPredsFinal)

def superbPredictions(setti, mname, stack_dir, outputdir, outputdir2):
    model = load_model(os.path.join(outputdir, mname))
    
    testsetti = 'test' + setti[-4:] 

    fp = testsetti + '.pkl'
    print(f'Read in test set {fp}')
    Xtest = load_intensities(os.path.join(stack_dir, fp))
    #fp = testsetti + 'y.pkl'
    #ytest = load_intensities(os.path.join(stack_dir, fp))

    print(f"Shape of testing set: {Xtest.shape}")
    
    if Xtest.shape[1] < 115:
        doysToAdd = 115 - Xtest.shape[1]
        print(f"Shape of testing set differs from training set. We need to pad it with {doysToAdd} DOYs.")
        b = np.zeros( (Xtest.shape[0],doysToAdd,Xtest.shape[2]) )
        Xtest = np.column_stack((Xtest,b))
        print(f'New shape of Xtest is {Xtest.shape}.')
        
    test_predictions = model.predict(Xtest)

    dfPreds = pd.DataFrame(test_predictions[:, :, 0])
    # starts 10.5. eli DOY 130
    # Predictions to compare with forecasts: 15.6. eli DOY 166, that is pythonic 165.
    # and 15.7. eli DOY 196 
    # and 15.8. eli DOY 227
    # and the last DOY 243 -> the final state
    dfPredsFinal = dfPreds.iloc[:,[166-129, 196-129, 227-129, 243-129]]

    preddir = os.path.join(outputdir2, testsetti)
    Path(preddir).mkdir(parents=True, exist_ok=True)
    predfile = os.path.join(preddir, mname + 'PredsSuperb.pkl')
    print(f'Saving predictions on test set into {predfile}...\n')
    save_intensities(predfile, dfPredsFinal)
        
def readHistoryData(setti, stack_dir):
    # concatenate all sets into one training set:

    trainsetti = 'train' + setti[-4:] 
    fp = trainsetti + '.pkl'
    print(f'Read in train and test set {fp}')

    Xtrain1 = load_intensities(os.path.join(stack_dir, fp))
    fp = trainsetti + 'y.pkl'
    ytrain1 = load_intensities(os.path.join(stack_dir, fp))

    trainsetti = 'test' + setti[-4:] 
    fp = trainsetti + '.pkl'
    Xtrain2 = load_intensities(os.path.join(stack_dir, fp))
    fp = trainsetti + 'y.pkl'
    ytrain2 = load_intensities(os.path.join(stack_dir, fp))

    Xtrain = np.concatenate((Xtrain1, Xtrain2), axis = 0)
    ytrain = np.concatenate((ytrain1, ytrain2), axis = 0)
    print(f"Shape of the full training set: {Xtrain.shape, ytrain.shape}")
    return Xtrain, ytrain

def rivalClassifier(Xtrain, ytrain, outputdir, mname):
    print("\nTraining LSTM...")

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))

    # monitor validation progress
    early = EarlyStopping(monitor = "val_loss", mode = "min", patience = 10)
    callbacks_list = [early]

    model.compile(loss = 'mean_squared_error',
                  optimizer = 'adam',
                  metrics = ['mse'])

    # and train the model
    history = model.fit(Xtrain, ytrain,
        epochs=200,  batch_size=128, verbose=0,
        validation_split = 0.20,
        callbacks = callbacks_list)

    # Model summary:
    print('\nNetwork architecture:')
    print(model.summary())

    outputdirmodel = os.path.join(outputdir, mname)
    Path(outputdirmodel).mkdir(parents=True, exist_ok=True)
    fname = os.path.join(outputdirmodel, 'networkArchitectureFullDataset.pdf')
    print(f'\nSaving network architecture into file {fname}')
    plot_model(model, to_file=fname, show_shapes=True, rankdir='LR')

    # Saving the model:
    print(f'\nSaving the model into {outputdirmodel}')
    model.save(outputdirmodel)

    # Saving training history into file...
    fname = os.path.join(outputdirmodel, 'trainingHistoryFulldata.pdf')
    print(f'\nSaving training history into file {fname}...\n')
    fig = plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.savefig(fname, dpi=300)


        
# HERE STARTS MAIN:

def main(args):
    try:
        if not args.stack_dir :
            raise Exception('Missing data set directory argument. Try --help .')

        print(f'\nrunClassifier3D.py')
        print(f'\nARD data set in: {args.stack_dir}')

        if args.makeTarget:
            print("\nMerging farmID and crop yields to make target set y...")
            makeTarget(args.stack_dir, args.setti, args.refe_dir)
        if args.savePreds:  
            basepath = args.stack_dir.split('/')[:-1]
            out_dir_results = os.path.join(os.path.sep, *basepath, 'RNNpreds', args.setti)
            Path(out_dir_results).mkdir(parents=True, exist_ok=True)
            out_dir_results2 = os.path.join(os.path.sep, *basepath, 'RNNpreds')

            print(f"\nRunning classifier and saving test set predictions into {out_dir_results}.")
            
            if not args.full:
                Xtrain, ytrain, Xtest, ytest = readData(args.stack_dir, args.setti)
                classifier(Xtrain, ytrain, Xtest, ytest, out_dir_results, savePred=args.savePreds)
            if args.full:
                mname = 'RNNfull'
                Xtrain, ytrain = readFullData(args.stack_dir)
                model = fullclassifier(Xtrain, ytrain, out_dir_results2, mname)
                partialPredictions(model, args.stack_dir, out_dir_results2)
                
        if args.production:
            basepath = args.stack_dir.split('/')[:-1]
            out_dir_results = os.path.join(os.path.sep, *basepath, 'RNNpreds', args.setti)
            Path(out_dir_results).mkdir(parents=True, exist_ok=True)
            out_dir_results2 = os.path.join(os.path.sep, *basepath, 'RNNpreds')
            # Read and combine train and test data sets from previous years:
            Xtrain, ytrain = readHistoryData(args.setti, args.stack_dir)
            mname = 'RNNsuperb'
            #model = fullclassifier(Xtrain, ytrain, out_dir_results, mname)
            # Make predictions for new season:
            #superbPredictions(args.setti, mname, args.stack_dir2, out_dir_results, out_dir_results2)
            mname = 'LSTMsuperb'
            #model = rivalClassifier(Xtrain, ytrain, out_dir_results, mname)
            # Make predictions for new season:
            #superbPredictions(args.setti, mname, args.stack_dir2, out_dir_results, out_dir_results2)

        print(f'\nDone.')

    except Exception as e:
        print('\n\nUnable to read input or write out statistics. Check prerequisites and see exception output below.')
        parser.print_help()
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent(__doc__))

    parser.add_argument('-i', '--stack_dir',
                        help='Directory for stacked data.',
                        type=str,
                        default='.')
    parser.add_argument('-j', '--setti',
                        help='Name of the data set.',
                        type=str,
                        default='.')
    parser.add_argument('-k', '--refe_dir',
                        help='Directory for reference data.',
                        type=str,
                        default='.')
    
    parser.add_argument('-m', '--makeTarget',
                        help='Merge target values to farmID and save.',
                        default=False,
                        action='store_true')
        
    parser.add_argument('-n', '--savePreds',
                        help='Run classifier and save test set predictions.',
                        default=False,
                        action='store_true')
    parser.add_argument('-f', '--full',
                        help='Run classifier on full data set and save test set predictions separately, aka RNNfull.',
                        default=False,
                        action='store_true')
    parser.add_argument('-p', '--production',
                        help='Run classifier on historical data set and save test set predictions, aka RNNsuperb and LSTMsuperb.',
                        default=False,
                        action='store_true')
    parser.add_argument('-r', '--stack_dir2',
                        help='Directory for new season stacked data.',
                        type=str,
                        default='.')
    
    parser.add_argument('--debug',
                        help='Verbose output for debugging.',
                        action='store_true')

    args = parser.parse_args()
    main(args)
