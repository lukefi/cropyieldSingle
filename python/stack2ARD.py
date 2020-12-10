"""

23.10.2020 

Combine annual stack-files into one array stack.

combineAllYears() reads all annuals into one big dataframe.

reshapeAndSave() pivots the dataframe by farmID and doy, converts to numpy array, fills with na (-> not ragged) and reshapes into 3D. Saves array and farmIDs into separate files.

RUN: 

python stack2ARD.py -o /Users/myliheik/Documents/myCROPYIELD/dataStack -f test1110 

After this into run3DNeuralNet.py.

In Puhti: module load geopandas (Python 3.8.) and also: pip install 'pandas==1.1.2' --user

"""
import glob
import os
import pandas as pd
import numpy as np
import pickle

from pathlib import Path

import argparse
import textwrap
from datetime import datetime


###### FUNCTIONS:

def load_intensities(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def save_intensities(filename, arrayvalues):
    with open(filename, 'wb+') as outputfile:
        pickle.dump(arrayvalues, outputfile)

def combineAllYears(data_folder3, setti):
    # open all years into one dataframe:
    allyears = pd.concat(map(pd.read_pickle, glob.glob(os.path.join(data_folder3, setti) + '*.pkl')), sort=False)
    return allyears

def reshapeAndSave(full_array_stack, out_dir_path, outputfile):    
    # reshape and save data to 3D:
    print(f"\nLength of the data stack dataframe: {len(full_array_stack)}")
    # group dataframe by farmID and date
    dfpivot = full_array_stack.pivot(index=['farmID', 'doy'], columns='band', values=[*full_array_stack.columns[full_array_stack.columns.str.startswith('bin')]])
    # to numpy array and reshape into 3D:
    pivotarray = dfpivot.unstack().to_numpy()
    pivotarray3d = pivotarray.reshape(*dfpivot.index.levshape,-1)
    pivotarray3dfilled = np.nan_to_num(pivotarray3d)
    print(f"Shape of the 3D stack dataframe: {pivotarray3dfilled.shape}")
    print(f"Output into file: {os.path.join(out_dir_path,outputfile)}")
    save_intensities(os.path.join(out_dir_path,outputfile), pivotarray3dfilled)
    
    # save farmIDs for later merging with target y:
    farmIDs = dfpivot.index.levels[0].str.rsplit('_',1).str[0].values
    print(f"\n\nNumber of farms: {len(farmIDs)}")
    outputfile2 = 'farmID_' + outputfile
    fp = os.path.join(out_dir_path, outputfile2)
    print(f"Output farmIDs in file: {fp}")
    save_intensities(fp, farmIDs)
              
def main(args):
    
    try:
        if not args.outdir or not args.setti:
            raise Exception('Missing output dir argument or dataset label (e.g. test1110). Try --help .')

        print(f'\n\nstack2ARD.py')
        print(f'\nInput files in {args.outdir}_temp')
        
        out_dir_path = Path(os.path.expanduser(args.outdir))
        out_dir_path.mkdir(parents=True, exist_ok=True)

        # temp directory for temp annual results:
        data_folder3 = args.outdir + "_temp"
        Path(data_folder3).mkdir(parents=True, exist_ok=True)
        
        outputfile = args.setti + '.pkl'
        
        print("\nPresuming preprocessing done earlier. If not done previously, please, run with histo2stack.py first!")

        print("\nCombining all years...")
        allyears = combineAllYears(data_folder3, args.setti)
        reshapeAndSave(allyears, out_dir_path, outputfile)
        

    except Exception as e:
        print('\n\nUnable to read input or write out results. Check prerequisites and see exception output below.')
        parser.print_help()
        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent(__doc__))
    parser.add_argument('-o', '--outdir',
                        type=str,
                        help='Name of the output directory.',
                        default='.')
    parser.add_argument('-f', '--setti',
                        type=str,
                        help='Name of the data set.')
    args = parser.parse_args()
    main(args)



