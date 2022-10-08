import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_results(results_dir):
    # Create a Pandas dataframe to store all results
    columns = ['Method', 'Antigen', 'Seed', 'Num BB Evals', 'Suggest Time', 'Last Protein', 'Last Binding Energy',
               'Last Charge', 'Last Hydropathicity', 'Last Instability Index', 'Best Protein', 'Best Binding Energy']

    new_columns = ['Unnamed: 0', 'Index', 'LastValue', 'BestValue', 'Time', 'LastProtein',
                   'BestProtein']
    results = pd.DataFrame(columns=columns)

    # Get the name of every folder in the results directory
    folders = glob.glob(os.path.join(results_dir, '*'))

    reg = tqdm(folders)
    for folder in reg:
        method = folder.split('/')[-1]
        reg.set_description(method)
        subfolders = glob.glob(os.path.join(folder, '*'))

        for subfolder in subfolders:
            if os.path.isdir(subfolder):
                try:
                    _name = subfolder.split('/')[-1]
                    # print(_name.split('antigen_')[1].split('_kernel'))
                    antigen = _name.split('antigen_')[1].split('_kernel')[0]
                    seed = int(_name.split('_seed_')[1].split('_cdr_')[0])
                    # method = _name.split('_kernel_')[1].split('_seed_')[0]
                    if not os.path.exists(os.path.join(subfolder, 'results.csv')):
                        continue
                    df = pd.read_csv(os.path.join(subfolder, 'results.csv'))

                    if len(df.columns) == len(new_columns) and np.all(df.columns == new_columns):
                        aux_df = pd.DataFrame(columns=columns)
                        aux_df["Num BB Evals"] = df["Index"] + 1
                        aux_df["Suggest Time"] = df["Time"]
                        aux_df["Last Protein"] = df["LastProtein"]
                        aux_df["Best Binding Energy"] = df["BestValue"]
                        aux_df["Last Binding Energy"] = df["LastValue"]
                        aux_df["Best Protein"] = df["BestProtein"]
                        df = aux_df
                except:
                    # shutil.rmtree(subfolder)
                    print(subfolder)
                    raise
                # Add method, antigen and seed column
                df['Method'] = len(df['Num BB Evals']) * [method]
                df['Antigen'] = len(df['Num BB Evals']) * [antigen]
                df['Seed'] = len(df['Num BB Evals']) * [seed]

                df = df[columns]

                # results.append(df, ignore_index=True)
                results = pd.concat([results, df], ignore_index=True, sort=False)

    return results
 
