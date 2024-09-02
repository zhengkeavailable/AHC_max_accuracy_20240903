# from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from scipy.io import arff


def sample_data(file_path=None, random_seed=None, positive_size=1.0, negative_size=1.0):
    # rice_cammeo_and_osmancik = fetch_ucirepo(id=545)
    # features = rice_cammeo_and_osmancik.data.features
    # targets = rice_cammeo_and_osmancik.data.targets
    # # Convert targets to DataFrame (assuming targets is a 1D array)
    # df_targets = pd.DataFrame(targets, columns=['Class'])
    # # Combine features and targets into a single DataFrame
    # df = pd.concat([features, df_targets], axis=1)
    # df.to_csv('rice_cammeo_osmancik.csv', index=False)
    df = pd.read_csv('rice_cammeo_osmancik.csv',encoding='utf-8')

    # Extract indices where Class == 1 (Cammeo) and Class == 0 (Osmancik)
    indices_cammeo = df.index[df['Class'] == 'Cammeo'].tolist()
    indices_osmancik = df.index[df['Class'] == 'Osmancik'].tolist()

    # Calculate the number of samples to extract
    num_osmancik_samples = int(len(indices_osmancik) * positive_size)
    num_cammeo_samples = int(len(indices_cammeo) * negative_size)

    if random_seed is not None:
        np.random.seed(random_seed)

    # Randomly sample indices
    sampled_indices_cammeo = np.random.choice(indices_cammeo, size=num_cammeo_samples, replace=False)
    sampled_indices_osmancik = np.random.choice(indices_osmancik, size=num_osmancik_samples, replace=False)

    # Combine sampled indices
    sampled_indices = np.concatenate([sampled_indices_cammeo, sampled_indices_osmancik])

    # Get the sampled data
    sampled_data = df.loc[sampled_indices].reset_index(drop=True)
    sampled_data['Class'] = sampled_data['Class'].map({'Osmancik': 1, 'Cammeo': 0})
    # Extract features and Classs
    X_sampled = sampled_data.drop('Class', axis=1)
    y_sampled = sampled_data['Class']

    return X_sampled, y_sampled

