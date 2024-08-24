from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np


def sample_data(file_path=None, random_seed=None, positive_size=1.0, negative_size=1.0):
    # fetch dataset
    adult = fetch_ucirepo(id=2)
    # data (as pandas dataframes)
    features = adult.data.features
    targets = adult.data.targets
    # Step 2: Handling missing values (if any)
    # Checking for missing values
    features = features.dropna()
    features.reset_index(drop=True, inplace=True)
    df_targets = pd.DataFrame(targets['income'][features.index.to_list()], columns=['income'])
    df_targets['income'] = df_targets['income'].str.rstrip('.')
    # Step 3: Encoding categorical features
    # Finding categorical columns
    categorical_cols = features.select_dtypes(include=['object']).columns
    # One-hot encoding categorical columns
    features = pd.get_dummies(features, columns=categorical_cols, drop_first=True)
    df = pd.concat([features, df_targets], axis=1)
    # df.to_csv('adult.csv', index=False)
    # Extract indices where Class == 1 (Cammeo) and Class == 0 (Osmancik)
    indices_morethan50k = df.index[df['income'] == '>50K'].tolist()
    indices_lessthan50k = df.index[df['income'] == '<=50K'].tolist()

    # Calculate the number of samples to extract
    num_morethan50k = int(len(indices_morethan50k) * positive_size)
    num_lessthan50k = int(len(indices_lessthan50k) * negative_size)

    if random_seed is not None:
        np.random.seed(random_seed)

    # Randomly sample indices
    sampled_morethan50k = np.random.choice(indices_morethan50k, size=num_morethan50k, replace=False)
    sampled_lessthan50k = np.random.choice(indices_lessthan50k, size=num_lessthan50k, replace=False)

    # Combine sampled indices
    sampled_indices = np.concatenate([sampled_morethan50k, sampled_lessthan50k])

    # Get the sampled data
    sampled_data = df.loc[sampled_indices].reset_index(drop=True)
    sampled_data['income'] = sampled_data['income'].map({'>50K': 1, '<=50K': 0})
    # Extract features and Classs
    X_sampled = sampled_data.drop('income', axis=1)
    y_sampled = sampled_data['income']

    return X_sampled, y_sampled
