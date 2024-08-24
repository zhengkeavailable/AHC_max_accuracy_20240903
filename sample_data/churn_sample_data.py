from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np


def sample_data(file_path=None, random_seed=None, positive_size=1.0, negative_size=1.0):
    iranian_churn = fetch_ucirepo(id=563)
    features = iranian_churn.data.features
    targets = iranian_churn.data.targets
    # Convert targets to DataFrame (assuming targets is a 1D array)
    df_targets = pd.DataFrame(targets, columns=['Churn'])

    # Combine features and targets into a single DataFrame
    df = pd.concat([features, df_targets], axis=1)

    # Extract indices where Class == 1 (Churn) and Class == 0 (Non-churn)
    indices_churn = df.index[df['Churn'] == 1].tolist()
    indices_non_churn = df.index[df['Churn'] == 0].tolist()

    # Calculate the number of samples to extract
    num_churn_samples = int(len(indices_churn) * positive_size)
    num_non_churn_samples = int(len(indices_non_churn) * negative_size)

    if random_seed is not None:
        np.random.seed(random_seed)

    # Randomly sample indices
    sampled_indices_churn = np.random.choice(indices_churn, size=num_churn_samples, replace=False)
    sampled_indices_non_churn = np.random.choice(indices_non_churn, size=num_non_churn_samples, replace=False)

    # Combine sampled indices
    sampled_indices = np.concatenate([sampled_indices_churn, sampled_indices_non_churn])

    # Get the sampled data
    sampled_data = df.loc[sampled_indices].reset_index(drop=True)
    # sampled_data['Churn'] = sampled_data['Churn'].map({1: 1, 0: 0})
    # Extract features and Classs
    X_sampled = sampled_data.drop(['Age Group','Churn'], axis=1)
    y_sampled = sampled_data['Churn']

    return X_sampled, y_sampled

