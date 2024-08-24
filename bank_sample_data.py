import pandas as pd
import numpy as np


def sample_data(file_path, random_seed=None, positive_size=1.0, negative_size=1.0):
    # Load the data
    df = pd.read_csv(file_path, sep=';')

    # Perform one-hot encoding
    df = pd.get_dummies(df,
                        columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
                                 'month', 'poutcome'], drop_first=True)

    # # Specify the columns you want to keep
    # selected_columns = ["duration", "job_student", "housing_yes", "contact_unknown",
    #                     "month_dec", "month_jun", "month_mar", "month_oct",
    #                     "month_sep", "poutcome_success", "y"]
    #
    # # Keep only the selected columns
    # df = df[selected_columns]

    # Extract indices where y == 'yes' and y == 'no'
    indices_yes = df.index[df['y'] == 'yes'].tolist()
    indices_no = df.index[df['y'] == 'no'].tolist()

    # Calculate the number of samples to extract
    num_yes_samples = int(len(indices_yes) * positive_size)
    num_no_samples = int(len(indices_no) * negative_size)

    if random_seed is not None:
        np.random.seed(random_seed)
    # Randomly sample indices
    sampled_indices_yes = np.random.choice(indices_yes, size=num_yes_samples, replace=False)
    sampled_indices_no = np.random.choice(indices_no, size=num_no_samples, replace=False)

    # Combine sampled indices
    sampled_indices = np.concatenate([sampled_indices_yes, sampled_indices_no])

    # Get the sampled data
    sampled_data = df.loc[sampled_indices].reset_index(drop=True)

    # Convert 'y' column to numeric (if not already)
    sampled_data['y'] = sampled_data['y'].map({'yes': 1, 'no': 0})

    # Extract features and labels
    X_sampled = sampled_data.drop('y', axis=1)
    y_sampled = sampled_data['y']

    return X_sampled, y_sampled