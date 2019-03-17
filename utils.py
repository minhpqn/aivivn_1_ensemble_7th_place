import pandas as pd


def load_csv_data(filepath, textcol="text"):
    """Load data from csv file

    Parameters
    -----------
    filepath: String
        Path to CSV data

    Return
    -----------
    samples: List
        list of samples
    labels:  List
        list of labels
    """
    df = pd.read_csv(filepath)
    samples = [ str(text) for text in df[textcol] ]
    labels  = [ str(intent) for intent in df["label"] ]

    return samples, labels
