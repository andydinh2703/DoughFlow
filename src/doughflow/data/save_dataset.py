"""
This module handles save & load processed data.

"""

from pathlib import Path
import pandas as pd

def save_processed_data(df: pd.DataFrame, filename: str, output_dir="data/processed"):
    # processing data into processed folder
    file_path = Path(output_dir) / filename
    df.to_csv(file_path, index = False)

    return file_path

def load_processed_data(filename:str, file_dir = "data/processed"):
    # load processed data from processed folder

    df = pd.read_csv(Path(file_dir) / filename)

    return df    
