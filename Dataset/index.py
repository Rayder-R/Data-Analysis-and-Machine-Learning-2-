# %%
import zipfile
import os
import pandas as pd

# %%

# Unzip the dataset
def unzip_dataset(zip_path, extract_to='.'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Unzipped {zip_path} to {extract_to}")
    
unzip_dataset('iris.zip', 'Dataset')
# %%
