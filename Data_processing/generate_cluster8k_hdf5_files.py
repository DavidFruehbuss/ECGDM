import h5py
import os
import pickle
import re
import numpy as np  # Import NumPy

# Step 1: Generate lists of keys from existing HDF5 files
file_pattern = '/gpfs/work3/0/einf2380/data/tcrspec/BA_pMHCI_cluster{}.hdf5'
keys_lists = []

# Loop through files numbered 0 to 9
for i in range(10):
    file_path = file_pattern.format(i)
    
    if os.path.exists(file_path):
        print(f"Processing file: {file_path}")
        
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            keys_lists.append(keys)
        
        print(f"File {i} keys: {keys}")
    else:
        print(f"File {file_path} not found. Skipping...")

# Step 2: Load the pdb_dict from the pickle file
pdb_dict_path = './Data/Peptide_data/pdb_index.pkl'
with open(pdb_dict_path, 'rb') as file:
    pdb_dict = pickle.load(file)

# Output directory for the new HDF5 files
output_directory = '/gpfs/home4/dfruhbuss/ECGDM/Data/Peptide_data/8k_clustered'
os.makedirs(output_directory, exist_ok=True)

# Function to read the content of a PDB file
def read_pdb_file(pdb_file_path):
    try:
        with open(pdb_file_path, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {pdb_file_path}: {e}")
        return None

# Function to extract the numeric part from the PDB name
def extract_numeric_part(pdb_name):
    match = re.search(r'\d+', pdb_name)
    return match.group() if match else None

# Iterate through the 10 lists and create HDF5 files
for i, pdb_list in enumerate(keys_lists):
    output_hdf5_path = os.path.join(output_directory, f'BA_pMHCI_cluster{i}.hdf5')
    
    with h5py.File(output_hdf5_path, 'w') as hdf5_file:
        for pdb_name in pdb_list:
            numeric_part = extract_numeric_part(pdb_name)  # Extract numeric part
            
            if numeric_part:
                pdb_file_path = pdb_dict.get(numeric_part)  # Look up using numeric part
                
                if pdb_file_path:
                    pdb_content = read_pdb_file(pdb_file_path)
                    
                    if pdb_content:
                        # Save the entire content as a single dataset
                        hdf5_file.create_dataset(pdb_name, data=np.string_(pdb_content))
                        print(f"Stored {pdb_name} in {output_hdf5_path}")
                    else:
                        print(f"Skipping {pdb_name} due to read error.")
                else:
                    print(f"{numeric_part} not found in pdb_dict. Skipping...")
            else:
                print(f"No numeric part found for {pdb_name}. Skipping...")
    
    print(f"HDF5 file {output_hdf5_path} created successfully.")
