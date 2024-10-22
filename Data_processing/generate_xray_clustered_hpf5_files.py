import h5py
import os
import pickle
import re
import numpy as np
import subprocess
from gdomain.scripts.find_gdomain import find_gdomain
# Define paths and patterns
file_pattern = '/gpfs/work3/0/einf2380/data/swiftmhc/xray_cluster{}.hdf5'
pdb_dict_path = '/gpfs/work3/0/einf2380/data/PANDORA_databases/default/PDBs/pMHCI/'
output_directory = '/gpfs/home4/dfruhbuss/ECGDM/Data/Peptide_data/xray_clustered'
keys_lists = []
# Step 1: Generate lists of keys from existing HDF5 files
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
# Output directory for the new HDF5 files
os.makedirs(output_directory, exist_ok=True)
# Step 2: Processing each PDB file from the keys
for i, keys in enumerate(keys_lists):
    file_path = file_pattern.format(i)
    output_hdf5 = os.path.join(output_directory, f'cleaned_pMHCI_cluster{i}.hdf5')
    with h5py.File(file_path, 'r') as input_h5, h5py.File(output_hdf5, 'w') as output_h5:
        for key in keys:
            # Get the PDB path
            pdb_file_name = key + '.pdb'
            pdb_path = os.path.join(pdb_dict_path, pdb_file_name)
            if not os.path.exists(pdb_path):
                print(f"PDB file {pdb_file_name} not found. Skipping...")
                continue
            # Step 3: Apply `find_gdomain` to extract relevant residues from the M chain
            residues_to_keep = find_gdomain(pdb_path)  # Assuming this returns residues of interest for chain M
            # Step 4: Extract the M chain's specific residues and the full P chain
            # Use pdb_selchain for this part
            temp_pdb_m = f'temp_chain_M_{key}.pdb'
            temp_pdb_p = f'temp_chain_P_{key}.pdb'
            temp_combined_pdb = f'temp_combined_{key}.pdb'
            # Extract residues from chain M
            # Use `grep` and `awk` or similar to get the required residues using pdb_selchain
            extract_m_cmd = f'pdb_selchain -A,M {pdb_path} | awk \'$6 >= {residues_to_keep[0].get_id()[1]} && $6 <= {residues_to_keep[-1].get_id()[1]}\' > {temp_pdb_m}'
            subprocess.run(extract_m_cmd, shell=True, check=True)
            # Extract full chain P
            extract_p_cmd = f'pdb_selchain -A P {pdb_path} > {temp_pdb_p}'
            subprocess.run(extract_p_cmd, shell=True, check=True)
            # Combine chain M and chain P into a single PDB file
            combine_cmd = f'cat {temp_pdb_m} {temp_pdb_p} > {temp_combined_pdb}'
            subprocess.run(combine_cmd, shell=True, check=True)
            # Step 5: Save the cleaned PDB file back into the HDF5 file
            with open(temp_combined_pdb, 'rb') as f:
                pdb_data = f.read()
                output_h5.create_dataset(key, data=np.void(pdb_data))
            # Clean up temporary files
            os.remove(temp_pdb_m)
            os.remove(temp_pdb_p)
            os.remove(temp_combined_pdb)
    print(f"New HDF5 file saved: {output_hdf5}")