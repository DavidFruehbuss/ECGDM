import h5py
import os
import subprocess
import numpy as np
import pymol
from pymol import cmd

# Initialize PyMOL in headless mode
pymol.finish_launching(['pymol', '-cq'])

# Corrected directories for input/output
file_pattern = '/gpfs/home4/dfruhbuss/ECGDM/Data/Peptide_data/xray_clustered/cleaned_pMHCI_cluster{}.hdf5'
output_directory = '/gpfs/home4/dfruhbuss/ECGDM/Data/Peptide_data/xray_clustered_aligned'
reference_pdb_path = '/gpfs/home4/dfruhbuss/ECGDM/Data/Peptide_data/reference_structure.pdb'

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Step 1: Define a function to correctly load PDB data
def load_pdb_data_from_hdf5(input_h5, key):
    """ Reads PDB data from an HDF5 dataset and returns it as UTF-8 encoded bytes if possible. """
    try:
        pdb_data = input_h5[key][()]

        # Convert to bytes if it is np.void
        if isinstance(pdb_data, np.void):
            pdb_bytes = pdb_data.tobytes()
        elif isinstance(pdb_data, bytes):
            pdb_bytes = pdb_data
        else:
            print(f"Skipping key {key}: Unexpected data type {type(pdb_data)}")
            return None

        # Attempt to decode the bytes to a UTF-8 string to ensure valid data
        try:
            pdb_text = pdb_bytes.decode('utf-8')
        except UnicodeDecodeError:
            print(f"Skipping key {key}: Unable to decode data as UTF-8")
            return None

        # Return the re-encoded UTF-8 bytes
        return pdb_text.encode('utf-8')
    except Exception as e:
        print(f"Error reading PDB data for key {key}: {e}")
        return None

# Step 2: Processing each HDF5 file
keys_lists = []
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

# Align each PDB based on the reference structure
for i, keys in enumerate(keys_lists):
    file_path = file_pattern.format(i)
    output_hdf5 = os.path.join(output_directory, f'aligned_pMHCI_cluster{i}.hdf5')

    with h5py.File(file_path, 'r') as input_h5, h5py.File(output_hdf5, 'w') as output_h5:
        for key in keys:
            pdb_data = load_pdb_data_from_hdf5(input_h5, key)
            if pdb_data is None:
                continue
            
            # Save the PDB data to a temporary file
            temp_input_pdb = f'temp_{key}.pdb'
            with open(temp_input_pdb, 'wb') as f:
                f.write(pdb_data)

            # Use PyMOL to align the structures
            try:
                cmd.load(reference_pdb_path, "reference")
                cmd.load(temp_input_pdb, "mobile")
                
                # Align both chains P and M
                cmd.align("mobile and (chain P or chain M)", "reference and (chain P or chain M)")
                
                # Save the aligned structure
                aligned_pdb = f'aligned_{key}.pdb'
                cmd.save(aligned_pdb, "mobile")
                
                # Read back the aligned data
                with open(aligned_pdb, 'rb') as aligned_f:
                    aligned_data = aligned_f.read()
                    output_h5.create_dataset(key, data=np.void(aligned_data))
                
                # Cleanup
                cmd.delete("all")
                os.remove(temp_input_pdb)
                os.remove(aligned_pdb)

            except Exception as e:
                print(f"Failed to align structure {key}: {e}")
                continue

        print(f"New aligned HDF5 file saved: {output_hdf5}")
