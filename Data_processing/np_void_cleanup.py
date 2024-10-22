import os
import h5py
import numpy as np

# Directories for input/output
combined_directory = '/gpfs/home4/dfruhbuss/ECGDM/Data/Peptide_data/combined'

def overwrite_with_utf8_hdf5(filepath):
    """ Reads an HDF5 file and rewrites it with all PDB entries saved as UTF-8. """
    with h5py.File(filepath, 'r') as f:
        combined_data = {}
        
        for pdb_name in f.keys():
            pdb_content = f[pdb_name][()]
            
            # Ensure the content is a UTF-8 string
            if isinstance(pdb_content, np.void):
                pdb_content = pdb_content.tobytes().decode('utf-8')  # Convert numpy.void to bytes and decode
            elif isinstance(pdb_content, bytes):
                pdb_content = pdb_content.decode('utf-8')  # Decode bytes to UTF-8 string
            
            # Re-encode the content as UTF-8 bytes
            combined_data[pdb_name] = pdb_content.encode('utf-8')
    
    # Overwrite the original HDF5 file
    with h5py.File(filepath, 'w') as f:
        for pdb_name, pdb_content in combined_data.items():
            f.create_dataset(pdb_name, data=pdb_content)  # Save as UTF-8 bytes

# Process all combined files (combined_cluster0_rotated.hdf5 to combined_cluster9_rotated.hdf5)
for i in range(10):
    combined_file = os.path.join(combined_directory, f'combined_cluster{i}.hdf5')

    # Check if the combined file exists
    if os.path.exists(combined_file):
        overwrite_with_utf8_hdf5(combined_file)
        print(f'Overwritten {combined_file} with UTF-8 encoded PDBs.')
    else:
        print(f'Skipping cluster {i}, file not found: {combined_file}')
