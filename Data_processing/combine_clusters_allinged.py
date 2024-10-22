import os
import h5py
import shutil

# Function to read and extract raw PDB entries from the XRAY HDF5 file
def read_hdf5_xray(filepath):
    """ Reads XRAY HDF5 PDB data as is, without conversion. """
    atom_lines = {}
    
    with h5py.File(filepath, 'r') as f:
        for dataset_name in f.keys():
            pdb_content = f[dataset_name][()]  # Directly get the raw bytes
            atom_lines[dataset_name] = pdb_content  # Keep it as is
    return atom_lines

# Function to read PDB entries from 8K dataset
def read_hdf5_8k(filepath):
    """ Reads and parses HDF5 files from 8K dataset, converting to UTF-8 string format. """
    with h5py.File(filepath, 'r') as f:
        data = {}
        for pdb_name in f.keys():
            pdb_file_content = f[pdb_name][()]
            if isinstance(pdb_file_content, bytes):
                pdb_text = pdb_file_content.decode('utf-8')  # Decode to UTF-8 for consistency
            else:
                pdb_text = str(pdb_file_content)
            data[pdb_name] = pdb_text.encode('utf-8')  # Ensure consistent UTF-8 bytes
    return data

# Function to save the combined datasets
def save_combined_hdf5(filepath, combined_data):
    """ Saves combined data to a new HDF5 file, ensuring all data is UTF-8 encoded. """
    with h5py.File(filepath, 'w') as f:
        for pdb_name, pdb_content in combined_data.items():
            if isinstance(pdb_content, bytes):
                # Ensure that the content is in UTF-8 encoding before saving
                pdb_content = pdb_content.decode('utf-8').encode('utf-8')
            f.create_dataset(pdb_name, data=pdb_content)  # Save as raw bytes

# Combines XRAY and 8K datasets without losing original format
def combine_datasets(xray_data, eight_k_data):
    """ Combines two datasets (from XRAY and 8K) into one. """
    combined_data = xray_data.copy()  # Start with XRAY data
    combined_data.update(eight_k_data)  # Merge with 8K data
    return combined_data

# Directories for input/output
xray_directory = '/gpfs/home4/dfruhbuss/ECGDM/Data/Peptide_data/xray_clustered_aligned'
eight_k_directory = '/gpfs/home4/dfruhbuss/ECGDM/Data/Peptide_data/8k_clustered'
output_directory = '/gpfs/home4/dfruhbuss/ECGDM/Data/Peptide_data/combined'
temp_directory = './temp_pdb_files'  # Temporary directory for PDB files

# Create output and temporary directories if they don't exist
os.makedirs(output_directory, exist_ok=True)
os.makedirs(temp_directory, exist_ok=True)

# Process files 0 to 9
for i in range(10):
    # Create a subfolder for each cluster
    cluster_temp_directory = os.path.join(temp_directory, f'cluster_{i}')
    os.makedirs(cluster_temp_directory, exist_ok=True)

    # Paths to the corresponding files in each dataset
    xray_file = os.path.join(xray_directory, f'aligned_pMHCI_cluster{i}.hdf5')
    eight_k_file = os.path.join(eight_k_directory, f'BA_pMHCI_cluster{i}.hdf5')

    # Check if both files exist
    if os.path.exists(xray_file) and os.path.exists(eight_k_file):
        # Read XRAY and 8K data
        xray_data = read_hdf5_xray(xray_file)  # Get raw XRAY PDBs without conversion
        eight_k_data = read_hdf5_8k(eight_k_file)  # Get 8K PDBs

        # Save temporary PDB files for XRAY data
        for pdb_name, pdb_content in xray_data.items():
            temp_pdb_file = os.path.join(cluster_temp_directory, f'{pdb_name}.pdb')
            with open(temp_pdb_file, 'wb') as temp_file:
                temp_file.write(pdb_content)  # Write the raw bytes directly

        # Combine datasets
        combined_data = combine_datasets(xray_data, eight_k_data)
        
        # Output path for the combined file
        output_file = os.path.join(output_directory, f'combined_cluster{i}.hdf5')
        
        # Save the combined HDF5 file
        save_combined_hdf5(output_file, combined_data)
        
        print(f'Combined files for cluster {i} and saved to {output_file}')
    else:
        print(f'Skipping cluster {i} due to missing files.')

# Cleanup: Remove temporary PDB files after saving
shutil.rmtree(temp_directory)
print("Temporary PDB files cleaned up.")
