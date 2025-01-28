import os
import h5py
import numpy as np

source_dir = '/gpfs/home4/dfruhbus/ECGDM/Data/Peptide_data/pmhc_xray_8K_aligned/'  # Added trailing slash
output_file = os.path.join(source_dir, 'test.hdf5')  # Output file named train.hdf5

# source_filenames = [f'combined_cluster{i}.hdf5' for i in range(8)]
source_filenames = [f'combined_cluster{9}.hdf5']

def merge_hdf5_files(source_dir, source_filenames, output_file):
    all_entries = []
    all_pdb_names = []  # New list to store pdb_names (entry names)

    for filename in source_filenames:
        filepath = os.path.join(source_dir, filename)
        print(f"Processing file: {filepath}")

        with h5py.File(filepath, 'r') as hdf_in:
            for entry_name in hdf_in.keys():
                print(f"Loading entry: {entry_name}")

                # Access the entry directly
                entry = hdf_in[entry_name]

                # Directly get the pdb_string, ensuring we handle it correctly
                try:
                    pdb_string = entry[()]  # This retrieves the value directly
                    if isinstance(pdb_string, bytes):  # Check if it's a byte string
                        pdb_string = pdb_string.decode('utf-8')  # Decode to string
                    all_entries.append(pdb_string)
                    all_pdb_names.append(entry_name)  # Store the pdb_name (entry name)
                except KeyError:
                    print(f"Warning: 'pdb_string' not found in entry {entry_name}")
                except Exception as e:
                    print(f"Error retrieving pdb_string from {entry_name}: {e}")

    # Save the collected entries to a new HDF5 file
    # Create an object array and fill it with entries
    all_entries_array = np.array(all_entries, dtype=object)  # Use object array for variable-length strings
    all_pdb_names_array = np.array(all_pdb_names, dtype=h5py.string_dtype(encoding='utf-8'))  # Save entry names
    
    with h5py.File(output_file, 'w') as hdf_out:
        hdf_out.create_dataset('pdb_strings', data=all_entries_array, dtype=h5py.string_dtype(encoding='utf-8'))
        hdf_out.create_dataset('pdb_names', data=all_pdb_names_array, dtype=h5py.string_dtype(encoding='utf-8'))  # Save pdb names

    print(f"All files merged successfully into {output_file}")


if __name__ == "__main__":
    merge_hdf5_files(source_dir, source_filenames, output_file)
