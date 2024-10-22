import h5py
import os
import numpy as np
import random

# Function to read the content of a PDB file and extract atom coordinates
def read_pdb_file(pdb_file_path):
    atom_coords = []
    atom_lines = []
    
    try:
        with open(pdb_file_path, 'r') as file:
            for line in file:
                if line.startswith("ATOM"):
                    # Extract coordinates
                    coords = list(map(float, line[30:54].split()))
                    atom_coords.append(coords)
                    atom_lines.append(line.strip())
    except Exception as e:
        print(f"Error reading {pdb_file_path}: {e}")
    
    return atom_coords, atom_lines

# Function to apply rotation to the coordinates
def rotate_coordinates(coords, angle_x, angle_y, angle_z):
    angle_x, angle_y, angle_z = map(np.radians, [angle_x, angle_y, angle_z])
    
    rotation_x = np.array([[1, 0, 0],
                            [0, np.cos(angle_x), -np.sin(angle_x)],
                            [0, np.sin(angle_x), np.cos(angle_x)]])
    
    rotation_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                            [0, 1, 0],
                            [-np.sin(angle_y), 0, np.cos(angle_y)]])
    
    rotation_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                            [np.sin(angle_z), np.cos(angle_z), 0],
                            [0, 0, 1]])
    
    rotation_matrix = rotation_z @ rotation_y @ rotation_x
    rotated_coords = np.dot(coords, rotation_matrix.T)
    
    return rotated_coords

# Function to save rotated coordinates to HDF5
def save_rotated_pdb_to_hdf5(output_path, atom_lines, all_rotated_coords, original_names):
    with h5py.File(output_path, 'w') as hdf5_file:
        for idx, coords in enumerate(all_rotated_coords):
            pdb_entry = []
            for line, coord in zip(atom_lines[idx], coords):
                new_line = line[:30] + f"{coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}" + line[54:]
                pdb_entry.append(new_line)
            dataset_name = original_names[idx]  # Use the original PDB name
            hdf5_file.create_dataset(dataset_name, data="\n".join(pdb_entry).encode('utf-8'))

# Directory containing the original HDF5 files
input_directory = '/gpfs/home4/dfruhbuss/ECGDM/Data/Peptide_data/8k_clustered'
# Output directory for the rotated HDF5 files
output_directory = '/gpfs/home4/dfruhbuss/ECGDM/Data/Peptide_data/8k_clustered_rotated'
os.makedirs(output_directory, exist_ok=True)

# Iterate over the 10 HDF5 files and process them
for i in range(10):
    input_hdf5_path = os.path.join(input_directory, f'BA_pMHCI_cluster{i}.hdf5')
    
    if os.path.exists(input_hdf5_path):
        print(f"Processing file: {input_hdf5_path}")
        
        with h5py.File(input_hdf5_path, 'r') as hdf5_file:
            atom_lines = []
            all_rotated_coords = []
            original_names = []  # To store original PDB names
            
            for pdb_name in hdf5_file.keys():
                pdb_file_content = hdf5_file[pdb_name][()]
                temp_pdb_path = f'temp_{pdb_name}.pdb'
                
                with open(temp_pdb_path, 'w') as temp_file:
                    temp_file.write(pdb_file_content.decode('utf-8'))

                atom_coords, lines = read_pdb_file(temp_pdb_path)
                atom_lines.append(lines)  # Collect lines for each PDB entry
                original_names.append(pdb_name)  # Collect original names
                
                if atom_coords:
                    # Generate random rotation angles
                    angle_x = random.uniform(0, 360)
                    angle_y = random.uniform(0, 360)
                    angle_z = random.uniform(0, 360)
                    
                    rotated_coords = rotate_coordinates(np.array(atom_coords), angle_x, angle_y, angle_z)
                    all_rotated_coords.append(rotated_coords)
                
                os.remove(temp_pdb_path)
        
        output_hdf5_path = os.path.join(output_directory, f'BA_pMHCI_cluster{i}_rotated.hdf5')  # Keep original naming
        save_rotated_pdb_to_hdf5(output_hdf5_path, atom_lines, all_rotated_coords, original_names)
        
        print(f"Saved rotated structure for cluster {i} to {output_hdf5_path}")
    else:
        print(f"File {input_hdf5_path} not found. Skipping...")
