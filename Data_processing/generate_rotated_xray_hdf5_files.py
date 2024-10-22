import h5py
import os
import numpy as np
import random

# Function to read the PDB datasets from an HDF5 file
def read_hdf5_file(hdf5_file_path):
    atom_lines = {}
    
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        for dataset_name in hdf5_file.keys():
            pdb_data = hdf5_file[dataset_name][()]
            
            if isinstance(pdb_data, np.void):
                pdb_bytes = pdb_data.tobytes()
            elif isinstance(pdb_data, bytes):
                pdb_bytes = pdb_data
            else:
                continue

            try:
                pdb_text = pdb_bytes.decode('utf-8')
            except UnicodeDecodeError:
                continue
            atom_lines[dataset_name] = pdb_text
    
    return atom_lines

# Function to parse PDB text and extract atom coordinates
def parse_pdb(pdb_text):
    atom_coords = []
    atom_lines = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                if len(line) < 54:
                    continue
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atom_coords.append([x, y, z])
                atom_lines.append(line)
            except ValueError:
                continue
    if not atom_coords:
        return None, None
    return np.array(atom_coords), atom_lines

# Function to apply random rotation to coordinates
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
    return np.dot(coords, rotation_matrix.T)

# Function to replace coordinates in PDB lines
def replace_coordinates(atom_lines, rotated_coords):
    new_pdb_lines = []
    for line, coord in zip(atom_lines, rotated_coords):
        new_x = f"{coord[0]:8.3f}"
        new_y = f"{coord[1]:8.3f}"
        new_z = f"{coord[2]:8.3f}"
        new_line = line[:30] + new_x + new_y + new_z + line[54:]
        new_pdb_lines.append(new_line)
    return new_pdb_lines

# Function to save rotated PDB data back to HDF5
def save_rotated_pdb_to_hdf5(output_path, rotated_pdbs):
    with h5py.File(output_path, 'w') as hdf5_file:
        for dataset_name, pdb_text in rotated_pdbs.items():
            pdb_bytes = pdb_text.encode('utf-8')
            hdf5_file.create_dataset(dataset_name, data=np.void(pdb_bytes))

# Main processing function
def process_rotated_dataset():
    input_directory = '/gpfs/home4/dfruhbuss/ECGDM/Data/Peptide_data/xray_clustered'
    output_directory = '/gpfs/home4/dfruhbuss/ECGDM/Data/Peptide_data/xray_rotated'
    os.makedirs(output_directory, exist_ok=True)

    for i in range(10):
        input_hdf5_path = os.path.join(input_directory, f'cleaned_pMHCI_cluster{i}.hdf5')
        if not os.path.exists(input_hdf5_path):
            continue

        atom_lines_dict = read_hdf5_file(input_hdf5_path)
        rotated_pdbs = {}

        for dataset_name, pdb_text in atom_lines_dict.items():
            coords, lines = parse_pdb(pdb_text)
            if coords is None or lines is None:
                continue

            centroid = np.mean(coords, axis=0)
            centered_coords = coords - centroid

            angle_x, angle_y, angle_z = random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360)
            rotated_coords = rotate_coordinates(centered_coords, angle_x, angle_y, angle_z)

            new_pdb_lines = replace_coordinates(lines, rotated_coords)
            rotated_pdbs[dataset_name] = '\n'.join(new_pdb_lines)

        if rotated_pdbs:
            output_hdf5_path = os.path.join(output_directory, f'cleaned_pMHCI_cluster{i}_rotated.hdf5')
            save_rotated_pdb_to_hdf5(output_hdf5_path, rotated_pdbs)

# Run the processing function
if __name__ == "__main__":
    process_rotated_dataset()
