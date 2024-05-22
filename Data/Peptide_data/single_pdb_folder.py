import os
import shutil

# Define the source and destination directories
source_dir = '/projects/0/einf2380/data/pMHCI/db2_selected_models/BA/'
destination_dir = '/projects/0/einf2380/data/pMHCI/db2_selected_models/single_folder/'

# Create the destination directory if it does not exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Function to copy pdb files
def copy_pdb_files(source, destination):
    for root, _, files in os.walk(source):
        depth = len(root.split(os.sep)) - len(source.split(os.sep))
        if depth > 5:
            continue
        for file in files:
            if file.endswith('.pdb'):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(destination, file)
                if not os.path.exists(dest_file):  # Avoid overwriting
                    shutil.copy(src_file, dest_file)
                else:
                    print(f"File {dest_file} already exists. Skipping.")

# Copy pdb files from source to destination
copy_pdb_files(source_dir, destination_dir)

print("Finished copying .pdb files.")
