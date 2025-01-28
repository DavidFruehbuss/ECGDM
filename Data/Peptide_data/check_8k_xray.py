import os
import h5py
from torch.utils.data import DataLoader
from dataset_8k_xray import PDB_Dataset

def test_dataset_loading():
    datadir = '/gpfs/home4/dfruhbus/ECGDM/Data/Peptide_data/pmhc_xray_8K_aligned/'  # Adjusted data directory
    dataset = PDB_Dataset(datadir=datadir, split='val')  # Load the dataset
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"Loaded {len(dataset)} entries from train split.")
    
    for i, batch in enumerate(loader):
        print(f"Batch {i}:")
        print(batch)
        # Here you can add more checks, such as shapes or specific values if needed.

if __name__ == "__main__":
    test_dataset_loading()
