'''
load data into correct format:
1. just graphs with type features
2. graphs with additional energy labels
'''

import h5py

with h5py.File('/gpfs/work3/0/einf2380/data/pMHCI/features_output_folder/GNN/residue/230201/residue-4162789.hdf5', 'r') as f5:

    print(f5.keys())


datadir = '/gpfs/work3/0/einf2380/data/pMHCI/features_output_folder/GNN/residue/230201/'

data = []

for file_name in datadir:

    with h5py.File(f'{datadir}/{file_name}', 'r') as f5:

        data_subset = f5

    for graph_names, graph in f5.itmes():

        data.append(graph)