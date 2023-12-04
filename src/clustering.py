import logging
import torch
import numpy as np
import faiss
import os
import copy
from tqdm import tqdm 
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.distributed as dist
import _pickle as pickle
import copy
from distributed import *
from torch_scatter import scatter


class Clustering():

    def __init__(self, args):
        self.episode_size = args.episode_size
        self.feature_dim = args.projection_dim
        self.maccs_dim = args.maccs_dim
        self.usrcat_dim = args.usrcat_dim
        self.k1 = args.k1
        self.k2 = args.k2
        self.k3 = args.k3
        self.kmeans_max_iter = args.kmeans_max_iter
        self.kmeans_nredo = args.kmeans_nredo
        self.max_points_per_centroid = args.max_points_per_centroid
        self.spherical = args.spherical
        self.use_kmeans_pp = args.use_kmeans_pp
        self.PBT = args.PBT
        self.reset()
        
    def reset(self):
        self.mol_2d_features = torch.zeros(size=(self.episode_size, self.feature_dim), dtype=torch.float32).fill_(float('nan')).share_memory_()
        self.mol_3d_features = torch.zeros(size=(self.episode_size, self.feature_dim), dtype=torch.float32).fill_(float('nan')).share_memory_()
        self.maccs_features = torch.zeros(size=(self.episode_size, self.maccs_dim), dtype=torch.float32).fill_(float('nan')).share_memory_()
        self.usrcat_features = torch.zeros(size=(self.episode_size, self.usrcat_dim), dtype=torch.float32).fill_(float('nan')).share_memory_()
        
        self.mol_2d_labels_1 = torch.zeros(self.episode_size, dtype=torch.long).fill_(int(-1))
        self.mol_3d_labels_1 = torch.zeros(self.episode_size, dtype=torch.long).fill_(int(-1))
        self.mol_2d_labels_2 = torch.zeros(self.episode_size, dtype=torch.long).fill_(int(-1))
        self.mol_3d_labels_2 = torch.zeros(self.episode_size, dtype=torch.long).fill_(int(-1))
        self.mol_2d_labels_3 = torch.zeros(self.episode_size, dtype=torch.long).fill_(int(-1))
        self.mol_3d_labels_3 = torch.zeros(self.episode_size, dtype=torch.long).fill_(int(-1))
        
        self.mol_2d_centroids_1 = torch.zeros([self.k1, self.feature_dim]).fill_(float('nan'))
        self.mol_3d_centroids_1 = torch.zeros([self.k1, self.feature_dim]).fill_(float('nan'))
        self.mol_2d_centroids_2 = torch.zeros([self.k2, self.feature_dim]).fill_(float('nan'))
        self.mol_3d_centroids_2 = torch.zeros([self.k2, self.feature_dim]).fill_(float('nan'))
        self.mol_2d_centroids_3 = torch.zeros([self.k3, self.feature_dim]).fill_(float('nan'))
        self.mol_3d_centroids_3 = torch.zeros([self.k3, self.feature_dim]).fill_(float('nan'))
        

    def load_batch(self, index, mol_2d_features, mol_3d_features):
        self.mol_2d_features[index] = copy.deepcopy(mol_2d_features.detach().cpu().type(torch.float32))
        self.mol_3d_features[index] = copy.deepcopy(mol_3d_features.detach().cpu().type(torch.float32))


    def load_batch_fp(self, index, maccs, usrcat, mol_2d_features, mol_3d_features):
        self.maccs_features[index] = copy.deepcopy(maccs.cpu().type(torch.float32))
        self.usrcat_features[index] = copy.deepcopy(usrcat.cpu().type(torch.float32))
        self.mol_2d_features[index] = copy.deepcopy(mol_2d_features.detach().cpu().type(torch.float32))
        self.mol_3d_features[index] = copy.deepcopy(mol_3d_features.detach().cpu().type(torch.float32))

    
    def dump(self, file, item):
        f = open(file, 'wb')
        pickle.dump(item, f, protocol=4)
        f.close() 


    def load(self, file):
        f = open(file, 'rb')
        item = pickle.load(f)
        f.close()
        return item
    
    
    def sync_prototypes(self, args): 
        if is_master(args):
            self.dump(os.path.join(args.cache_path, f'mol_2d_labels_1.pkl'), self.mol_2d_labels_1)
            self.dump(os.path.join(args.cache_path, f'mol_2d_centroids_1.pkl'), self.mol_2d_centroids_1)
            self.dump(os.path.join(args.cache_path, f'mol_3d_labels_1.pkl'), self.mol_3d_labels_1)
            self.dump(os.path.join(args.cache_path, f'mol_3d_centroids_1.pkl'), self.mol_3d_centroids_1)

            self.dump(os.path.join(args.cache_path, f'mol_2d_labels_2.pkl'), self.mol_2d_labels_2)
            self.dump(os.path.join(args.cache_path, f'mol_2d_centroids_2.pkl'), self.mol_2d_centroids_2)
            self.dump(os.path.join(args.cache_path, f'mol_3d_labels_2.pkl'), self.mol_3d_labels_2)
            self.dump(os.path.join(args.cache_path, f'mol_3d_centroids_2.pkl'), self.mol_3d_centroids_2)

            self.dump(os.path.join(args.cache_path, f'mol_2d_labels_3.pkl'), self.mol_2d_labels_3)
            self.dump(os.path.join(args.cache_path, f'mol_2d_centroids_3.pkl'), self.mol_2d_centroids_3)
            self.dump(os.path.join(args.cache_path, f'mol_3d_labels_3.pkl'), self.mol_3d_labels_3)
            self.dump(os.path.join(args.cache_path, f'mol_3d_centroids_3.pkl'), self.mol_3d_centroids_3)
            
        if args.distributed:
            dist.barrier()

        if not is_master(args):
            self.mol_2d_labels_1 = self.load(os.path.join(args.cache_path, f'mol_2d_labels_1.pkl'))
            self.mol_2d_centroids_1 = self.load(os.path.join(args.cache_path, f'mol_2d_centroids_1.pkl'))
            self.mol_3d_labels_1 = self.load(os.path.join(args.cache_path, f'mol_3d_labels_1.pkl'))
            self.mol_3d_centroids_1 = self.load(os.path.join(args.cache_path, f'mol_3d_centroids_1.pkl'))

            self.mol_2d_labels_2 = self.load(os.path.join(args.cache_path, f'mol_2d_labels_2.pkl'))
            self.mol_2d_centroids_2 = self.load(os.path.join(args.cache_path, f'mol_2d_centroids_2.pkl'))
            self.mol_3d_labels_2 = self.load(os.path.join(args.cache_path, f'mol_3d_labels_2.pkl'))
            self.mol_3d_centroids_2 = self.load(os.path.join(args.cache_path, f'mol_3d_centroids_2.pkl'))
            
            self.mol_2d_labels_3 = self.load(os.path.join(args.cache_path, f'mol_2d_labels_3.pkl'))
            self.mol_2d_centroids_3 = self.load(os.path.join(args.cache_path, f'mol_2d_centroids_3.pkl'))
            self.mol_3d_labels_3 = self.load(os.path.join(args.cache_path, f'mol_3d_labels_3.pkl'))
            self.mol_3d_centroids_3 = self.load(os.path.join(args.cache_path, f'mol_3d_centroids_3.pkl'))
        
        if args.distributed:
            dist.barrier()

        if is_master(args):
            print(f'Constructed prototypes are synchronized')
            for file in os.listdir(args.cache_path):
                # os.remove(os.path.join(args.cache_path, file))
                a = 1
            print(f'Cache path {args.cache_path} has been cleared')

                
    def generate_labels(self):
        # check NaN
        assert (torch.isnan(self.mol_2d_features)).sum().item() == 0, '2d feature error'
        assert (torch.isnan(self.mol_3d_features)).sum().item() == 0, '3d feature error'
        
        print(f'Constructing 2d prototypes with K-Means')
        self.mol_2d_labels_1[:], self.mol_2d_centroids_1[:], _, _ = self.kmeans(self.mol_2d_features, self.k1)
        self.mol_2d_labels_2[:], self.mol_2d_centroids_2[:], _, _ = self.kmeans(self.mol_2d_features, self.k2)
        self.mol_2d_labels_3[:], self.mol_2d_centroids_3[:], _, _ = self.kmeans(self.mol_2d_features, self.k3)
        print(f'Constructing 3d prototypes with K-Means')
        self.mol_3d_labels_1[:], self.mol_3d_centroids_1[:], _, _ = self.kmeans(self.mol_3d_features, self.k1)
        self.mol_3d_labels_2[:], self.mol_3d_centroids_2[:], _, _ = self.kmeans(self.mol_3d_features, self.k2)
        self.mol_3d_labels_3[:], self.mol_3d_centroids_3[:], _, _ = self.kmeans(self.mol_3d_features, self.k3)
        
        if self.PBT:
            self.mol_2d_centroids_1[:] = scatter(self.mol_2d_features, self.mol_3d_labels_1, dim=0, reduce='mean')
            self.mol_2d_centroids_2[:] = scatter(self.mol_2d_features, self.mol_3d_labels_2, dim=0, reduce='mean')
            self.mol_2d_centroids_3[:] = scatter(self.mol_2d_features, self.mol_3d_labels_3, dim=0, reduce='mean')

            self.mol_3d_centroids_1[:] = scatter(self.mol_3d_features, self.mol_2d_labels_1, dim=0, reduce='mean')
            self.mol_3d_centroids_2[:] = scatter(self.mol_3d_features, self.mol_2d_labels_2, dim=0, reduce='mean')
            self.mol_3d_centroids_3[:] = scatter(self.mol_3d_features, self.mol_2d_labels_3, dim=0, reduce='mean')

        print('construction complete')


    def generate_labels_from_fp(self):    
        temp = self.spherical
        self.spherical = False
        # check NaN 
        # print(self.maccs_features)
        # print((torch.isnan(self.maccs_features)).sum())
        assert (torch.isnan(self.maccs_features)).sum().item() == 0, 'maccs error'
        assert (torch.isnan(self.usrcat_features)).sum().item() == 0, 'usrcat error'
        assert (torch.isnan(self.mol_2d_features)).sum().item() == 0, '2d feature error'
        assert (torch.isnan(self.mol_3d_features)).sum().item() == 0, '3d feature error'

        print(f'Constructing 2d fp prototypes with K-Means')
        self.mol_2d_labels_1[:], _, _, _ = self.kmeans(self.maccs_features, self.k1)
        self.mol_2d_labels_2[:], _, _, _ = self.kmeans(self.maccs_features, self.k2)
        self.mol_2d_labels_3[:], _, _, _ = self.kmeans(self.maccs_features, self.k3)

        print(f'Constructing 3d fp prototypes with K-Means')
        self.mol_3d_labels_1[:], _, _, _ = self.kmeans(self.usrcat_features, self.k1)
        self.mol_3d_labels_2[:], _, _, _ = self.kmeans(self.usrcat_features, self.k2)
        self.mol_3d_labels_3[:], _, _, _ = self.kmeans(self.usrcat_features, self.k3)

        if self.PBT:
            self.mol_2d_centroids_1[:] = scatter(self.mol_2d_features, self.mol_3d_labels_1, dim=0, reduce='mean')
            self.mol_2d_centroids_2[:] = scatter(self.mol_2d_features, self.mol_3d_labels_2, dim=0, reduce='mean')
            self.mol_2d_centroids_3[:] = scatter(self.mol_2d_features, self.mol_3d_labels_3, dim=0, reduce='mean')

            self.mol_3d_centroids_1[:] = scatter(self.mol_3d_features, self.mol_2d_labels_1, dim=0, reduce='mean')
            self.mol_3d_centroids_2[:] = scatter(self.mol_3d_features, self.mol_2d_labels_2, dim=0, reduce='mean')
            self.mol_3d_centroids_3[:] = scatter(self.mol_3d_features, self.mol_2d_labels_3, dim=0, reduce='mean')
        else:
            self.mol_2d_centroids_1[:] = scatter(self.mol_2d_features, self.mol_2d_labels_1, dim=0, reduce='mean')
            self.mol_2d_centroids_2[:] = scatter(self.mol_2d_features, self.mol_2d_labels_2, dim=0, reduce='mean')
            self.mol_2d_centroids_3[:] = scatter(self.mol_2d_features, self.mol_2d_labels_3, dim=0, reduce='mean')

            self.mol_3d_centroids_1[:] = scatter(self.mol_3d_features, self.mol_3d_labels_1, dim=0, reduce='mean')
            self.mol_3d_centroids_2[:] = scatter(self.mol_3d_features, self.mol_3d_labels_2, dim=0, reduce='mean')
            self.mol_3d_centroids_3[:] = scatter(self.mol_3d_features, self.mol_3d_labels_3, dim=0, reduce='mean')

        print('construction complete')

        self.spherical = temp

    def kmeans(self, feature, k):
        feature = copy.deepcopy(feature)
        if self.use_kmeans_pp:
            init_centroids = self.kmeans_plus_plus_fast(feature, k).numpy()
        else:
            init_centroids = feature[np.random.choice(self.episode_size, size=k, replace=False)].numpy()

        feature = feature.cpu().numpy()

        centroids = torch.zeros([k, feature.shape[1]])
        
        kmeans = faiss.Kmeans(
            d=feature.shape[1], 
            k=k, 
            niter=self.kmeans_max_iter, 
            nredo=self.kmeans_nredo,
            verbose=False, 
            gpu=0)
        kmeans.cp.max_points_per_centroid = self.max_points_per_centroid
        kmeans.cp.spherical = self.spherical

        kmeans.train(feature, init_centroids=init_centroids)

        centroids = torch.from_numpy(kmeans.centroids)
        distance, labels = kmeans.index.search(feature, 1)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0])

        del kmeans
        del feature
        gc.collect()
        torch.cuda.empty_cache()

        return torch.from_numpy(labels), centroids, 0, 1


    def kmeans_plus_plus_fast(self, data, num_clusters, max_kmeanspp=100, device='cuda'):
        data = data.to(device)
        num_points = len(data)

        # Randomly select the first centroid from the data
        centroids = [data[torch.randint(0, num_points, (1,)).item()].unsqueeze(0)]

        # Select remaining centroids
        if num_clusters <= max_kmeanspp:
            for _ in range(1, num_clusters):
                distances = torch.stack([torch.sum((data - c) ** 2, dim=1) for c in centroids], dim=1)
                min_distances = torch.min(distances, dim=1)[0]
                probabilities = min_distances / torch.sum(min_distances)
                centroid_index = torch.multinomial(probabilities, 1).item()
                centroids.append(data[centroid_index].unsqueeze(0))

            centroids = torch.cat(centroids, dim=0).cpu()
        else:
            for _ in range(1, max_kmeanspp):
                distances = torch.stack([torch.sum((data - c) ** 2, dim=1) for c in centroids], dim=1)
                min_distances = torch.min(distances, dim=1)[0]
                probabilities = min_distances / torch.sum(min_distances)
                centroid_index = torch.multinomial(probabilities, 1).item()
                centroids.append(data[centroid_index].unsqueeze(0))

            centroids = torch.cat(centroids, dim=0).cpu()
            # Randomly select remaining centroids
            rand_points = torch.randint(0, num_points, (num_clusters-max_kmeanspp,))
            centroids = torch.cat([centroids, data[rand_points].cpu()], dim=0)
        
        return centroids

    