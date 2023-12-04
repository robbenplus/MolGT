import argparse

parser = argparse.ArgumentParser()

# about basic info
  # to reproduce results, finetune with seed 1, 2, 3
parser.add_argument('--num_workers', type=int, default=30)  
parser.add_argument("--local_rank", default=-1, type=int)

# for custom dataset: lmdb_path
parser.add_argument('--lmdb_path', type=str, default='/YourPath/data/example_data.lmdb')
parser.add_argument('--train_max_atoms', type=int, default=128) 
parser.add_argument('--valid_max_atoms', type=int, default=40)

# about pretrain
parser.add_argument('--train_batch_size', type=int, default=88)  
parser.add_argument('--valid_batch_size', type=int, default=400)  
parser.add_argument('--valid_interval', type=int, default=2) 
parser.add_argument('--valid_ratio', type=float, default=0.1) 
parser.add_argument('--lr', type=float, default=5e-5)  
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--eps', type=float, default=1e-6)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.99)
parser.add_argument('--max_norm', type=float, default=1.0)  
parser.add_argument('--train_epoch', type=int, default=15)
parser.add_argument('--save_dir', type=str, default='/YourPath/3d-graphormer/src/ckpt')
parser.add_argument('--ft_save_dir', type=str, default='/YourPath/3d-graphormer/src/ckpt')
parser.add_argument('--log_step', type=int, default=50)  
parser.add_argument('--ft_log_step', type=int, default=5)
parser.add_argument('--use_amp', type=bool, default=True)
parser.add_argument('--stop_amp_epoch', type=int, default=999) 
parser.add_argument('--warmup_proportion', type=float, default=0.01)
parser.add_argument('--max_ckpts', type=int, default=10)
parser.add_argument('--ft_max_ckpts', type=int, default=1)
parser.add_argument('--load_ckpt', type=str, default='/YourPath/3d-graphormer/src/ckpt/pytorch_model.bin')
#parser.add_argument('--load_ckpt', type=str, default='')
parser.add_argument('--max_degree', type=int, default=25)
parser.add_argument('--multi_hop_max_dist', type=int, default=3)
parser.add_argument('--max_spd', type=int, default=128)

# for clustering
parser.add_argument('--k1', type=int, default=500) 
parser.add_argument('--k2', type=int, default=1000)
parser.add_argument('--k3', type=int, default=3000)
parser.add_argument('--maccs_dim', type=int, default=167)
parser.add_argument('--usrcat_dim', type=int, default=60)
parser.add_argument('--fp_epoch', type=int, default=1)
parser.add_argument('--episode_size', type=int, default=1000000) 
parser.add_argument('--PBT', type=bool, default=True) 
parser.add_argument('--distributed', type=bool, default=True) 
parser.add_argument('--cache_path', type=str, default='./cache/')  
parser.add_argument('--kmeans_max_iter', type=int, default=10)
parser.add_argument('--kmeans_nredo', type=int, default=1)  
parser.add_argument('--max_points_per_centroid', type=int, default=8192) 
parser.add_argument('--spherical', type=bool, default=False) 
parser.add_argument('--use_kmeans_pp', type=bool, default=False) 
parser.add_argument('--dataset_size', type=int, default=1000000)  
parser.add_argument('--target_T', type=float, default=1)
parser.add_argument('--proto_reduction', type=str, default='sum')  

# about task --- 2D&3D CL
parser.add_argument('--CL_similarity_metric', type=str, default='infonce') # directclr infonce
parser.add_argument('--CL_projection', type=str, default='linear') # linear nonlinear
parser.add_argument('--CL_T', type=float, default=0.1)  
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--CL_reduction', type=str, default='sum')
parser.add_argument('--projection_dim', type=int, default=64)

# about task --- InfoMotif
parser.add_argument('--pos_n', type=int, default=2) 
parser.add_argument('--tau', type=float, default=0.1) 
parser.add_argument('--neg_n', type=int, default=250)
parser.add_argument('--node_proj_dim', type=int, default=64)
parser.add_argument('--infomotif_reduction', type=str, default='sum')

# about task --- Denoising
parser.add_argument('--denoising_reduction', type=str, default='sum')
parser.add_argument('--denoising_loss', type=str, default='regression')
parser.add_argument('--noise_scale', type=float, default=1)  
parser.add_argument('--global_denoise', type=bool, default=False)

# about Multi task
parser.add_argument('--denoising_weight', type=float, default=2) 
parser.add_argument('--CL_weight', type=float, default=1) 
parser.add_argument('--infomotif_2d_weight', type=float, default=0.05) 
parser.add_argument('--proto_weight', type=float, default=0.01) 
parser.add_argument('--norm_weight', type=float, default=0.0001)
parser.add_argument('--do_denoising', type=bool, default=True)

# for downstream tasks
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--molnet_path', type=str, default='/YourPath/3d-graphormer/data/moleculenet')
parser.add_argument('--classification_dataset', type=str, default='bbbp')
parser.add_argument('--regression_dataset', type=str, default='esol') # esol freesolv lipo malaria cep

# for BBBP  
parser.add_argument('--bbbp_batch_size', type=int, default=32) 
parser.add_argument('--bbbp_lr', type=float, default= 3e-5)  
parser.add_argument('--bbbp_train_epoch', type=int, default=15)  
parser.add_argument('--bbbp_warmup', type=float, default=0.01)  
parser.add_argument('--bbbp_dropout', type=float, default=0.3)  
parser.add_argument('--bbbp_update_freq', type=int, default=1)  
parser.add_argument('--bbbp_weight_decay', type=float, default=0) 
parser.add_argument('--bbbp_classes', type=int, default=1) 

# for Bace
parser.add_argument('--bace_batch_size', type=int, default=32)
parser.add_argument('--bace_lr', type=float, default=3e-5)  
parser.add_argument('--bace_train_epoch', type=int, default=15) 
parser.add_argument('--bace_warmup', type=float, default=0.01)
parser.add_argument('--bace_dropout', type=float, default=0.5)
parser.add_argument('--bace_update_freq', type=int, default=1) 
parser.add_argument('--bace_weight_decay', type=float, default=1e-2) 
parser.add_argument('--bace_classes', type=int, default=1)

# for Tox21
parser.add_argument('--tox21_batch_size', type=int, default=16) 
parser.add_argument('--tox21_lr', type=float, default=1e-4)   
parser.add_argument('--tox21_train_epoch', type=int, default=15) 
parser.add_argument('--tox21_warmup', type=float, default=0.01) 
parser.add_argument('--tox21_dropout', type=float, default=0.1) 
parser.add_argument('--tox21_update_freq', type=int, default=4) 
parser.add_argument('--tox21_weight_decay', type=float, default=1e-4) 
parser.add_argument('--tox21_classes', type=int, default=12)

# for ToxCast
parser.add_argument('--toxcast_batch_size', type=int, default=32) 
parser.add_argument('--toxcast_lr', type=float, default=1e-4)  
parser.add_argument('--toxcast_train_epoch', type=int, default=30) 
parser.add_argument('--toxcast_warmup', type=float, default=0.1) 
parser.add_argument('--toxcast_dropout', type=float, default=0.0) 
parser.add_argument('--toxcast_update_freq', type=int, default=1) 
parser.add_argument('--toxcast_weight_decay', type=float, default=0) 
parser.add_argument('--toxcast_classes', type=int, default=617)

# for Sider
parser.add_argument('--sider_batch_size', type=int, default=4) 
parser.add_argument('--sider_lr', type=float, default=5e-5) 
parser.add_argument('--sider_train_epoch', type=int, default=10) 
parser.add_argument('--sider_warmup', type=float, default=0.1) 
parser.add_argument('--sider_dropout', type=float, default=0.0) 
parser.add_argument('--sider_update_freq', type=int, default=1)  
parser.add_argument('--sider_weight_decay', type=float, default=1e-1) 
parser.add_argument('--sider_classes', type=int, default=27)

# for ClinTox
parser.add_argument('--clintox_batch_size', type=int, default=32)
parser.add_argument('--clintox_lr', type=float, default=5e-5)  
parser.add_argument('--clintox_train_epoch', type=int, default=15)
parser.add_argument('--clintox_warmup', type=float, default=0.1)
parser.add_argument('--clintox_dropout', type=float, default=0.5)
parser.add_argument('--clintox_update_freq', type=int, default=1) 
parser.add_argument('--clintox_weight_decay', type=float, default=1e-2)
parser.add_argument('--clintox_classes', type=int, default=2)

# for ESOL
parser.add_argument('--esol_batch_size', type=int, default=32)
parser.add_argument('--esol_lr', type=float, default=2e-5)  
parser.add_argument('--esol_train_epoch', type=int, default=20)
parser.add_argument('--esol_warmup', type=float, default=0.01)
parser.add_argument('--esol_model_dropout', type=float, default=0.0)
parser.add_argument('--esol_dropout', type=float, default=0.0)
parser.add_argument('--esol_update_freq', type=int, default=1) 
parser.add_argument('--esol_weight_decay', type=float, default=1e-2)
parser.add_argument('--esol_norm', type=float, default=-1)
parser.add_argument('--esol_loss', type=str, default='mse') # mae mse smooth_l1
parser.add_argument('--esol_classes', type=int, default=1)

# for FreeSolv
parser.add_argument('--freesolv_batch_size', type=int, default=32)
parser.add_argument('--freesolv_lr', type=float, default=2e-5)  
parser.add_argument('--freesolv_train_epoch', type=int, default=20)
parser.add_argument('--freesolv_warmup', type=float, default=0.01)
parser.add_argument('--freesolv_model_dropout', type=float, default=0.0)
parser.add_argument('--freesolv_dropout', type=float, default=0.1)
parser.add_argument('--freesolv_update_freq', type=int, default=1) 
parser.add_argument('--freesolv_weight_decay', type=float, default=1e-2)
parser.add_argument('--freesolv_norm', type=float, default=-1)
parser.add_argument('--freesolv_loss', type=str, default='mse') # mae mse smooth_l1
parser.add_argument('--freesolv_classes', type=int, default=1)

# for Lipo
parser.add_argument('--lipo_batch_size', type=int, default=32)
parser.add_argument('--lipo_lr', type=float, default=3e-5)  
parser.add_argument('--lipo_train_epoch', type=int, default=30)
parser.add_argument('--lipo_warmup', type=float, default=0.01)
parser.add_argument('--lipo_model_dropout', type=float, default=0.1)
parser.add_argument('--lipo_dropout', type=float, default=0.0)
parser.add_argument('--lipo_update_freq', type=int, default=1) 
parser.add_argument('--lipo_weight_decay', type=float, default=1e-2)
parser.add_argument('--lipo_norm', type=float, default=5.0)
parser.add_argument('--lipo_loss', type=str, default='smooth_l1') # mae mse smooth_l1
parser.add_argument('--lipo_classes', type=int, default=1)

# for Malaria
parser.add_argument('--malaria_batch_size', type=int, default=32) 
parser.add_argument('--malaria_lr', type=float, default=3e-5)   
parser.add_argument('--malaria_train_epoch', type=int, default=15)
parser.add_argument('--malaria_warmup', type=float, default=0.01) 
parser.add_argument('--malaria_model_dropout', type=float, default=0.0) 
parser.add_argument('--malaria_dropout', type=float, default=0.0) 
parser.add_argument('--malaria_update_freq', type=int, default=1) 
parser.add_argument('--malaria_weight_decay', type=float, default=1e-2) 
parser.add_argument('--malaria_norm', type=float, default=5.0) 
parser.add_argument('--malaria_loss', type=str, default='smooth_l1') # smooth_l1
parser.add_argument('--malaria_classes', type=int, default=1)

# for CEP
parser.add_argument('--cep_batch_size', type=int, default=64) 
parser.add_argument('--cep_lr', type=float, default=5e-5)  
parser.add_argument('--cep_train_epoch', type=int, default=80) 
parser.add_argument('--cep_warmup', type=float, default=0.01) 
parser.add_argument('--cep_model_dropout', type=float, default=0.0) 
parser.add_argument('--cep_dropout', type=float, default=0.0) 
parser.add_argument('--cep_update_freq', type=int, default=1)  
parser.add_argument('--cep_weight_decay', type=float, default=1e-2) 
parser.add_argument('--cep_norm', type=float, default=5.0) 
parser.add_argument('--cep_loss', type=str, default='mse') # mae mse smooth_l1
parser.add_argument('--cep_classes', type=int, default=1)

# args = parser.parse_args(args=[])
args = parser.parse_args()
