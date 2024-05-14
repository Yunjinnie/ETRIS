#!/bin/bash

#SBATCH --job-name=seg                 # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:1                          # Using 1 gpu
#SBATCH --time=0-12:00:00                     # 12 hour timelimit
#SBATCH --mem=200000MB                         # Using 10GB CPU Memory
#SBATCH --partition=P2                        # Using "b" partition 
#SBATCH --cpus-per-task=16                     # Using 4 maximum processor
#SBATCH --output=out/lr0001_depth4_proj.out

source /home/s2/yunjinna/.bashrc
source /home/s2/yunjinna/anaconda/bin/activate
conda activate env2

# "refcoco", "refcoco+", "refcocog_g", "refcocog_u"
python -u train.py --config config/refcoco/my.yaml --opts  TRAIN.exp_name lr0001_depth4_proj encoder.predictor_depth 4 TRAIN.base_lr 0.0001 Distributed.dist_url tcp://localhost:1240

#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -u train.py --config config/refcoco/my.yaml --opts  TRAIN.exp_name lr0001_depth4_exp2_multi encoder.predictor_depth 4 TRAIN.base_lr 0.0001 TRAIN.epochs 40 Distributed.dist_url tcp://localhost:1238
#python3 -u test.py --config config/refcoco/my.yaml --opts  TRAIN.exp_name lr0001_depth6 encoder.predictor_depth 6 Distributed.dist_url tcp://localhost:1236 TEST.test_split testA TEST.test_lmdb datasets/lmdb/refcoco/testA.lmdb
