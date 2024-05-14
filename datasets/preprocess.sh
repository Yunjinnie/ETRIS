#!/bin/bash

#SBATCH --job-name=data          # Submit a job named "example"
#SBATCH --nodes=1                    # Using 1 node, cpu
#SBATCH --gres=gpu:0                 # Using 1 gpu
#SBATCH --time=0-12:00:00            # 1 hour time limit
#SBATCH --mem=10000MB                # Using 10GB CPU Memory
#SBATCH --cpus-per-task=4            # Using 4 maximum processor

eval "$(conda shell.bash hook)"
conda activate etris

python ../tools/folder2lmdb.py -j anns/refcoco/train.json -i images/train2014/ -m masks/refcoco -o lmdb/refcoco
python ../tools/folder2lmdb.py -j anns/refcoco/val.json -i images/train2014/ -m masks/refcoco -o lmdb/refcoco
python ../tools/folder2lmdb.py -j anns/refcoco/testA.json -i images/train2014/ -m masks/refcoco -o lmdb/refcoco
python ../tools/folder2lmdb.py -j anns/refcoco/testB.json -i images/train2014/ -m masks/refcoco -o lmdb/refcoco



python ../tools/data_process.py --data_root . --output_dir . --dataset refcoco+ --split unc --generate_mask

# lmdb
python ../tools/folder2lmdb.py -j anns/refcoco+/train.json -i images/train2014/ -m masks/refcoco+ -o lmdb/refcoco+
python ../tools/folder2lmdb.py -j anns/refcoco+/val.json -i images/train2014/ -m masks/refcoco+ -o lmdb/refcoco+
python ../tools/folder2lmdb.py -j anns/refcoco+/testA.json -i images/train2014/ -m masks/refcoco+ -o lmdb/refcoco+
python ../tools/folder2lmdb.py -j anns/refcoco+/testB.json -i images/train2014/ -m masks/refcoco+ -o lmdb/refcoco+


# convert
python ../tools/data_process.py --data_root . --output_dir . --dataset refcocog --split umd --generate_mask  # umd split
mv anns/refcocog anns/refcocog_u
mv masks/refcocog masks/refcocog_u

python ../tools/data_process.py --data_root . --output_dir . --dataset refcocog --split google --generate_mask  # google split
mv anns/refcocog anns/refcocog_g
mv masks/refcocog masks/refcocog_g

# lmdb
python ../tools/folder2lmdb.py -j anns/refcocog_u/train.json -i images/train2014/ -m masks/refcocog_u -o lmdb/refcocog_u
python ../tools/folder2lmdb.py -j anns/refcocog_u/val.json -i images/train2014/ -m masks/refcocog_u -o lmdb/refcocog_u
python ../tools/folder2lmdb.py -j anns/refcocog_u/test.json -i images/train2014/ -m masks/refcocog_u -o lmdb/refcocog_u

python ../tools/folder2lmdb.py -j anns/refcocog_g/train.json -i images/train2014/ -m masks/refcocog_g -o lmdb/refcocog_g
python ../tools/folder2lmdb.py -j anns/refcocog_g/val.json -i images/train2014/ -m masks/refcocog_g -o lmdb/refcocog_g

