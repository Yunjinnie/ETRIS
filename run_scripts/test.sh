dataset_name="refcoco" # "refcoco", "refcoco+", "refcocog_g", "refcocog_u"
config_name="bridge_r101.yaml"
gpu=0
split_name="testA" # "val", "testA", "testB" 
# Evaluation on the specified of the specified dataset
CUDA_VISIBLE_DEVICES=$gpu python3 -u test.py \
      --config config/$dataset_name/$config_name \
      --opts TEST.test_split $split_name \
             TEST.test_lmdb datasets/lmdb/$dataset_name/$split_name.lmdb

#python3 -u test.py --config config/refcoco/my.yaml --opts  TRAIN.exp_name lr0001_depth6 encoder.predictor_depth 6 Distributed.dist_url tcp://localhost:1236 TEST.test_split testA TEST.test_lmdb datasets/lmdb/refcoco/testA.lmdb TEST.attention_map True