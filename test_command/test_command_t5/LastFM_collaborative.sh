# 1. 激活虚拟环境
. /root/autodl-tmp/venv_openp5/bin/activate

# 对 seen:0 测试的命令
python ../../src/src_t5/main.py \
  --datasets LastFM \
  --distributed 0 \
  --gpu 0 \
  --tasks sequential,straightforward \
  --item_indexing collaborative \
  --epochs 20 \
  --batch_size 128 \
  --prompt_file ../../prompt.txt \
  --sample_prompt 1 \
  --eval_batch_size 20 \
  --dist_sampler 0 \
  --max_his 20 \
  --sample_num 3,3 \
  --train 0 \
  --test_prompt seen:0 \
  --lr 1e-3 \
  --test_before_train 0 \
  --test_epoch 0 \
  --collaborative_token_size 500 \
  --collaborative_cluster 30 \
  --test_filtered 0 \
  --model_name LastFM_collaborative.pt \
  --data_path ../../data

# 对 unseen:0 测试的命令
python ../../src/src_t5/main.py \
  --datasets LastFM \
  --distributed 0 \
  --gpu 0 \
  --tasks sequential,straightforward \
  --item_indexing collaborative \
  --epochs 20 \
  --batch_size 128 \
  --prompt_file ../../prompt.txt \
  --sample_prompt 1 \
  --eval_batch_size 20 \
  --dist_sampler 0 \
  --max_his 20 \
  --sample_num 3,3 \
  --train 0 \
  --test_prompt unseen:0 \
  --lr 1e-3 \
  --test_before_train 0 \
  --test_epoch 0 \
  --collaborative_token_size 500 \
  --collaborative_cluster 30 \
  --test_filtered 0 \
  --model_name LastFM_collaborative.pt \
  --data_path ../../data
