# 1. 激活虚拟环境
. /root/autodl-tmp/venv_openp5/bin/activate

# 对 seen:0 测试的命令
torchrun \
  --nproc_per_node=2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29500 \
  --master_port=1994 ../../src/src_t5/main.py \
  --datasets ML100K \
  --distributed 1 \
  --gpu 0,1 \
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
  --collaborative_cluster 2 \
  --test_filtered 0 \
  --model_name ML100K_collaborative.pt \
  --data_path ../../data

# 对 unseen:0 测试的命令
torchrun \
  --nproc_per_node=2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29500 \
  --master_port=1994 ../../src/src_t5/main.py \
  --datasets ML100K \
  --distributed 1 \
  --gpu 0,1 \
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
  --collaborative_cluster 2 \
  --test_filtered 0 \
  --model_name ML100K_collaborative.pt \
  --data_path ../../data
