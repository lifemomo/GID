# 激活虚拟环境
. /root/autodl-tmp/venv_openp5/bin/activate

# 执行训练任务（协同索引）
torchrun --nproc_per_node=2 --master_port=12345 ../../src/src_t5/main.py \
  --datasets LastFM \
  --distributed 1 \
  --tasks sequential,straightforward \
  --item_indexing collaborative \
  --epochs 12 \
  --batch_size 64 \
  --prompt_file ../prompt.txt \
  --sample_prompt 0 \
  --eval_batch_size 1 \
  --dist_sampler 0 \
  --max_his 20 \
  --sample_num 3,3 \
  --train 1 \
  --test_prompt seen:0 \
  --lr 1e-3 \
  --test_before_train 0 \
  --test_epoch 1 \
  --collaborative_token_size 500 \
  --collaborative_cluster 30 \
  --data_path ../../data
