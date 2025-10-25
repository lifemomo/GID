# 激活虚拟环境
. /root/autodl-tmp/venv_openp5/bin/activate

# 执行训练任务（协同索引）
#python ../../src/src_t5/main.py \
torchrun --nproc_per_node=2 --master_port=1994 ../../src/src_t5/main.py \
  --datasets ML100K \
  --distributed 1 \
  --gpu 0,1 \
  --tasks sequential,straightforward \
  --item_indexing collaborative \
  --epochs 20 \
  --batch_size 128 \
  --master_port 1994 \
  --prompt_file ../prompt.txt \
  --sample_prompt 1 \
  --eval_batch_size 20 \
  --dist_sampler 0 \
  --max_his 20 \
  --sample_num 3,3 \
  --train 1 \
  --test_prompt seen:0 \
  --lr 1e-3 \
  --test_before_train 0 \
  --test_epoch 1 \
  --collaborative_token_size 500 \
  --collaborative_cluster 2 \
  --data_path ../../data
