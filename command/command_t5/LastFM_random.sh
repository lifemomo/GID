
# 激活虚拟环境
. /root/autodl-tmp/venv_openp5/bin/activate

# 启动分布式训练（2卡，GPU 4,5）
torchrun --nproc_per_node=2 --master_port=1995 ../../src/src_t5/main.py \
  --datasets LastFM \
  --distributed 1 \
  --gpu 0,1 \
  --tasks sequential,straightforward \
  --item_indexing random \
  --epochs 20 \
  --batch_size 64 \
  --master_port 1995 \
  --prompt_file ../../prompt.txt \
  --sample_prompt 1 \
  --eval_batch_size 20 \
  --dist_sampler 0 \
  --max_his 20 \
  --sample_num 3,3 \
  --train 1 \
  --test_prompt seen:0 \
  --lr 1e-3 \
  --test_before_train 0 \
  --test_epoch 0 \
  --data_path ../../data