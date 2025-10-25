# 1. 激活虚拟环境
. /root/autodl-tmp/venv_openp5/bin/activate

# 2. 执行测试任务（seen 测试）
python ../../src/src_t5/main.py \
  --datasets ML100K \
  --distributed 0 \
  --gpu 0 \
  --tasks sequential,straightforward \
  --item_indexing random \
  --epochs 20 \
  --batch_size 64 \
  --master_port 1994 \
  --prompt_file prompt.txt \
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
  --test_filtered 0 \
  --model_path ./ML100K_sequential.pt \
  --data_path /root/autodl-tmp/OpenP5-main/data

# 3. 执行测试任务（unseen 测试）
python ../../src/src_t5/main.py \
  --datasets ML100K \
  --distributed 0 \
  --gpu 0 \
  --tasks sequential,straightforward \
  --item_indexing random \
  --epochs 20 \
  --batch_size 64 \
  --master_port 1994 \
  --prompt_file prompt.txt \
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
  --test_filtered 0 \
  --model_path ./ML100K_sequential.pt \
  --data_path /root/autodl-tmp/OpenP5-main/data
