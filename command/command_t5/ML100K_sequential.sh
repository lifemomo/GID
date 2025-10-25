#conda activate t5
#python ../../src/src_t5/main.py --datasets ML100K --distributed 1 --gpu 6,7 --tasks sequential,straightforward --item_indexing sequential --epochs 20 --batch_size 64 --master_port 1994 --prompt_file ../prompt.txt --sample_prompt 1 --eval_batch_size 20 --dist_sampler 0 --max_his 20  --sample_num 3,3 --train 1 --test_prompt seen:0 --lr 1e-3 --test_before_train 0 --test_epoch 0

#!/bin/bash
#--max_his：影响输入序列长度，越大推荐模型越“有上下文”，但内存占用也越高。
#--batch_size：越大训练快但显存消耗越高，需与 --max_his 配合调试。
#--sample_prompt / --sample_num：控制了 prompt 多样性，会影响模型泛化能力。
#--epochs 20	训练轮数
# 激活虚拟环境
. /root/autodl-tmp/venv_openp5/bin/activate
#注意\不能有任何其他内容
#python ../../src/src_t5/main.py \
torchrun --nproc_per_node=2 --master_port=1994 ../../src/src_t5/main.py \
  --datasets ML100K \
  --distributed 1 \
  --gpu 0,1 \
  --tasks sequential,straightforward \
  --item_indexing sequential \
  --epochs 20 \
  --batch_size 64 \
  --master_port 1994 \
  --prompt_file ./prompt.txt \
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

