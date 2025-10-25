# 激活虚拟环境
. /root/autodl-tmp/venv_openp5/bin/activate
# 启动分布式训练（3卡）
torchrun --nproc_per_node=3 --master_port=12345 ../../src/src_t5/main.py \
--datasets Movies --distributed 1  --tasks sequential,straightforward \
--item_indexing random --epochs 20 --batch_size 128  --prompt_file ../prompt.txt --sample_prompt 1 \
--eval_batch_size 20 --dist_sampler 0 --max_his 20  --sample_num 3,3 --train 1 \
--test_prompt seen:0 --lr 1e-3 --test_before_train 0 --test_epoch 0 \
--data_path ../../data



