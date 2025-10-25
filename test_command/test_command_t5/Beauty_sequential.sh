# 激活虚拟环境
. /root/autodl-tmp/venv_openp5/bin/activate

# 启动分布式训练（4卡）
torchrun --nproc_per_node=2 --master_port=2000 ../../src/src_t5/main.py \
--datasets Beauty --distributed 1 --gpu 0,1 --tasks sequential,straightforward --item_indexing sequential --epochs 20 \
--batch_size 64 --master_port 2000 --prompt_file ../../prompt.txt --sample_prompt 1 --eval_batch_size 1 \
--dist_sampler 0 --max_his 20  --sample_num 3,3 --test_prompt seen:0 --lr 1e-3 --train 0 --model_name Beauty_sequential.pt \
--data_path /root/autodl-tmp/OpenP5-main/data
# 启动分布式训练（4卡）
torchrun --nproc_per_node=2 --master_port=2000 ../../src/src_t5/main.py \
--datasets Beauty --distributed 1 --gpu 0,1 --tasks sequential,straightforward --item_indexing sequential --epochs 20 \
--batch_size 64 --master_port 2000 --prompt_file ../../prompt.txt --sample_prompt 1 --eval_batch_size 1 \
--dist_sampler 0 --max_his 20  --sample_num 3,3 --test_prompt unseen:0 --lr 1e-3 --train 0 --model_name Beauty_sequential.pt \
--data_path /root/autodl-tmp/OpenP5-main/data


