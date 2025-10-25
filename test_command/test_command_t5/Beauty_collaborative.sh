# 激活虚拟环境
. /root/autodl-tmp/venv_openp5/bin/activate

# 启动分布式训练（4卡）
torchrun --nproc_per_node=2 --master_port=12345 ../../src/src_t5/main.py \
--datasets Beauty --distributed 1 --gpu 0,1 \
--tasks sequential,straightforward --item_indexing collaborative --epochs 20 --batch_size 64 \
--master_port 12345 --prompt_file ../../prompt.txt --sample_prompt 1 --eval_batch_size 1 --dist_sampler 0 \
--max_his 20  --sample_num 3,3 --test_prompt seen:0 --lr 1e-3 \
--collaborative_token_size 500 \
--collaborative_cluster 2 \
--train 0 \
--model_name Beauty_collaborative.pt --test_filtered 1 --test_filtered_batch 0 \
--data_path /root/autodl-tmp/OpenP5-main/data

torchrun --nproc_per_node=2 --master_port=12345 ../../src/src_t5/main.py \
--datasets Beauty --distributed 1 --gpu 0,1 \
--tasks sequential,straightforward --item_indexing collaborative --epochs 20 --batch_size 64 \
--master_port 12345 --prompt_file ../../prompt.txt --sample_prompt 1 --eval_batch_size 1 --dist_sampler 0 \
--max_his 20  --sample_num 3,3 --test_prompt unseen:0 --lr 1e-3 \
--collaborative_token_size 500 \
--collaborative_cluster 2 \
--train 0 \
--model_name Beauty_collaborative.pt --test_filtered 1 --test_filtered_batch 0 \
--data_path /root/autodl-tmp/OpenP5-main/data
