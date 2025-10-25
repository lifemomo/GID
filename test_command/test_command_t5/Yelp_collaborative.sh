# 激活虚拟环境
. /root/autodl-tmp/venv_openp5/bin/activate

# 启动分布式训练（4卡）
torchrun --nproc_per_node=3 --master_port=12345 ../../src/src_t5/main.py \
--datasets Yelp \
--distributed 1 --tasks sequential,straightforward --item_indexing collaborative --epochs 10 --batch_size 128 \
--prompt_file ../../prompt.txt --sample_prompt 1 --eval_batch_size 20 --dist_sampler 0 --max_his 20  --sample_num 2,2 \
--train 0 --test_prompt seen:0 --lr 1e-3 \
--collaborative_token_size 200 \
--collaborative_cluster 20 \
--test_before_train 0 --test_epoch 0 \
--collaborative_float32 1 --test_filtered 0 \
--model_path ./Yelp_collaborative.pt \
--data_path /root/autodl-tmp/OpenP5-main/data

torchrun --nproc_per_node=3 --master_port=12345 ../../src/src_t5/main.py \
--datasets Yelp \
--distributed 1 --tasks sequential,straightforward --item_indexing collaborative --epochs 10 --batch_size 128 \
--prompt_file ../../prompt.txt --sample_prompt 1 --eval_batch_size 20 --dist_sampler 0 --max_his 20  --sample_num 2,2 \
--train 0 --test_prompt unseen:0 --lr 1e-3 \
--collaborative_token_size 200 \
--collaborative_cluster 20 \
--test_before_train 0 --test_epoch 0 \
--collaborative_float32 1 --test_filtered 0 \
--model_path ./Yelp_collaborative.pt \
--data_path /root/autodl-tmp/OpenP5-main/data

