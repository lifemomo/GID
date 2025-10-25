import torch
import os
import argparse
import logging
from transformers import AutoTokenizer
from torch.utils.data import ConcatDataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from data.MultiTaskDataset import MultiTaskDataset
from runner.SingleRunner import SingleRunner
from runner.DistributedRunner import DistributedRunner
from processor.Collator import Collator, TestCollator
from processor.SingleMultiDataTaskSampler import SingleMultiDataTaskSampler
from processor.DistMultiDataTaskSampler import DistMultiDataTaskSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import T5Config
from model.P5_T5 import P5_T5
from utils import utils
from utils import initialization
from torch.nn.parallel import DistributedDataParallel as DDP
import pdb


def get_dataset(args):


    # init dataset
    datasets = args.datasets.split(',')
    train_all_datasets = []
    valid_all_datasets = []
    test_all_datasets = []
    for data in datasets:
        TrainDataset = MultiTaskDataset(args, data, 'train')
        train_all_datasets.append(TrainDataset)
        if args.valid_select > 0:
            ValidDataset = MultiTaskDataset(args, data, 'validation')
            valid_all_datasets.append(ValidDataset)

    TrainSet = ConcatDataset(train_all_datasets)
    if args.valid_select > 0:
        ValidSet = ConcatDataset(valid_all_datasets)
    else:
        ValidSet = None

    return TrainSet, ValidSet

def get_loader(args, tokenizer, TrainSet, ValidSet, rank=0):

    # generate training validation loader.
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node == 1:
        args.distributed = 0
    if args.dist_sampler == 0:
        train_sampler = DistMultiDataTaskSampler(TrainSet, args.batch_size, args.world_size, rank, args.seed, shuffle=True) if args.distributed else SingleMultiDataTaskSampler(TrainSet, args.batch_size, args.seed, shuffle=True)
    else:
        train_sampler = DistributedSampler(TrainSet) if args.distributed else None
    if args.valid_select > 0:
        valid_sampler = DistributedSampler(ValidSet) if args.distributed else None

    collator = Collator(tokenizer)
    train_loader = DataLoader(dataset=TrainSet, sampler=train_sampler, batch_size=args.batch_size, collate_fn=collator, shuffle=False)
    if args.valid_select > 0:
        valid_loader = DataLoader(dataset=ValidSet, sampler=valid_sampler, batch_size=args.batch_size, collate_fn=collator, shuffle=False)
    else:
        valid_loader = None

    return train_loader, valid_loader

def single_main():
    # running on single gpu.

    # init args
    parser = argparse.ArgumentParser(description='OpenP5')
    parser = utils.parse_global_args(parser)
    parser = MultiTaskDataset.parse_dataset_args(parser)
    parser = SingleMultiDataTaskSampler.parse_sampler_args(parser)
    parser = SingleRunner.parse_runner_args(parser)

    args, extras = parser.parse_known_args()

    utils.setup_logging(args)
    utils.setup_model_path(args)
    utils.set_seed(args.seed)

    args.rank = 0

    device = torch.device("cuda", int(args.gpu.split(',')[0]))


    logging.info(vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    # init dataset and dataloader
    TrainSet, ValidSet = get_dataset(args)

    train_loader, valid_loader = get_loader(args, tokenizer, TrainSet, ValidSet)

    if 't5' in args.backbone:
        config = T5Config.from_pretrained(args.backbone)
        logging.info(f"Use {args.backbone} backbone model")
    else:
        raise NotImplementedError

    model = P5_T5.from_pretrained(args.backbone, config=config)
    model.to(device)

    if args.item_indexing == 'collaborative':
        for ds in train_loader.dataset.datasets:
            tokenizer.add_tokens(ds.new_token)
    model.resize_token_embeddings(len(tokenizer))

    if args.random_initialize == 1:
        logging.info("Random initialize number related tokens")
        initialization.random_initialization(model, tokenizer, args.backbone)

    if args.load:
        logging.info(f"Load model from {args.model_path}")
        model = utils.load_model(model, args.model_path, args)

    runner = SingleRunner(model, tokenizer, train_loader, valid_loader, device, args)

    if args.train:
        logging.info("Start training")
        runner.train()

    logging.info(f"Load model from {args.model_path}")
    runner.test(args.model_path)



def distributed_launch():
    parser = argparse.ArgumentParser(description='OpenP5')
    parser = utils.parse_global_args(parser)
    parser = MultiTaskDataset.parse_dataset_args(parser)
    parser = DistMultiDataTaskSampler.parse_sampler_args(parser)
    parser = DistributedRunner.parse_runner_args(parser)
    args, extras = parser.parse_known_args()

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node



    mp.spawn(
        distributed_main, args=(args, ), nprocs=ngpus_per_node, join=True
    )

"此代码适合用于单卡训练"
# def distributed_main(local_rank, args):
#
#     # distributed learning
#     args.rank = local_rank
#     utils.set_seed(args.seed)
#     os.environ['MASTER_ADDR'] = args.master_addr
#     os.environ['MASTER_PORT'] = args.master_port
#     torch.cuda.set_device(local_rank)
#     dist.init_process_group(
#         backend="nccl", world_size=args.world_size, rank=local_rank
#     )
#     utils.setup_logging(args)
#     utils.setup_model_path(args)
#
#     if args.rank == 0:
#         logging.info(vars(args))
#     TrainSet, ValidSet = get_dataset(args)
#
#     device = f"cuda:{local_rank}"
#     args.gpu = local_rank
#
#     tokenizer = AutoTokenizer.from_pretrained(args.backbone)
#
#     train_loader, valid_loader = get_loader(args, tokenizer, TrainSet, ValidSet, local_rank)
#
#     if 't5' in args.backbone:
#         config = T5Config.from_pretrained(args.backbone)
#         if local_rank == 0:
#             logging.info(f"Use {args.backbone} backbone model")
#     else:
#         raise NotImplementError
#
#
#
#     model = P5_T5.from_pretrained(args.backbone, config=config)
#     model.to(device)
#
#     if args.item_indexing == 'collaborative':
#         for ds in train_loader.dataset.datasets:
#             tokenizer.add_tokens(ds.new_token)
#     model.resize_token_embeddings(len(tokenizer))
#
#     if args.random_initialize == 1:
#         if local_rank == 0:
#             logging.info("Random initialize number related tokens")
#         initialization.random_initialization(model, tokenizer, args.backbone)
#
#     if args.load:
#         if local_rank == 0:
#             logging.info(f"Load model from {args.model_path}")
#         model = utils.load_model(model, args.model_path, args, loc=device)
#         model.to(device)
#
#     runner = DistributedRunner(model, tokenizer, train_loader, valid_loader, device, args, local_rank)
#
#     if args.train:
#         if local_rank == 0:
#             logging.info("Start training")
#         runner.train()
#     dist.barrier()
#
#     if local_rank == 0:
#         logging.info(f"Load model from {args.model_path}")
#     runner.test(args.model_path)
#
#     return
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='OpenP5')
#     parser = utils.parse_global_args(parser)
#     init_args, extras = parser.parse_known_args()
#     os.environ["CUDA_VISIBLE_DEVICES"] = init_args.gpu
#     ngpus_per_node = torch.cuda.device_count()
#     if init_args.distributed and ngpus_per_node > 1:
#         distributed_launch()
#     else:
#         single_main()


"此代码适合用于分布式训练"
def distributed_main(local_rank, args):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    # 设置基本环境变量
    args.rank = local_rank
    utils.set_seed(args.seed)
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", world_size=args.world_size, rank=local_rank)

    utils.setup_logging(args)
    utils.setup_model_path(args)

    if args.rank == 0:
        logging.info(vars(args))

    # 数据加载
    TrainSet, ValidSet = get_dataset(args)
    device = torch.device(f"cuda:{local_rank}")
    args.gpu = local_rank
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    train_loader, valid_loader = get_loader(args, tokenizer, TrainSet, ValidSet, local_rank)

    # 模型加载
    if 't5' in args.backbone:
        config = T5Config.from_pretrained(args.backbone)
        if local_rank == 0:
            logging.info(f"Use {args.backbone} backbone model")
    else:
        raise NotImplementedError

    model = P5_T5.from_pretrained(args.backbone, config=config)
    model.to(device)

    # 添加 collaborative tokens 并 resize embedding（必须在封装 DDP 前完成）
    if args.item_indexing == 'collaborative':
        for ds in train_loader.dataset.datasets:
            tokenizer.add_tokens(ds.new_token)
    model.resize_token_embeddings(len(tokenizer))

    # 封装为 DDP 模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 随机初始化嵌入
    if args.random_initialize == 1:
        if local_rank == 0:
            logging.info("Random initialize number related tokens")
        initialization.random_initialization(model.module, tokenizer, args.backbone)

    # 加载已有模型
    if args.load:
        if local_rank == 0:
            logging.info(f"Load model from {args.model_path}")
        model.module = utils.load_model(model.module, args.model_path, args, loc=device)

    # 初始化训练器并训练
    runner = DistributedRunner(model, tokenizer, train_loader, valid_loader, device, args, local_rank)

    if args.train:
        if local_rank == 0:
            logging.info("Start training")
        runner.train()

    # 同步等待所有进程
    dist.barrier()

    if local_rank == 0:
        logging.info(f"Load model from {args.model_path}")
    runner.test(args.model_path)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OpenP5')
    parser = utils.parse_global_args(parser)
    parser = MultiTaskDataset.parse_dataset_args(parser)
    parser = DistMultiDataTaskSampler.parse_sampler_args(parser)
    parser = DistributedRunner.parse_runner_args(parser)
    args, extras = parser.parse_known_args()

    # torchrun 会自动设置这些环境变量
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))

    if args.distributed and args.world_size > 1:
        distributed_main(args.local_rank, args)
    else:
        single_main()
