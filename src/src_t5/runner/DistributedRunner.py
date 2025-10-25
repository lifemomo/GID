from runner.SingleRunner import SingleRunner
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.distributed as dist
import logging
from tqdm import tqdm
from utils import utils
import torch
import utils.generation_trie as gt
import utils.evaluate as evaluate
from torch.utils.data.distributed import DistributedSampler
from data.TestDataset import TestDataset
from torch.utils.data import DataLoader
from processor.Collator import Collator, TestCollator
import time
import numpy as np
import random
import shutil  # 用于删除旧checkpoint
import pdb
from torch.nn.parallel import DistributedDataParallel as DDP
import os


class DistributedRunner(SingleRunner):

    def __init__(self, model, tokenizer, train_loader, valid_loader, device, args, rank):
        super().__init__(model, tokenizer, train_loader, valid_loader, device, args)
        self.rank = rank
        # ✅ 避免重复包裹
        if not isinstance(self.model, DDP):
            self.model = DDP(self.model, device_ids=[self.args.gpu], find_unused_parameters=True)

        self.best_score = -float('inf')
        self.best_checkpoint_path = None  # 保存之前的best模型路径，便于删除
        self.current_epoch = 0  # 记录当前epoch数

    def train(self):

        self.model.zero_grad()
        train_losses = []
        valid_losses = []
        best_epoch = -1
        if self.test_before_train > 0:
            self.test()

        for epoch in range(self.args.epochs):
            self.current_epoch = epoch + 1  # 👈 放在这里，最开头！！
            if self.rank == 0:
                logging.info(f"Start training for epoch {epoch+1}")

            dist.barrier()
            if self.regenerate_candidate:
                for ds in self.train_loader.dataset.datasets:
                    ds.generate_candidates()
                    ds.construct_sentence()
            elif self.reconstruct_data:
                for ds in self.train_loader.dataset.datasets:
                    ds.construct_sentence()

            self.train_loader.sampler.set_epoch(epoch)
            dist.barrier()

            self.model.train()
            losses = []

            for batch in tqdm(self.train_loader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)

                output = self.model.module(
                    input_ids=input_ids,
                    whole_word_ids=whole_input_ids,
                    attention_mask=attn,
                    labels=output_ids,
                    alpha=self.args.alpha,
                    return_dict=True,
                )
                # compute loss masking padded tokens
                loss = output["loss"]
                lm_mask = output_attention != 0
                lm_mask = lm_mask.float()
                B, L = output_ids.size()
                loss = loss.view(B, L) * lm_mask
                loss = (loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

                # update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

                dist.barrier()

                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()


                dist.all_reduce(loss.detach(), op=dist.ReduceOp.SUM)
                loss /= dist.get_world_size()

                dist.barrier()

                if self.rank == 0:
                    losses.append(loss.detach())

            if self.rank == 0:
                train_epoch_loss = sum(losses)/len(losses)
                train_losses.append(train_epoch_loss)
                logging.info(f"The average training loss for epoch {epoch+1} is {train_epoch_loss}")



            if self.valid_select > 0:
                if self.rank == 0:
                    logging.info(f"Start validation for epoch {epoch+1}")
                losses = []
                self.model.eval()
                with torch.no_grad():
                    if self.args.valid_prompt_sample > 0:
                        for ds in self.valid_loader.dataset.datasets:
                            ds.construct_sentence()
                    for batch in tqdm(self.valid_loader):
                        input_ids = batch[0].to(self.device)
                        attn = batch[1].to(self.device)
                        whole_input_ids = batch[2].to(self.device)
                        output_ids = batch[3].to(self.device)
                        output_attention = batch[4].to(self.device)

                        output = self.model.module(
                            input_ids=input_ids,
                            whole_word_ids=whole_input_ids,
                            attention_mask=attn,
                            labels=output_ids,
                            alpha=self.args.alpha,
                            return_dict=True,
                        )
                        # compute loss masking padded tokens
                        loss = output["loss"]
                        lm_mask = output_attention != 0
                        lm_mask = lm_mask.float()
                        B, L = output_ids.size()
                        loss = loss.view(B, L) * lm_mask
                        loss = (loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

                        dist.barrier()

                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        loss /= dist.get_world_size()

                        dist.barrier()

                        if self.rank == 0:
                            losses.append(loss)

                    if self.rank == 0:
                        valid_epoch_loss = sum(losses)/len(losses)
                        valid_losses.append(valid_epoch_loss)
                        logging.info(f"The average valid loss for epoch {epoch+1} is {valid_epoch_loss}")

                        if valid_epoch_loss == min(valid_losses):
                            logging.info(f"The minimal validation loss so far.")
                            best_epoch = epoch + 1
                            torch.save(self.model.module.state_dict(), self.args.model_path)
                            logging.info(f"Save the current model to {self.args.model_path}")

            if self.test_epoch > 0:
                self.current_epoch = epoch + 1
                if (epoch + 1) % self.test_epoch == 0:
                    self.model.eval()
                    self.test()

            dist.barrier()
        if self.valid_select > 0:
            if self.rank == 0:
                logging.info(f"The best validation at Epoch {best_epoch}")
        else:
            if self.rank == 0:
                torch.save(self.model.module.state_dict(), self.args.model_path)
                logging.info(f"Save the current model to {self.args.model_path}")

        return

    def get_testloader(self):
        self.testloaders = []
        datasets = self.args.datasets.split(',')
        tasks = self.args.tasks.split(',')
        if self.test_filtered > 0:
            collator = TestCollator(self.tokenizer)
        else:
            collator = Collator(self.tokenizer)
        for dataset in datasets:
            for task in tasks:

                testdata = TestDataset(self.args, dataset, task)
                test_sampler = DistributedSampler(testdata)
                testloader = DataLoader(dataset=testdata, sampler=test_sampler, batch_size=self.args.eval_batch_size, collate_fn=collator, shuffle=False)
                self.testloaders.append(testloader)

    def test(self, path=None):
        print("[DEBUG] Entered DistributedRunner.test() ✅")
        # ✅ 1. 先unwrap model
        real_model = self.model.module if isinstance(self.model,
                                                     torch.nn.parallel.DistributedDataParallel) else self.model
        print(f"[DEBUG] real_model type in test(): {type(real_model)}")  # 🌟打印一下确认
        print(f"[DEBUG] real_model id: {id(real_model)}")
        # ✅ 2. set eval mode
        real_model.eval()

        # ✅ 3. load checkpoint（如果有path）
        if path:
            real_model.load_state_dict(torch.load(path, map_location=self.device), strict=False)

        # ✅ 4. 测试
        for loader in self.testloaders:
            if self.test_filtered > 0:
                if self.test_filtered_batch > 0:
                    self.test_dataset_task_filtered_batch(loader, real_model)  # ✅传的是unwrap后的real_model
                else:
                    assert self.args.eval_batch_size == 1
                    self.test_dataset_task_filtered(loader, real_model)  # ✅传的是unwrap后的real_model
            else:
                self.test_dataset_task(loader, real_model)  # ✅传的是unwrap后的real_model

    def test_dataset_task_filtered_batch(self, testloader, model):
        # ❌ 不要再重新unwrap了！ model已经是外面unwrap过的 real_model！
        print(f"[DEBUG] Received model type in test_dataset_task_filtered_batch: {type(model)}")
        assert not isinstance(model, torch.nn.parallel.DistributedDataParallel), \
            "[ERROR] model is still DDP wrapped!"

        logging.info(f'testing filtered {testloader.dataset.dataset} dataset on {testloader.dataset.task} task')
        test_total = 0

        with torch.no_grad():
            candidates = testloader.dataset.all_items
            candidate_trie = gt.Trie(
                [
                    [0] + self.tokenizer.encode(f"{testloader.dataset.dataset} item_{candidate}")
                    for candidate in candidates
                ]
            )
            prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)

            metrics_res = np.array([0.0] * len(self.metrics))
            for batch in tqdm(testloader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                user_idx = batch[5]

                # ✅ 用传入的model直接generate
                prediction = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    whole_word_ids=whole_input_ids,
                    max_length=30,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=self.generate_num + testloader.dataset.max_positive,
                    num_return_sequences=self.generate_num + testloader.dataset.max_positive,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                prediction_ids = prediction["sequences"]
                prediction_scores = prediction["sequences_scores"]

                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )

                rel_results = evaluate.rel_results_filtered(
                    testloader.dataset.positive,
                    testloader.dataset.id2user,
                    user_idx.detach().cpu().numpy(),
                    self.generate_num + testloader.dataset.max_positive,
                    generated_sents,
                    gold_sents,
                    prediction_scores,
                    self.generate_num
                )

                test_total += len(rel_results)
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)

            dist.barrier()
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            dist.all_reduce(metrics_res, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_total, op=dist.ReduceOp.SUM)

            metrics_res /= test_total

            if self.rank == 0:
                for i in range(len(self.metrics)):
                    logging.info(f'{self.metrics[i]}: {metrics_res[i]}')

    def test_dataset_task_filtered(self, testloader, model):
        # ❌ 不能重新unwrap！直接用传进来的 real_model
        print(f"[DEBUG] Received model type in test_dataset_task_filtered: {type(model)}")
        assert not isinstance(model, torch.nn.parallel.DistributedDataParallel), \
            "[ERROR] model is still DDP wrapped!"

        logging.info(f'testing filtered {testloader.dataset.dataset} dataset on {testloader.dataset.task} task')
        test_total = 0

        with torch.no_grad():
            candidates = set(testloader.dataset.all_items)

            metrics_res = np.array([0.0] * len(self.metrics))
            for batch in tqdm(testloader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                user_idx = int(batch[5][0])

                positive = testloader.dataset.positive[testloader.dataset.id2user[user_idx]]
                user_candidate = candidates - positive

                candidate_trie = gt.Trie(
                    [
                        [0] + self.tokenizer.encode(f"{testloader.dataset.dataset} item_{candidate}")
                        for candidate in user_candidate
                    ]
                )
                prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)

                # ✅ 用传入的model直接generate
                prediction = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    whole_word_ids=whole_input_ids,
                    max_length=30,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=self.generate_num,
                    num_return_sequences=self.generate_num,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                prediction_ids = prediction["sequences"]
                prediction_scores = prediction["sequences_scores"]

                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )

                rel_results = evaluate.rel_results(
                    generated_sents,
                    gold_sents,
                    prediction_scores,
                    self.generate_num
                )

                test_total += len(rel_results)
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)

            dist.barrier()
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            dist.all_reduce(metrics_res, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_total, op=dist.ReduceOp.SUM)

            metrics_res /= test_total

            if self.rank == 0:
                for i in range(len(self.metrics)):
                    logging.info(f'{self.metrics[i]}: {metrics_res[i]}')

    def test_dataset_task(self, testloader, model):
        # ❌ 不能再次unwrap！
        print(f"[DEBUG] Received model type in test_dataset_task: {type(model)}")
        assert not isinstance(model, torch.nn.parallel.DistributedDataParallel), \
            "[ERROR] model is still DDP wrapped!"

        if self.rank == 0:
            logging.info(f'testing {testloader.dataset.dataset} dataset on {testloader.dataset.task} task')
        print(f"[DEBUG] model id in test_dataset_task: {id(model)}")
        test_total = 0
        with torch.no_grad():
            candidates = testloader.dataset.all_items
            candidate_trie = gt.Trie(
                [
                    [0] + self.tokenizer.encode(f"{testloader.dataset.dataset} item_{candidate}")
                    for candidate in candidates
                ]
            )
            prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)

            metrics_res = np.array([0.0] * len(self.metrics))
            for batch in tqdm(testloader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)

                # ✅ 用传入的real model直接generate！
                prediction = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    whole_word_ids=whole_input_ids,
                    max_length=50,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=self.generate_num,
                    num_return_sequences=self.generate_num,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                prediction_ids = prediction["sequences"]
                prediction_scores = prediction["sequences_scores"]

                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )

                rel_results = evaluate.rel_results(
                    generated_sents,
                    gold_sents,
                    prediction_scores,
                    self.generate_num
                )

                test_total += len(rel_results)
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)

            dist.barrier()
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            dist.all_reduce(metrics_res, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_total, op=dist.ReduceOp.SUM)

            metrics_res /= test_total

            if self.rank == 0:
                for i in range(len(self.metrics)):
                    logging.info(f'{self.metrics[i]}: {metrics_res[i]}')

                # ✅ 保存 checkpoint
                weights = torch.tensor([0.3, 0.15, 0.3, 0.15]).to(self.device)
                current_score = (metrics_res * weights).sum().item()
                logging.info(
                    f"[Weighted Score] Current score: {current_score:.4f} (previous best: {self.best_score:.4f})")

                if current_score > self.best_score:
                    logging.info(f"New best score found: {current_score:.4f} (previous best: {self.best_score:.4f})")
                    self.best_score = current_score

                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    save_dir = os.path.dirname(self.args.model_path)
                    save_name = f"best_epoch{self.current_epoch:02d}_all{current_score:.4f}.pt"
                    save_path = os.path.join(save_dir, save_name)

                    torch.save(model_to_save.state_dict(), save_path)
                    logging.info(f"🌟 Best model saved to {save_path}")

                    if self.best_checkpoint_path is not None and os.path.exists(self.best_checkpoint_path):
                        os.remove(self.best_checkpoint_path)
                        logging.info(f"🗑️ Deleted old checkpoint: {self.best_checkpoint_path}")

                    self.best_checkpoint_path = save_path



