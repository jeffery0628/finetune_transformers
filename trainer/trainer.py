# -*- coding: utf-8 -*-
# @Time    : 2020/8/27 3:17 下午
# @Author  : jeffery
# @FileName: trainer.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
import time

import numpy as np
import torch

from base import BaseTrainer
from model import get_entities
from utils import inf_loop, MetricTracker, convert_tag_ids_tags


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_inference = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()

        self.train_metrics.reset()
        t0 = time.time()
        for batch_idx, data in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            tids, input_ids, attention_masks, tag_ids, tags, text_lengths = data
            if 'cuda' == self.device.type:
                input_ids = input_ids.cuda()
                attention_masks = attention_masks.cuda()
                tag_ids = tag_ids.cuda()
            pred = self.model(input_ids, attention_masks)
            loss = self.criterion[0](pred, tag_ids)
            loss.backward()
            self.optimizer.step()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            pred_np = torch.argmax(pred, dim=-1).cpu().detach().numpy()

            pred_tags = []
            for pred_id, length, gold_tag in zip(pred_np, text_lengths, tags):
                pred_id = pred_id[1:length + 1]
                pred_tag = convert_tag_ids_tags(pred_id)
                assert len(pred_tag) == len(gold_tag), 'tag length is not equal'
                pred_tags.append(pred_tag)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(pred_tags, tags))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.3f}'.format(epoch, self._progress(batch_idx),
                                                                           loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        print('training time : {}'.format(time.time() - t0))
        t1 = time.time()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
        print('valid time : {}'.format(time.time() - t1))
        if self.do_inference:
            self._inference_epoch(epoch)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                tids, input_ids, attention_masks, tag_ids, tags, text_lengths = data
                if 'cuda' == self.device.type:
                    input_ids = input_ids.cuda()
                    attention_masks = attention_masks.cuda()
                    tag_ids = tag_ids.cuda()
                pred = self.model(input_ids, attention_masks)
                loss = self.criterion[0](pred, tag_ids)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                pred_np = torch.argmax(pred, dim=-1).cpu().detach().numpy()

                pred_tags = []
                for pred_id, length, gold_tag in zip(pred_np, text_lengths, tags):
                    pred_id = pred_id[1:length + 1]
                    pred_tag = convert_tag_ids_tags(pred_id)
                    assert len(pred_tag) == len(gold_tag), 'tag length is not equal'
                    pred_tags.append(pred_tag)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(pred_tags, tags))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _inference_epoch(self, epoch):
        """
        Inference after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        label_result = []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_data_loader):
                tids, input_ids, attention_masks, tag_ids, tags, text_lengths = data
                if 'cuda' == self.device.type:
                    input_ids = input_ids.cuda()
                    attention_masks = attention_masks.cuda()
                    tag_ids = tag_ids.cuda()
                pred = self.model(input_ids, attention_masks)

                pred_np = torch.argmax(pred, dim=-1).cpu().detach().numpy()

                for tid, pred_id, length, gold_tag in zip(tids, pred_np, text_lengths, tags):
                    pred_id = pred_id[1:length + 1]
                    pred_tag = convert_tag_ids_tags(pred_id)
                    assert len(pred_tag) == len(gold_tag), 'tag length is not equal'
                    entities = get_entities(pred_tag)
                    label_result.append((tid, entities))

        label_result = sorted(label_result, key=lambda x: x[0])
        result_file = self.logger_dir / '{}_{}.txt'.format(self.ner_type, epoch)
        print('result saved to {}'.format(result_file))
        with open(result_file, 'w') as f:
            for item in label_result:
                f.write(str(item[0]) + '\t' + str(item[1]) + '\n')

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
