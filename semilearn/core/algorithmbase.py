# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import contextlib
import numpy as np
from inspect import signature
from collections import OrderedDict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from semilearn.core.hooks import Hook, get_priority, CheckpointHook, TimerHook, LoggingHook, DistSamplerSeedHook, ParamUpdateHook, EvaluationHook, EMAHook, WANDBHook, AimHook
from semilearn.core.utils import get_dataset, get_data_loader, get_optimizer, get_cosine_schedule_with_warmup, Bn_Controller, get_net_builder
from semilearn.core.criterions import CELoss, ConsistencyLoss


class AlgorithmBase:
    """
    Base class for semi-supervised learning algorithms.
    Initializes algorithm-specific and common training components.

    Args:
        args (argparse.Namespace): Command-line arguments or configuration object.
        net_builder (callable): Network architecture builder function.
        tb_log (TBLog, optional): TensorBoard logger instance. Defaults to None.
        logger (logging.Logger, optional): System logger instance. Defaults to None.
    """
    def __init__(
        self,
        args,
        net_builder,
        tb_log=None,
        logger=None,
        **kwargs):
        
        # Common arguments
        self.args = args
        self.num_classes = args.num_classes
        self.ema_m = args.ema_m
        self.epochs = args.epoch
        self.num_train_iter = args.num_train_iter
        self.num_eval_iter = args.num_eval_iter
        self.num_log_iter = args.num_log_iter
        self.num_iter_per_epoch = int(self.num_train_iter // self.epochs)
        self.lambda_u = args.ulb_loss_ratio 
        self.use_cat = args.use_cat
        self.use_amp = args.amp
        self.clip_grad = args.clip_grad
        self.save_name = args.save_name
        self.save_dir = args.save_dir
        self.resume = args.resume
        self.algorithm = args.algorithm

        # Common utility arguments
        self.tb_log = tb_log
        self.print_fn = print if logger is None else logger.info
        self.ngpus_per_node = torch.cuda.device_count()
        self.loss_scaler = GradScaler()
        self.amp_cm = autocast if self.use_amp else contextlib.nullcontext
        self.gpu = args.gpu
        self.rank = args.rank
        self.distributed = args.distributed
        self.world_size = args.world_size
        
        # Common model parameters
        self.it = 0
        self.start_epoch = 0
        self.best_eval_acc, self.best_it = 0.0, 0
        self.bn_controller = Bn_Controller()
        self.net_builder = net_builder
        self.ema = None
        
        # Flag for enabling EMA parameter replacement
        self.use_pis_ema = getattr(self.args, 'use_pis_ema', True)

        # Build dataset
        self.dataset_dict = self.set_dataset()

        # Build data loaders
        self.loader_dict = self.set_data_loader()

        # Build models
        self.model = self.set_model()
        self.model_pis = self.set_model()
        self.ema_model = self.set_ema_model()
        
        # Initialize EMA model for model_pis if enabled
        if self.use_pis_ema:
            self.ema_model_pis = self.set_ema_model_pis()
            self.pis_ema_decay = getattr(self.args, 'pis_ema_decay', 0.999)
        else:
            self.ema_model_pis = None

        # Build optimizers and schedulers
        self.optimizer, self.scheduler = self.set_optimizer()
        self.optimizer_pis, self.scheduler_pis = self.set_optimizer_pis()

        # Build supervised and unsupervised loss functions
        self.ce_loss = CELoss()
        self.consistency_loss = ConsistencyLoss()

        # Register training hooks
        self._hooks = []  
        self.hooks_dict = OrderedDict() 
        self.set_hooks()

        
    def init(self, **kwargs):
        """
        Algorithm-specific initialization function.
        """
        raise NotImplementedError
    

    def set_dataset(self):
        """
        Initialize dataset dictionary.
        """
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        dataset_dict = get_dataset(self.args, self.algorithm, self.args.dataset, self.args.num_labels, self.args.num_classes, self.args.data_dir, self.args.include_lb_to_ulb)
        if dataset_dict is None:
            return dataset_dict

        self.args.ulb_dest_len = len(dataset_dict['train_ulb']) if dataset_dict['train_ulb'] is not None else 0
        self.args.lb_dest_len = len(dataset_dict['train_lb'])
        self.print_fn("Unlabeled data count: {}, Labeled data count: {}".format(self.args.ulb_dest_len, self.args.lb_dest_len))
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict

    def set_data_loader(self):
        """
        Initialize data loaders.
        """
        if self.dataset_dict is None:
            return
            
        self.print_fn("Creating train and test data loaders")
        loader_dict = {}
        loader_dict['train_lb'] = get_data_loader(self.args,
                                                  self.dataset_dict['train_lb'],
                                                  self.args.batch_size,
                                                  data_sampler=self.args.train_sampler,
                                                  num_iters=self.num_train_iter,
                                                  num_epochs=self.epochs,
                                                  num_workers=self.args.num_workers,
                                                  distributed=self.distributed)

        loader_dict['train_ulb'] = get_data_loader(self.args,
                                                   self.dataset_dict['train_ulb'],
                                                   self.args.batch_size * self.args.uratio,
                                                   data_sampler=self.args.train_sampler,
                                                   num_iters=self.num_train_iter,
                                                   num_epochs=self.epochs,
                                                   num_workers=2 * self.args.num_workers,
                                                   distributed=self.distributed)

        loader_dict['eval'] = get_data_loader(self.args,
                                              self.dataset_dict['eval'],
                                              self.args.eval_batch_size,
                                              data_sampler=None,
                                              num_workers=self.args.num_workers,
                                              drop_last=False)
        
        if self.dataset_dict['test'] is not None:
            loader_dict['test'] = get_data_loader(self.args,
                                                  self.dataset_dict['test'],
                                                  self.args.eval_batch_size,
                                                  data_sampler=None,
                                                  num_workers=self.args.num_workers,
                                                  drop_last=False)
        self.print_fn(f'[!] Data loader keys: {loader_dict.keys()}')
        return loader_dict

    def set_optimizer(self):
        """
        Initialize optimizer and scheduler for primary model.
        """
        self.print_fn("Creating optimizer and scheduler")
        optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    self.num_train_iter,
                                                    num_warmup_steps=self.args.num_warmup_iter)
        return optimizer, scheduler
    
    def set_optimizer_pis(self):
        """
        Initialize optimizer and scheduler for PIS model.
        """
        self.print_fn("Creating PIS optimizer and scheduler")
        optimizer = get_optimizer(self.model_pis, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    self.num_train_iter,
                                                    num_warmup_steps=self.args.num_warmup_iter)
        return optimizer, scheduler

    def set_model(self):
        """
        Initialize model instance.
        """
        model = self.net_builder(num_classes=self.num_classes, pretrained=self.args.use_pretrain, pretrained_path=self.args.pretrain_path)
        return model

    def set_ema_model(self):
        """
        Initialize EMA model from target model.
        """
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model
        
    def set_ema_model_pis(self):
        """
        Initialize EMA model from PIS model and transfer to target GPU device.
        """
        ema_model_pis = self.net_builder(num_classes=self.num_classes)
        ema_model_pis.load_state_dict(self.model_pis.state_dict())
        if hasattr(self, 'gpu') and self.gpu is not None:
            ema_model_pis = ema_model_pis.cuda(self.gpu)
        return ema_model_pis

    def set_hooks(self):
        """
        Register standard training hooks.
        """
        self.register_hook(ParamUpdateHook(), None, "HIGHEST")
        self.register_hook(EMAHook(), None, "HIGH")
        self.register_hook(EvaluationHook(), None, "HIGH")
        self.register_hook(CheckpointHook(), None, "HIGH")
        self.register_hook(DistSamplerSeedHook(), None, "NORMAL")
        self.register_hook(TimerHook(), None, "LOW")
        self.register_hook(LoggingHook(), None, "LOWEST")
        if getattr(self.args, 'use_wandb', False):
            self.register_hook(WANDBHook(), None, "LOWEST")
        if getattr(self.args, 'use_aim', False):
            self.register_hook(AimHook(), None, "LOWEST")

    def process_batch(self, input_args=None, **kwargs):
        """
        Process batch data and transfer tensors to target GPU device.
        """
        if input_args is None:
            input_args = signature(self.train_step).parameters
            input_args = list(input_args.keys())

        input_dict = {}

        for arg, var in kwargs.items():
            if arg not in input_args or var is None:
                continue
            
            if isinstance(var, dict):
                var = {k: v.cuda(self.gpu) for k, v in var.items()}
            else:
                var = var.cuda(self.gpu)
            input_dict[arg] = var
        return input_dict
    
    def copy_params_inplace(self, model_target, model_source):
        """
        Copy parameters from source model to target model in-place without tracking gradients.
        """
        with torch.no_grad():
            for p_t, p_s in zip(model_target.parameters(), model_source.parameters()):
                p_t.copy_(p_s)  
    
    def _get_overwrite_config(self):
        """
        Dynamically calculate warmup steps and overwrite interval steps based on total training iterations.
        """
        warmup_ratio = getattr(self.args, 'warmup_ratio', 0.08)
        reset_ratio = getattr(self.args, 'reset_ratio', 0.025)

        warmup_iters = int(200000 * warmup_ratio)
        reset_interval = int(200000 * reset_ratio)

        return warmup_iters, reset_interval

    def _should_overwrite_params(self) -> bool:
        """
        Evaluate whether current iteration satisfies parameter overwrite criteria.
        """
        warmup_enabled = getattr(self.args, 'warmup', False)
        warmup_iters, reset_interval = self._get_overwrite_config()

        if reset_interval <= 0:
            return False

        if warmup_enabled:
            return (self.it >= warmup_iters) and ((self.it - warmup_iters -1) % reset_interval == 0)
        else:
            return self.it % reset_interval == 0

    def _apply_pis_overwrite_and_reset(self):
        """
        Execute parameter overwrite and reset operations.
        If EMA mode is enabled, overwrite primary model with accumulated EMA parameters and reset EMA state.
        Otherwise, overwrite primary model directly with current PIS model parameters.
        """
        if self.use_pis_ema:
            self.copy_params_inplace(self.model, self.ema_model_pis)
            self.copy_params_inplace(self.ema_model_pis, self.model_pis)
        else:
            self.copy_params_inplace(self.model, self.model_pis)

    def update_ema_model_pis(self):
        """
        Update EMA parameters and buffers of PIS model using exponential moving average.
        """
        with torch.no_grad():
            for param_train, param_eval in zip(self.model_pis.parameters(), self.ema_model_pis.parameters()):
                if param_eval.device != param_train.device:
                    param_eval.data = param_eval.data.to(param_train.device)
                param_eval.copy_(param_eval * self.pis_ema_decay + param_train.detach() * (1 - self.pis_ema_decay))
                
            for buffer_train, buffer_eval in zip(self.model_pis.buffers(), self.ema_model_pis.buffers()):
                if buffer_eval.device != buffer_train.device:
                    buffer_eval.data = buffer_eval.data.to(buffer_train.device)
                buffer_eval.copy_(buffer_train)
                    
    def process_out_dict(self, out_dict=None, **kwargs):
        """
        Process output dictionary returned by train_step.
        """
        if out_dict is None:
            out_dict = {}

        for arg, var in kwargs.items():
            out_dict[arg] = var
        return out_dict

    def process_log_dict(self, log_dict=None, prefix='train', **kwargs):
        """
        Process logging dictionary for TensorBoard or external loggers.
        """
        if log_dict is None:
            log_dict = {}

        for arg, var in kwargs.items():
            log_dict[f'{prefix}/' + arg] = var
        return log_dict

    def compute_prob(self, logits):
        """
        Compute class probabilities from logits.
        """
        return torch.softmax(logits, dim=-1)

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s, x_ulb_pis, y_ulb):
        """
        Execute single training iteration step. Must be implemented in subclasses.
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop execution.
        """
        self.model.train()
        self.model_pis.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            if self.it >= self.num_train_iter:
                break
            
            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                if self.it >= self.num_train_iter:
                    break  
                
                if self._should_overwrite_params():
                    self._apply_pis_overwrite_and_reset()
          
                self.call_hook("before_train_step")
                self.out_dict, self.out_pis_dict, self.log_dict = self.train_step(
                    **self.process_batch(**data_lb, **data_ulb)
                )
                
                if self.use_pis_ema:
                    self.update_ema_model_pis()

                self.call_hook("after_train_step")
                self.it += 1
            
            self.call_hook("after_train_epoch")

        self.call_hook("after_run")
        
    def get_logits(self, data, out_key):
        """
        Forward pass through primary model to obtain output logits.
        """
        x = data['x_lb']
        if isinstance(x, dict):
            x = {k: v.cuda(self.gpu) for k, v in x.items()}
        else:
            x = x.cuda(self.gpu)

        return self.model(x)[out_key]
    
    def get_logits_pis(self, data, out_key):
        """
        Forward pass through PIS model to obtain output logits.
        """
        x = data['x_lb']
        if isinstance(x, dict):
            x = {k: v.cuda(self.gpu) for k, v in x.items()}
        else:
            x = x.cuda(self.gpu)

        return self.model_pis(x)[out_key]
    
    def get_targets(self, data):
        """
        Extract labeled target tensor and move to GPU device.
        """
        y = data['y_lb']
        return y.cuda(self.gpu)

    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        """
        Evaluate performance metrics on evaluation or test dataset.
        """
        self.model.eval()
        self.model_pis.eval()
        if self.ema is not None:
            self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true, y_pred, y_pred_pis, y_probs, y_logits = [], [], [], [], []

        with torch.no_grad():
            for data in eval_loader:
                logits = self.get_logits(data, out_key)
                logits_pis = self.get_logits_pis(data, out_key)
                y = self.get_targets(data)

                num_batch = y.shape[0]
                total_num += num_batch
                
                loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_pred_pis.extend(torch.max(logits_pis, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                y_probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
                total_loss += loss.item() * num_batch

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_pis = np.array(y_pred_pis)
        y_logits = np.concatenate(y_logits)
        
        top1 = accuracy_score(y_true, y_pred)
        top1_pis = accuracy_score(y_true, y_pred_pis)
        top5 = top_k_accuracy_score(y_true, y_probs, k=5)
        balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('Confusion Matrix:\n' + np.array_str(cf_mat))

        if self.ema is not None:
            self.ema.restore()
            
        self.model.train()
        self.model_pis.train()

        eval_dict = {
            eval_dest + '/loss': total_loss / total_num,
            eval_dest + '/top-1-acc': top1,
            eval_dest + '/pis-top-1-acc': top1_pis,
            eval_dest + '/top-5-acc': top5, 
            eval_dest + '/balanced_acc': balanced_top1,
            eval_dest + '/precision': precision,
            eval_dest + '/recall': recall,
            eval_dest + '/F1': f1
        }
        if return_logits:
            eval_dict[eval_dest + '/logits'] = y_logits
        return eval_dict

    def get_save_dict(self):
        """
        Construct dictionary containing parameters and states for checkpoint saving.
        """
        save_dict = {
            'model': self.model.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_scaler': self.loss_scaler.state_dict(),
            'it': self.it + 1,
            'epoch': self.epoch + 1,
            'best_it': self.best_it,
            'best_eval_acc': self.best_eval_acc,
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        return save_dict

    def save_model(self, save_name, save_path):
        """
        Save model checkpoint to specified directory.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        save_filename = os.path.join(save_path, save_name)
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_filename)
        self.print_fn(f"Model saved: {save_filename}")

    def load_model(self, load_path):
        """
        Load model weights and training state from checkpoint file.
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        self.it = checkpoint['it']
        self.start_epoch = checkpoint['epoch']
        self.epoch = self.start_epoch
        self.best_it = checkpoint['best_it']
        self.best_eval_acc = checkpoint['best_eval_acc']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler is not None and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.print_fn('Model loaded')
        return checkpoint

    def check_prefix_state_dict(self, state_dict):
        """
        Remove module prefix from state dictionary keys generated during distributed training.
        """
        new_state_dict = dict()
        for key, item in state_dict.items():
            if key.startswith('module'):
                new_key = '.'.join(key.split('.')[1:])
            else:
                new_key = key
            new_state_dict[new_key] = item
        return new_state_dict

    def register_hook(self, hook, name=None, priority='NORMAL'):
        """
        Register a training hook into the priority queue.
        Ref: https://github.com/open-mmlab/mmcv/blob/a08517790d26f8761910cac47ce8098faac7b627/mmcv/runner/base_runner.py#L263
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  
        hook.name = name if name is not None else type(hook).__name__

        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        
        if not inserted:
            self._hooks.insert(0, hook)

        self.hooks_dict = OrderedDict()
        for hook in self._hooks:
            self.hooks_dict[hook.name] = hook

    def call_hook(self, fn_name, hook_name=None, *args, **kwargs):
        """
        Call specified function across registered hooks.
        """
        if hook_name is not None:
            return getattr(self.hooks_dict[hook_name], fn_name)(self, *args, **kwargs)
        
        for hook in self.hooks_dict.values():
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args, **kwargs)

    def registered_hook(self, hook_name):
        """
        Check whether a specific hook is registered.
        """
        return hook_name in self.hooks_dict



class ImbAlgorithmBase(AlgorithmBase):
    """
    Base class for class-imbalanced semi-supervised learning algorithms.
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        
        # Class imbalance arguments
        self.lb_imb_ratio = self.args.lb_imb_ratio
        self.ulb_imb_ratio = self.args.ulb_imb_ratio
        self.imb_algorithm = self.args.imb_algorithm
    
    def imb_init(self, *args, **kwargs):
        """
        Initialize class-imbalanced algorithm parameters.
        """
        pass 

    def set_optimizer(self):
        """
        Initialize optimizer with dataset and network specific configurations for class imbalance settings.
        """
        if 'vit' in self.args.net and self.args.dataset in ['cifar100', 'food101', 'semi_aves', 'semi_aves_out']:
            return super().set_optimizer() 
        elif self.args.dataset in ['imagenet', 'imagenet127']:
            return super().set_optimizer() 
        else:
            self.print_fn("Creating optimizer and scheduler")
            optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay, bn_wd_skip=False)
            scheduler = None
            return optimizer, scheduler