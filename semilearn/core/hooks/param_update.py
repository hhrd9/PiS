# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from .hook import Hook


class ParamUpdateHook(Hook):
    """
    Parameter Update Hook

    necessary for update the model parameters
    """
    
    def before_train_step(self, algorithm):
        if hasattr(algorithm, 'start_run'):
            torch.cuda.synchronize()
            algorithm.start_run.record()

    # call after each train_step to update parameters
    def after_train_step(self, algorithm):
        loss = algorithm.out_dict['loss']
        loss_pis = algorithm.out_pis_dict['loss']
        # algorithm.optimizer.zero_grad()
        # update parameters
        if algorithm.use_amp:
            algorithm.loss_scaler.scale(loss).backward()
            if (algorithm.clip_grad > 0):
                algorithm.loss_scaler.unscale_(algorithm.optimizer)
                torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
            algorithm.loss_scaler.step(algorithm.optimizer)
            # algorithm.loss_scaler.update()
        else:
            loss.backward()
            if (algorithm.clip_grad > 0):
                torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
            algorithm.optimizer.step()

        if algorithm.scheduler is not None:
            algorithm.scheduler.step()
        algorithm.model.zero_grad()
        # update pis model parameters
        if algorithm.use_amp:
            algorithm.loss_scaler.scale(loss_pis).backward()
            if (algorithm.clip_grad > 0):
                algorithm.loss_scaler.unscale_(algorithm.optimizer_pis)
                torch.nn.utils.clip_grad_norm_(algorithm.model_pis.parameters(), algorithm.clip_grad)
            algorithm.loss_scaler.step(algorithm.optimizer_pis)
            # algorithm.loss_scaler.update()
        else:
            loss_pis.backward()
            if (algorithm.clip_grad > 0):
                torch.nn.utils.clip_grad_norm_(algorithm.model_pis.parameters(), algorithm.clip_grad)
            algorithm.optimizer_pis.step()

        if algorithm.scheduler_pis is not None:
            algorithm.scheduler_pis.step()
        algorithm.model_pis.zero_grad()
        
        if algorithm.use_amp:
            algorithm.loss_scaler.update()

        if hasattr(algorithm, 'end_run'):
            algorithm.end_run.record()
            torch.cuda.synchronize()
            algorithm.log_dict['train/run_time'] = algorithm.start_run.elapsed_time(algorithm.end_run) / 1000.


# class ParamUpdateHook(Hook):

#     def before_train_step(self, algorithm):
#         if hasattr(algorithm, 'start_run'):
#             torch.cuda.synchronize()
#             algorithm.start_run.record()

#     def after_train_step(self, algorithm):
#         loss = algorithm.out_dict['loss']
#         loss_pis = algorithm.out_pis_dict['loss']
#         alpha = getattr(algorithm, 'pis_from_model_grad_alpha', 0.6)
#         mode = getattr(algorithm, 'pis_merge_mode', 'delta')

#         loss.backward()  # grads -> model.parameters().grad

#         if (algorithm.clip_grad > 0):
#             torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)

#         old_params = [p.data.clone() for p in algorithm.model.parameters()]  # clone the .data (device-local)
#         # step model
#         algorithm.optimizer.step()
#         if algorithm.scheduler is not None:
#             algorithm.scheduler.step()
#         added_count = 0
#         for old_p, p_new, p_pis in zip(old_params, algorithm.model.parameters(), algorithm.model_pis.parameters()):
#             # delta on same device
#             delta = p_new.data - old_p  # this is a view of p_new.data - old_p
#             if delta is None:
#                 continue
#             # check shape (should match)
#             if delta.shape != p_pis.data.shape:
#                 print("[ParamUpdateHook WARN] delta shape mismatch; skipping (delta mode).")
#                 continue
#             # apply to model_pis parameter data in-place
#             # p_pis.data += alpha * delta
#             with torch.no_grad():
#                 p_pis.data.add_(alpha * delta)
#             added_count += 1

#         algorithm.model.zero_grad()

#         loss_pis.backward()
#         if (algorithm.clip_grad > 0):
#             torch.nn.utils.clip_grad_norm_(algorithm.model_pis.parameters(), algorithm.clip_grad)
#         algorithm.optimizer_pis.step()
#         if algorithm.scheduler_pis is not None:
#             algorithm.scheduler_pis.step()
#         algorithm.model_pis.zero_grad()

#         if hasattr(algorithm, 'end_run'):
#             algorithm.end_run.record()
#             torch.cuda.synchronize()
#             algorithm.log_dict['train/run_time'] = algorithm.start_run.elapsed_time(algorithm.end_run) / 1000.

