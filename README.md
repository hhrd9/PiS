# Decoupled Optimization for Teacher–Student Semi-Supervised Learning via a Pioneer Student

This repository provides the implementation of **Pioneer Student (PiS)**, a method for decoupled optimization in teacher–student semi-supervised learning. The released experiments integrate PiS with **FreeMatch**.

## Relationship to USB

This codebase is derived from Microsoft's [Unified Semi-supervised Learning Benchmark (USB)](https://github.com/microsoft/Semi-supervised-learning). Most of the data pipeline, training framework, model code, utilities, and baseline implementation are inherited from USB. The PiS-specific changes are primarily in the training logic and the FreeMatch path, including the additional Pioneer Student model and its optimization/update procedure.

USB is distributed under the MIT License. The original copyright and license notices have been retained in the source files and are reproduced in [`LICENSE.txt`](LICENSE.txt). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution and a summary of the derived-work relationship.

## Method and implementation note

In the paper, after each fixed interval, the current PiS model is directly copied to the student model. In this released implementation, the default behavior is intentionally more stable:

- `use_pis_ema: True` is enabled by default in the provided configuration;
- the PiS model is maintained with an exponential moving average (EMA);
- at an overwrite point, the EMA PiS model is copied to the student model, and the EMA state is reset from the current PiS model;
- `warmup: False` is the default in the provided configuration.

The motivation is that directly using the PiS model from the last iteration of a stage may select an unfavorable checkpoint and can be unstable. Using the averaged PiS model as the overwrite source is more stable and performs better in our implementation. It also means that the warmup mechanism described in the paper is not needed in as many settings.

**Important reproducibility note:** the results reported in the paper were obtained **without** `use_pis_ema`. Therefore, the current repository defaults do not exactly correspond to the paper's reported setting. To reproduce the paper's overwrite rule, set `use_pis_ema: False` and use the remaining training settings specified by the corresponding experiment configuration. To use the implementation's more stable default, keep `use_pis_ema: True` and `warmup: False`.

## Getting started

### 1. Installation and data

Please follow the environment setup and dataset preparation instructions in the original [USB README](https://github.com/microsoft/Semi-supervised-learning/blob/main/README.md) and the instructions in [`preprocess/README.md`](preprocess/README.md).

Install the dependencies with:

```bash
pip install -r requirements.txt
```

### 2. Training PiS+FreeMatch

For example, to train PiS+FreeMatch on CIFAR-100 with 200 labeled examples:

```bash
python train.py --c config/usb_cv/freematch/freematch_cifar100_200_0.yaml
```

The provided configuration files are under [`config/usb_cv/freematch`](config/usb_cv/freematch). They contain the PiS-specific options, including `use_pis_ema`, `pis_ema_decay`, `warmup_ratio`, `reset_ratio`, and `warmup`.

### 3. Evaluation

After training, evaluate a saved checkpoint with the USB evaluation script, adapting the arguments and checkpoint path to your experiment:

```bash
python eval.py --dataset cifar100 --num_classes 100 --load_path /PATH/TO/CHECKPOINT
```

## Citation

If you use PiS, please cite the PiS paper:

```bibtex
@article{han2026decoupled,
  title   = {Decoupled Optimization for Teacher--Student Semi-Supervised Learning via a Pioneer Student},
  author  = {Han, Haorong and Yuan, Jidong and Wei, Chixuan and Sun, Yongqi},
  journal = {arXiv preprint arXiv:2610.09609},
  year    = {2026}
}
```

If you use the underlying USB codebase, please also cite USB:

```bibtex
@inproceedings{usb2022,
  title     = {USB: A Unified Semi-supervised Learning Benchmark for Classification},
  author    = {Wang, Yidong and Chen, Hao and Fan, Yue and others},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022},
  doi       = {10.48550/ARXIV.2208.07204}
}
```

## License

The inherited USB code and this derived work are distributed under the MIT License. See [`LICENSE.txt`](LICENSE.txt) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
