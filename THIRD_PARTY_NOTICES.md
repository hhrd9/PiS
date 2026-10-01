# Third-party notices

## Microsoft USB

This repository is derived from Microsoft's **Unified Semi-supervised Learning Benchmark (USB)**:

- Project: https://github.com/microsoft/Semi-supervised-learning
- License: MIT License
- Copyright: Copyright (c) Microsoft Corporation

Most of the framework, data-loading pipeline, utilities, model implementations, training infrastructure, and FreeMatch baseline are inherited from USB. The original copyright and license notices in the inherited source files have been retained. The PiS-specific work adds the Pioneer Student training path and related training configuration/logic.

The USB MIT License permits use, copying, modification, merger, publication, distribution, sublicensing, and sale, provided that the copyright notice and permission notice are included in copies or substantial portions of the software. The full applicable license text is in [`LICENSE.txt`](LICENSE.txt).

## PiS paper

The PiS method and the PiS-specific modifications are described in:

> Haorong Han, Jidong Yuan, Chixuan Wei, and Yongqi Sun. *Decoupled Optimization for Teacher–Student Semi-Supervised Learning via a Pioneer Student.*

This repository contains the PiS+FreeMatch release. The implementation additionally provides an EMA-based PiS overwrite option; see the implementation note in [`README.md`](README.md).
