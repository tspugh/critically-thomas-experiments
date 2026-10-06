# Critically Thomas: Growing Neural Cellular Automata

![A neural cellular automaton growing the Critically Thomas wordmark](result.gif)

This repository adapts the neural cellular automata (NCA) training method from
[Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) by
Mordvintsev, Randazzo, Niklasson, and Levin (Distill, 2020). Instead of the
original emoji target, the checked-in experiment trains a 16-channel cellular
automaton to grow the **Critically Thomas** wordmark from a single live cell.

The model sees each cell and its Sobel-filtered neighbourhood, applies a small
learned update network, and randomly updates roughly half of the cells at each
step. A pool of intermediate states exposes the model to different stages of
growth during training. `result.gif` is a generated output from this project;
it is included as a visual record, not as a benchmark.

## Run the experiment

The project uses Python 3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run critically-thomas-nca
```

The entry point runs `title-cellular-automata/nca.py`. On its first run it
trains from scratch, writes a model checkpoint to
`title-cellular-automata/nca.pt`, and renders `result.gif`. If that checkpoint
already exists, it is loaded and training continues before the GIF is
regenerated.

Training is compute-intensive. The script selects CUDA first, then Apple MPS,
and falls back to CPU. CUDA requires a working NVIDIA driver compatible with
the PyTorch build installed by uv. MPS requires a supported Apple-silicon Mac
and macOS/PyTorch combination. CPU execution needs no accelerator, but the
10,000-step default can take considerably longer. The repository does not
include a trained checkpoint.

## Repository contents

- `title-cellular-automata/nca.py` contains the checked-in 1× target training
  and rendering experiment.
- `title-cellular-automata/target.npy` is the RGBA training target;
  `target.png` and `target@11x.png` are viewable versions.
- `result.gif` is a checked-in generated result.
- `main.py` provides the installed command-line entry point.

## Attribution

The architecture, perception filters, stochastic cell updates, alive-mask
logic, and training-pool approach are adapted from the Distill article and its
accompanying implementation:

> Mordvintsev, A., Randazzo, E., Niklasson, E., & Levin, M. (2020).
> *Growing Neural Cellular Automata*. Distill.
> <https://doi.org/10.23915/distill.00023>

The target artwork and experiment-specific changes are part of this project.
No license is asserted here; consult the upstream article and implementation
for the terms that apply to their material.
