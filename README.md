# Qres: Protein design by RL from AI feedback

<div align="center">
  <img src="./qres/data/img/Screenshot%202024-11-09%20at%201.18.00%E2%80%AFPM.png" style="max-width: 700px;">
</div>

## Overview

ESMFold (which serves the same purpose as AlphaFold2) allows us to infer the
structure of a protein given its sequence, but the protein design problem, which
is roughly the inverse of this, is still quite difficult. A simplified
formulation of the design problem is to generate some sequence that, when
folded, possesses some provided structural characteristic(s), e.g. a binding
pocket with some roughly-specified shape. One approach to this particular
problem formulation is that a user provides a structural design criteria, such
as a pairwise distance between two residues, to an RL agent that generates a
sequence incrementally, with its reward signal computed based on how well the
ESMFold-predicted structure of the constructed sequence satisfies the criteria.

In my first experiment, I solved the easier problem of having the RL agent
generate stable protein structures with minimal edit distance from some initial
sequence, which is shown in the image above. I'm currently working on allowing
more complicated design criteria, such as pairwise distance constraints.

## Usage

This project uses [rye](https://github.com/astral-sh/rye) to manage Python
dependencies. To install the dependencies, run `rye sync` in the root directory.

To evaluate the most recent model on the stability task, run `rye run python qres/eval.py`.
You can also visualize the model design trajectories in `qres/visualize.ipynb`.

