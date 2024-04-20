# Generative model for the residue distance-constrained design of proteins

- We have a relatively easy-to-verify objective in silico for a protein, which
are ranges of distances between one or more residues or groups of residues. We
start out with single residue--single residue constraints though. Namely, we can
predict the structure with AF2 or ESM-fold, and then calculate these residue
distances. For example we may want residue 1 and 5 to be 8-10A apart.
- But we want some way of generating proteins that satisfy this constraint. This
is where the generative model comes in. We introduce a solution, first using an
RL agent and then using a GFlowNet, that solves this problem by iteratively
proposing solutions, which are then verified in silico.
- In our base case, we use ESM-fold with efficient options to allow quick
structure prediction. The first generative model will be an RL agent with a deep
Q network. The RL agent will only generate sequences with fixed length, and thus
will only accept constraints that are consistent with this fixed length.
Actually, we generally want to manually check that the constraint is consistent
with the sequence length. This can be done by considering the unfolded length of
the amino acid sequence using minimum and maximum size amino acids. The RL agent
will suggest a batch of sequences at each iteration. It will update residues in
the batch based on the rewards from the batch (the reward is associated with
each protein in the batch separately, of course).
