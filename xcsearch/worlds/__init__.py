"""Worlds: everything interaction-specific lives under a world package.

STAGE2.md D9: the core (`ports`, `records`, `numerics`, `ks`, `driver`,
`store`, `ranking`, `evaluators`, `optimizers/`) is world-free; a world
package owns its grid, its interaction, its reference solver and its
dataset. `worlds/exp1d/` is the 1D exponential world of Stage 1 and the
only world Stage 2 ships.
"""
