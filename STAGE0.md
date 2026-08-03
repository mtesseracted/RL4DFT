# Stage 0
This doc details searching for the exact functional of H2, the smallest system with correlation. 

The problem: 
We know T_s = T_W at N=2, and E_x = -1/2 E_H, so we only need the unknown E_c
For a given interactomic distance R and known solution density n_R(r), can can get
the KS potential from inversion:
  v_s = epsilon_HOMO + 1/2 \nabla^2 sqrt{n_R(r)} / sqrt{n_R(r)}.
  (we can remove the e_HOMO term as an additive constant)
This gives us the value and partial derivative of the E_xc at n_R, but only for that
reference density. We need a functional form that works even for trial densities so
that it is useable for minimization/SCF. It obviously should also work for different
distances, R. 

Papers exist that have the exact functional at the infinite dissociation limit, but you
cannot treat the non-inifinite limit as a perturbation as far as I can tell. The 

# Plan

Creat a bunch of training data for trial densities, i.e. E_xc and dE/dn, then train an
ML model and see if we can distill it down to an analytic form. The input to the NN
should be non-local, so it has to take either the density on a grid, or perhaps the 
coefficients of a fixed basis expansion. Need to figure out how we enforce the training
data derivative on the network, the easiest way would be to just incorporate it into
the loss. 


# References

The Hubbard Dimer: A density functional case study of a many-body problem
https://arxiv.org/abs/1502.02194
- Maps stretched H2 to the Hubbard dimer, which is analytically solvable


Exact Exchange-Correlation Functional for the Infinitely Stretched Hydrogen Molecule
https://arxiv.org/abs/1608.02553
- gives a functional for infinite stretched H2 with the assumption you can split the 
  density into rho_A and rho_B, densities on each atom