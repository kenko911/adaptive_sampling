# Sampling corrections (2026-10-11)

- Public CV ranges, widths, and periodic boundaries use the same input unit: angstrom for distances and degrees for angles and torsions. The samplers keep bohr and radians internally.
- Public `get_pmf()` results are in kJ/mol; bias potentials and forces remain in atomic units internally. The OPES explore estimate scales by gamma divided by gamma minus one at the biased-density fixed point.
- CZAR grids represent bin centers. Integration of a sampled mean force starts at the first center with zero reference PMF and integrates between adjacent centers.
- Plain ABF's angular correction is the Cartesian divergence of its inverse-gradient field. It is evaluated with second derivatives of the angle or torsion CV. Distance and Cartesian-coordinate divergences retain their analytic forms.
- ASE temperature uses the existing molecular 3N minus 6 degree convention for more than two active atoms. The two-dimensional test engine uses two degrees of freedom.
- PLUMED is an optional dependency. Its tests are skipped when the module is unavailable, and direct use raises the import error.
- The arithmetic path CV tests use finite lambda, so their progress is a weighted average and their transverse value is a soft minimum of RMSD; neither equals the exact geometric path coordinate at all inputs.
