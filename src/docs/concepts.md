## Volumetric data in materials science

Density Functional Theory (DFT) has been one of the most successfully applied quantum theories with wide range of of applications
ranging from in silco materials discovery to simulation of quantum processes.
The fundamental tenet of DFT is that all of the energy of a quantum system is completely determined by the charge density alone, which is a much simpler object interms of computational complexity compared with the many-body electronic wave function that is typically needed to describe the system.
This make the charge density the central quantity in all of quantum chemistry research.

Modern DFT codes can solve for the charge density in finite and periodic systems, allowing to the generation and storage of millions of high-quality charge densities in databases like the Materials Project.
Such data sets are often useful for machine learning applications.
Since the charge density is scalar field in 3-D $\psi(x,y,z)$, one would assuming that 3D neural networks can be directly applied to this data.
But one major technical challenge remains: representation of the data so that results from different calculations can be directly compared.

This is especially problematic for periodic systems where the definition of the periodic lattice vectors determines the basic representation.
The data is given as a regular grid in fractional coordinates.

In order the compare charge densities from different calculations, we have to represent them all on a Cartesian grid.

## Storage of volumetric data on a regular grid

## Interpolation of Periodic data

## Re-griding
