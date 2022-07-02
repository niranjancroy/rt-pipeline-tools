# rt-pipeline-tools

This repository contains a collection of scripts and tools for our Radiative Transfer analysis pipeline. These scripts can be used to create input files from hydrodynamic simulation snapshots, which can then be used as inputs to the Radmc-3D radiative transfer code (Dullemond et al. 2012, Astrophysics Source Code Library, record ascl:1202.015; http://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d). We also include scripts to analyse the resulting outputs from Radmc-3D.

The main tools in this repository are as follows:

## create_radmc3d_input.py 

This script takes a snapshot from a hydrodynamic simulation (currently we only support GIZMO snapshots) and projects the particles onto an AMR grid. It then writes out the data in a format that can be used as input files for the Radmc-3D. 

This script can be run as follows: 

`mpirun -np 16 python3 create_radmc3d_input.py create.param` 

You will need to provide the script with two text files, a parameter file (create.param, passed to it as an argument on the command line) and a species list file (species_list.txt, which is specified via the parameter file). Examples of these can be found in the `examples/create_radmc3d_input/` directory. 

### Parameter file 

The parameters that can be specified via the parameter file are described below. If a parameter is not specified in this file, it will revert to a default value. These default values are defined in the `parameters` dictionary at the beginning of `create_radmc3d_input.py`. 

* `infile` -- Input snapshot file
* `ChimeOutputFile` -- Output file from Chimes when abundances are not saved in snapshot #niranjan
* `Multifile` -- Equals 1 if snapshot has multiple files 
* `species_file` -- A text file listing the CHIMES species to include on the AMR grid 
* `SB99_dir` --  Directory containing the Starburst99 spectra 
* `n_cell_base` --  size of base grid (in x, y and z) 
* `max_part_per_cell` -- Max number of particles in a cell 
* `refinement_scheme` -- 0 - Refine on gas; 1 - refine on gas + stars 
* `box_size` -- Total size of the AMR grid, in code units 
* `centre_x` -- Position of the centre in the x direction 
* `centre_y` -- Position of the centre in the y direction 
* `centre_z` -- Position of the centre in the z direction 
* `dust_model` -- DC16 - De Cia+ 2016 depletion, constant - constant dust-to-metals ratio
* `dust_thresh` -- Exclude dust at gas temperatures above this threshold 
* `b_turb` -- Turbulent doppler width (km s^-1) 
* `include_H_level_pop` -- If set to 1, write out H level populations for Halpha and Hbeta 
* `unit_density` --  Convert code units -> g cm^-3 
* `unit_time_Myr` --  Convert code units -> Myr 
* `cosmo_flag` -- 0 - non-cosmological (Type 2 and 3 are also stars), 1 - cosmological (NOTE: not yet fully implemented).
* `use_eqm_abundances` -- 0 - use non-eq abundance array, 1 - use eqm abundances. 
* `adaptive_domain_decomposition` -- 0 - Equal domain sizes, 2 - adaptive domain sizes 
* `automatic_domain_decomposition` -- 0 - Manual, 1 - Automatic 
* `initial_domain_spacing` -- 0 - uniform spacing, 1 - concentrated in the centre
* `N_chunk_x` -- Number of chunks in the x-direction for manual decomposition 
* `N_chunk_y` -- Number of chunks in the y-direction for manual decomposition 
* `N_chunk_z` -- Number of chunks in the z-direction for manual decomposition 
* `include_stars` -- 0 - ignore stars; 1 - include stars 
* `N_age_bins` -- Specifies how many stellar age bins to include, starting from the youngest
* `N_age_bins_refine` -- How many stellar age bins to refine on, if refinement_scheme == 1
* `smooth_stars` -- Smooth stars if this flag is set to 1. 
* `star_hsml_file` -- File containing the stellar smoothing lengths. 
* `max_star_hsml` -- Max star hsml, in code units. If < 0, do not impose a maximum.
* `max_velocity` -- If positive, limit the maximum velocities of gas particles to be no greater than this value in x, y and z (in code units). In most cases you won't need this so you can set it to a negative value (which means this option is ignored). 
* `verbose_log_files` -- Set to 1 to write out log files from each MPI task. 

### Species list file 

Radmc-3D needs separate input files giving the densities of each ion and molecule species. You will need to include the ions/molecules for which you want to produce emission/absorption line spectra, and also the species that collisionally excite those ions/molecules. For example, if you want to look at emission lines from CIV, you would need the densities of CIV and electrons. 

The species list file defines which species to include when calculating densities in the AMR grid. You will need to include each species on a separate line, and for each one you need to give the name of the species (following the CHIMES naming convention), the atomic mass, and an integer flag that is either 0 (meaning the species is only used to collisionally excite another ion/molecule) or 1 (meaning that we want to calculate emission or absorption lines from this species and so the script will also compute ion-weighted temperature and velocity projections on the AMR grid for this species). So in the above example for CIV, the species list file would look like this: 

CIV   12.0       1  
elec  0.0005455  0

**Note:** if you are including molecular hydrogen to collisionally excite other species, you will need to specify ortho- and para-H2 separately like so (this is the only exception to the CHIMES naming convention): 

oH2   2.0  0  
pH2   2.0  0  

These are then calculated with a fixed ortho-to-para ratio of 3:1. If you also want to look at the emission lines from H2 you will need to additionally specify H2 as normal like so: 

H2    2.0  1  

### A few additional things to bear in mind 

#### Parallelisation 

The `create_radmc3d_input.py` script can be parallelised via MPI (using the mpi4py module). As noted above, you can run it with `mpirun`, where the `-np` argument is used to specify how many processes to use. 

The script parallelises the calculation by dividing the initial base grid between MPI tasks. Each task then takes its chunk of the base grid, refines the AMR cells, and projects the particles onto the refined AMR grid. There are a few parameters that control the behaviour of the domain decomposition, in particular: 

`automatic_domain_decomposition` -- If this option is used (i.e. this integer flag is set to 1), it automatically determines how many chunks to divide the domain into in the x, y and z directions, based on the number of MPI tasks. Basically, it is just finding combinations of three integers that multiply to give N, where N is the number of MPI tasks. If you set this parameter to 0 you can instead manually specify the number of chunks in each direction using `N_chunk_x`, `N_chunk_y`, `N_chunk_z`. However, the product of these three numbers must be equal to the number of MPI tasks (otherwise it will exit with an error message). For example, if you are running with 16 MPI tasks you could use: 

`N_chunk_x  4`  
`N_chunk_y  2`  
`N_chunk_z  2`

`adaptive_domain_decomposition` -- If this option is not used (i.e. this flag is set to 0), the chunks of the domain decomposition are set to be as close to equal in size as possible. For example, suppose you have 4, 2 and 2 chunks in the x, y and z directions as above, and your base AMR grid is 32x32x32 cells. Then it each chunk would consist of 8x16x16 cells. However, if you set this flag to 1 it will shift the boundaries of the chunks to try and balance the number of particles in each chunk. This can be useful if your simulation snapshot has a highly non-uniform distribution throughout the domain, as it will improve the work load balancing between tasks. 

`initial_domain_spacing` -- This is only used when `adaptive_domain_decomposition` is set to 1. If you set `initial_domain_spacing` to 0, it starts with equal sized chunks in the domain decomposition and then starts shifting the chunk boundaries from there. However, if you set `initial_domain_spacing` to 1, it starts with the chunks in the centre of the domain being 1 base cell across, with the outer chunks filling up the rest of the domain. I use this option a lot in my isolated disc galaxy simulations, where I know most particles are near the centre, as otherwise it can take a long time to find the optimal decomposition. 

#### Restart files 

The `create_radmc3d_input.py` script will produce restart files at certain points throughout the calculation (e.g. after calculating the densities on the AMR grid etc.). If it subsequently crashes, or if your job runs out of wall clock time, it can then restart from the last saved point. You can just run the script as normal, you don't need specify that it is a restart - it will automatically detect that the restart files are present and will read in the data from them instead of repeating that part of the calculation. Each MPI task produces its own restart files, so you will need to use the same number of MPI tasks (and all other parameters must also be kept the same). 

#### Cosmological simulations 

So far I have only used this script on non-cosmological simulations of idealised isolated disc galaxies. There is a parameter, `cosmo_flag`, that can be set to 1 to tell it that it is a cosmological simulation, however **this option has not yet been fully implemented!** Currently, this just tells it whether Type 2 and 3 particles should be considered as star particles along with the usual Type 4. It does not convert co-moving quantities to physical quantities, and I'm not sure if the calculation of the stellar ages is correct for a cosmological simulations (as I'm not sure how it stores the stellar formation times in this case). 

#### Chemical abundances 

This script assumes that the snapshot aleady contains the ion and molecule abundances for each gas particle, in the format of a standard CHIMES abundance array. You can either use non-equilibrium abundance (with `use_eqm_abundances` set to 0, meaning it looks for the abundance array in `PartType0/ChimesAbundances`), or you can use equilibrium abundances (with `use_eqm_abundances` set to 1, meaning it looks for the abundance array in `PartType0/EqmChimesAbundances`). If your snapshot does not already contain the CHIMES abundance array, you can use CHIMES-Driver to compute the equilibrium abundances of each gas particle (see the CHIMES web page at https://richings.bitbucket.io/chimes/home.html). 

## drive_radmc3d.py 

Python wrapper script used to run Radmc-3D over MPI. 

## compute_depletion.py 

Calculates dust depletion factors, based on Jenkins 2009, ApJ, 700, 1299 and De Cia et al. 2016, A&A, 596, 97. This is used by `create_radmc3d_input.py` to calculate the dust density of each particle. 

## compute_raga15.py 

Calculates the H-alpha and H-beta emissivities from recombination and collisional excitation based on Raga et al. 2015, RMxAA, 51, 231. This is used by `create_radmc3d_input.py` to calculate the hydrogen level population files (for recombination lines like the hydrogen Balmer lines we cannot just use Radmc-3D to compute the level populations, as it doesn't account for emission from recombinations. We instead create the level population files by hand so as to give the required emissivities for the given line). 

## common_data_files 

This directory contains various data files that will be needed by Radmc-3D. The atomic data in this directory were obtained from version 7.1 of the CHIANTI database (Dere et al. 1997, A&AS, 125, 149; Landi et al. 2013, ApJ, 763, 86; https://www.chiantidatabase.org) and from the LAMDA database (Schoier et al. 2005, A&A, 432, 369; https://home.strw.leidenuniv.nl/~moldata).
