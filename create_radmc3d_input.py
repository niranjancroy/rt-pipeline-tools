import numpy as np 
import tables 
import sys 
import struct 
import os

sys.path.append('/mnt/home/nroy/rt-pipeline-tools')

from mpi4py import MPI 
import read_chimes as rc 
import compute_raga15 as r15 
import compute_depletion as depl 
import sys 

sys.path.append('ext-lib/pfh_python')
import gadget as g
from gizmopy.load_fire_snap import load_fire_snap
from gizmopy.load_from_snapshot import load_from_snapshot




parameters = {"infile" : None,              # Input snapshot file 
              "species_file" : None,        # CHIMES species to include on the AMR grid 
              "SB99_dir" : None,            # Directory containing the Starburst99 spectra 
              "n_cell_base" : 16,           # size of base grid (in x, y and z) 
              "max_part_per_cell" : 8,      # Max number of particles in a cell 
              "refinement_scheme" : 0,      # 0 - Refine on gas; 1 - refine on gas + stars 
              "box_size" : 1.0,             # Total size of the AMR grid, in code units 
              "centre_x" : 0.0,             # Position of the centre in the x direction 
              "centre_y" : 0.0,             # Position of the centre in the y direction 
              "centre_z" : 0.0,             # Position of the centre in the z direction 
              "dust_model" : "DC16",        # DC16 - De Cia+ 2016 depletion, constant - constant dust-to-metals ratio
              "dust_thresh" : 1.0e6,        # Exclude dust at gas temperatures above this threshold 
              "b_turb" : 7.1,               # Turbulent doppler width (km s^-1) 
              "include_H_level_pop" : 1,    # If set to 1, write out H level populations for Halpha and Hbeta 
              "unit_density" : 6.771194847794874e-22,  # Convert code units -> g cm^-3 
              "unit_time_Myr" : 977.8943899409334,     # Convert code units -> Myr 
              "cosmo_flag" : 0,                        # 0 - non-cosmological (Type 2 and 3 are also stars), 
                                                       # 1 - cosmological (NOTE: not yet fully implemented) #niranjan: now implemented.
              "use_eqm_abundances" : 0,                # 0 - use non-eq abundance array, 1 - use eqm abundances. 
              "adaptive_domain_decomposition" : 0,     # 0 - Equal domain sizes, 2 - adaptive domain sizes 
              "automatic_domain_decomposition" : 1,    # 0 - Manual, 1 - Automatic 
              "initial_domain_spacing" : 0, # 0 - uniform spacing, 1 - concentrated in the centre
              "N_chunk_x" : 1,              # Number of chunks in the x-direction for manual decomposition 
              "N_chunk_y" : 1,              # Number of chunks in the y-direction for manual decomposition 
              "N_chunk_z" : 1,              # Number of chunks in the z-direction for manual decomposition 
              "include_stars" : 0,          # 0 - ignore stars; 1 - include stars 
              "N_age_bins" : 9,             # Specifies how many stellar age bins to include, starting from the youngest
              "N_age_bins_refine" : 8,      # How many stellar age bins to refine on, if refinement_scheme == 1
              "smooth_stars" : 1,           # Smooth stars if this flag is set to 1. 
              "star_hsml_file" : None,      # File containing the stellar smoothing lengths. 
              "max_star_hsml" : 0.1,        # Max star hsml, in code units. If < 0, do not impose a maximum.
              "max_velocity" : -1.0,        # If positive, limit the maximum velocities of gas particles to be no greater than this value in x, y and z (in code units).
              "verbose_log_files" : 0,      # Set to 1 to write out log files from each MPI task. 
              "ChimesOutputFile" : None,    #niranjan: when Chimes data is not abailable with the simulation
              "Multifiles" : 1 ,              #niranjan: In case of FIRE like simulations with multifile snapshots}
              "Rotate_to_faceon" : 0,     #niranjan: If the face on version of galaxy is wanted
              "Subtract_CoMvelocity" : 1 } #niranjan: If want to subtract velocity of the center of mass of the galaxy

# Defines the stellar age bins, as used with CHIMES. 
# Note that we have also included >1 Gyr as an 
# additional age bin. 
log_stellar_age_Myr_bin_max = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 10.0])

def read_parameters(infile): 
    fd = open(infile, "r") 
    
    for line in fd: 
        if len(line.strip()) < 1:
            continue 
        elif line.strip()[0] == "#":
            continue 
        else: 
            values = line.split() 
            if values[0] in parameters: 
                try: 
                    parameters[values[0]] = eval(values[1]) 
                except: 
                    parameters[values[0]] = values[1] 
            else: 
                raise KeyError("Parameter %s not recognised. Aborting" % (values[0], )) 

    fd.close() 

dust_to_gas_saturated = depl.compute_dust_to_gas_ratio(1000.0, 1) 
dust_to_gas_graphite = 2.34311e-3  # At solar metallicity (0.0129) and full depletion 
dust_to_gas_silicate = 3.96089e-3  # At solar metallicity (0.0129) and full depletion 

def cubic_spline_low(x, h): 
    return (8.0 / (np.pi * (h ** 3.0))) * (1.0 - (6.0 * (x ** 2.0)) + (6.0 * (x ** 3.0))) 

def cubic_spline_hi(x, h): 
    return (8.0 / (np.pi * (h ** 3.0))) * 2.0 * ((1.0 - x) ** 3.0) 

def cubic_spline(x, h): 
    if x < 0.5: 
        return cubic_spline_low(x, h)
    elif x < 1.0: 
        return cubic_spline_hi(x, h)
    else: 
        return 0.0 

# This routine takes an integer, decomposes it into 
# all combinations of three factors, then selects the 
# three with the minimum range. We use these to decompose 
# the base grid between MPI tasks. 
def decompose_3d_factors(x):
    factor_list= []
    range_list = []

    for i in range(1, x + 1):
        if x % i == 0:
            y = x // i
            for j in range(1, y + 1):
                if y % j == 0:
                    z = y // j
                    r = max([i, j, z]) - min([i, j ,z])
                    factor_list.append([i, j, z])
                    range_list.append(r)

    factor_list = np.array(factor_list)
    range_list = np.array(range_list)

    ind_sort = range_list.argsort()

    return factor_list[ind_sort][0] 

def write_task_log(task_no, message): 
    try: 
        log_file = "log_task_%d.txt" % (task_no, ) 
        fd = open(log_file, "a") 
        fd.write(message) 
        fd.write("\n")
        fd.close()
    except OSError: 
        print("OS error when writing task log file. Continuing.") 
        sys.stdout.flush() 

    return 

class node(): 
    def __init__(self, parent_node, N_species, emitter_flag_list, atomic_mass_list): 
        if parent_node == 0: 
            self.parent_node = parent_node 
            self.level = 1 
        else: 
            self.parent_node = parent_node 
            self.level = parent_node.level + 1 

        self.pos = np.zeros(3) 
        self.width = 0 
        self.N_part = 0 
        self.N_star_part = 0   # Only includes star particles in age bins that we refine the AMR grid on.
        self.daughter_nodes = 0 
        self.leaf = 0     # Set to 1 if this cell has no daughter nodes. 
        self.N_species = N_species 
        self.emitter_flag_list = emitter_flag_list 
        self.atomic_mass_list = atomic_mass_list 

        # Ion abundances 
        self.species = np.zeros(N_species, dtype = np.float64) 
 
        # Ion-weighted temperatures 
        self.temperature_species = np.zeros(N_species, dtype = np.float64) 
 
        # Ion-weighted nH
        self.nH_species = np.zeros(N_species, dtype = np.float64) 

        # Ion-weighted velocities 
        self.velocity_species = np.zeros((N_species, 3), dtype = np.float64) 

        # Dust density 
        self.rho_dust = 0.0 

        # Stellar densities 
        self.rho_star = np.zeros(parameters["N_age_bins"], dtype = np.float64) 

        # H level populations, for recombination lines 
        # Using Osterbrock & Ferland (2006) 
        self.H_level_pop_OF06 = np.zeros((2, 3), dtype = np.float64) 

        # Using Raga et al. (2015) 
        self.HII_level_pop_R15 = np.zeros((2, 3), dtype = np.float64) 
        self.HI_level_pop_R15 = np.zeros((2, 3), dtype = np.float64) 

    def write_restart_node(self, fd): 
        buf = struct.pack("3i", self.leaf, self.N_part, self.N_star_part) 
        fd.write(buf) 
        
        buf = struct.pack("4d", self.pos[0], self.pos[1], self.pos[2], self.width) 
        fd.write(buf) 

        if self.leaf == 0: 
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        self.daughter_nodes[i, j, k].write_restart_node(fd) 
                        
        return 

    def read_restart_node(self, fd): 
        buf = fd.read(3 * 4) 
        data = struct.unpack("3i", buf) 
        self.leaf = data[0] 
        self.N_part = data[1] 
        self.N_star_part = data[2] 
    
        buf = fd.read(4 * 8) 
        data = struct.unpack("4d", buf) 
        self.pos[0] = data[0] 
        self.pos[1] = data[1] 
        self.pos[2] = data[2] 
        self.width = data[3] 

        if self.leaf == 0: 
            self.daughter_nodes = np.ndarray((2, 2, 2), dtype = np.object) 
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        self.daughter_nodes[i, j, k] = node(self, self.N_species, self.emitter_flag_list, self.atomic_mass_list) 
                        self.daughter_nodes[i, j, k].read_restart_node(fd) 
        
        return 

    def write_restart_densities(self, fd): 
        if self.leaf == 1: 
            for idx in range(self.N_species): 
                buf = struct.pack("6f", self.species[idx], 
                                  self.temperature_species[idx], 
                                  self.nH_species[idx], 
                                  self.velocity_species[idx, 0], 
                                  self.velocity_species[idx, 1], 
                                  self.velocity_species[idx, 2])
                fd.write(buf) 
            
            buf = struct.pack("f", self.rho_dust) 
            fd.write(buf) 

            for idx in range(parameters["N_age_bins"]): 
                buf = struct.pack("f", self.rho_star[idx]) 
                fd.write(buf) 
            
            for idx_x in range(2): 
                for idx_y in range(3): 
                    buf = struct.pack("3f", self.H_level_pop_OF06[idx_x, idx_y], self.HII_level_pop_R15[idx_x, idx_y], self.HI_level_pop_R15[idx_x, idx_y]) 
                    fd.write(buf) 
        else:
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        self.daughter_nodes[i, j, k].write_restart_densities(fd) 

        return 


    def read_restart_densities(self, fd): 
        if self.leaf == 1: 
            for idx in range(self.N_species): 
                buf = fd.read(6 * 4) 
                data = struct.unpack("6f", buf) 
                self.species[idx] = data[0] 
                self.temperature_species[idx] = data[1] 
                self.nH_species[idx] = data[2] 
                self.velocity_species[idx, 0] = data[3] 
                self.velocity_species[idx, 1] = data[4] 
                self.velocity_species[idx, 2] = data[5] 

            buf = fd.read(4) 
            data = struct.unpack("f", buf) 
            self.rho_dust = data[0] 
            
            for idx in range(parameters["N_age_bins"]): 
                buf = fd.read(4) 
                data = struct.unpack("f", buf) 
                self.rho_star[idx] = data[0] 
            
            for idx_x in range(2): 
                for idx_y in range(3): 
                    buf = fd.read(3 * 4) 
                    data = struct.unpack("3f", buf) 
                    self.H_level_pop_OF06[idx_x, idx_y] = data[0]
                    self.HII_level_pop_R15[idx_x, idx_y] = data[1] 
                    self.HI_level_pop_R15[idx_x, idx_y] = data[2] 
        else: 
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        self.daughter_nodes[i, j, k].read_restart_densities(fd) 

        return 
            
    def find_particles(self, particle_coords, particle_star_coords): 
        # This routine finds the indices of all particles in 
        # particle_coords that are within this cell 
        x_min = self.pos[0] - (self.width / 2.0) 
        x_max = self.pos[0] + (self.width / 2.0) 
        y_min = self.pos[1] - (self.width / 2.0) 
        y_max = self.pos[1] + (self.width / 2.0) 
        z_min = self.pos[2] - (self.width / 2.0) 
        z_max = self.pos[2] + (self.width / 2.0) 

        part_ind = ((particle_coords[:, 0] > x_min) & (particle_coords[:, 0] < x_max) & 
                    (particle_coords[:, 1] > y_min) & (particle_coords[:, 1] < y_max) & 
                    (particle_coords[:, 2] > z_min) & (particle_coords[:, 2] < z_max)) 
        self.N_part = len(particle_coords[part_ind, 0]) 

        if parameters["refinement_scheme"] == 1:
            part_ind_star = ((particle_star_coords[:, 0] > x_min) & (particle_star_coords[:, 0] < x_max) & 
                             (particle_star_coords[:, 1] > y_min) & (particle_star_coords[:, 1] < y_max) & 
                             (particle_star_coords[:, 2] > z_min) & (particle_star_coords[:, 2] < z_max)) 
            self.N_star_part = len(particle_star_coords[part_ind_star, 0]) 
        else: 
            part_ind_star = None 
        
        return part_ind, part_ind_star

    def compute_cell_densities(self, 
                               particle_coords, 
                               particle_hsml, 
                               particle_mass, 
                               particle_mass_species, 
                               particle_temperature, 
                               particle_velocity, 
                               particle_metallicity, 
                               particle_nH, 
                               sum_wk): 
        # Performs an SPH interpolation of gas densities onto the 
        # centre of this cell. 
        if self.leaf == 1: 
            relative_pos = self.pos - particle_coords
                        
            ind_smooth = ((relative_pos[:, 0] > -particle_hsml) & 
                          (relative_pos[:, 0] < particle_hsml) & 
                          (relative_pos[:, 1] > -particle_hsml) & 
                          (relative_pos[:, 1] < particle_hsml) & 
                          (relative_pos[:, 2] > -particle_hsml) & 
                          (relative_pos[:, 2] < particle_hsml)) 
            
            radii = np.sqrt(((relative_pos[ind_smooth, 0]) ** 2.0) + 
                            ((relative_pos[ind_smooth, 1]) ** 2.0) + 
                            ((relative_pos[ind_smooth, 2]) ** 2.0)) 

            hsml_smooth = particle_hsml[ind_smooth]

            radii /= hsml_smooth 
            sum_wk_smooth = sum_wk[ind_smooth] 
            mass_smooth = particle_mass[ind_smooth] 
            species_smooth = particle_mass_species[ind_smooth] 
            T_smooth = particle_temperature[ind_smooth] 
            vel_smooth = particle_velocity[ind_smooth] 
            Z_smooth = particle_metallicity[ind_smooth] 
            nH_smooth = particle_nH[ind_smooth] 
            
            ind_r = (radii < 1.0) 
            radii_r = radii[ind_r]
            hsml_r = hsml_smooth[ind_r]
            sum_wk_r = sum_wk_smooth[ind_r]
            mass_r = mass_smooth[ind_r]
            species_r = species_smooth[ind_r]
            T_r = T_smooth[ind_r]
            vel_r = vel_smooth[ind_r]
            Z_r = Z_smooth[ind_r]
            nH_r = nH_smooth[ind_r]

            wk = np.zeros(len(mass_r), dtype = np.float64) 
            ind_low = (radii_r < 0.5) 
            wk[ind_low] += cubic_spline_low(radii_r[ind_low], hsml_r[ind_low]) 
            ind_hi = (radii_r >= 0.5) 
            wk[ind_hi] += cubic_spline_hi(radii_r[ind_hi], hsml_r[ind_hi]) 

            self.species += np.sum(species_r.transpose() * wk / sum_wk_r, axis = 1)   # Msol 
            self.temperature_species += np.sum(species_r.transpose() * T_r * wk / sum_wk_r, axis = 1)  
            self.nH_species += np.sum(species_r.transpose() * nH_r * wk / sum_wk_r, axis = 1)

            self.velocity_species[:, 0] +=  np.sum(species_r.transpose() * vel_r[:, 0] * wk / sum_wk_r, axis = 1)
            self.velocity_species[:, 1] +=  np.sum(species_r.transpose() * vel_r[:, 1] * wk / sum_wk_r, axis = 1)
            self.velocity_species[:, 2] +=  np.sum(species_r.transpose() * vel_r[:, 2] * wk / sum_wk_r, axis = 1)

            for i in range(len(mass_r)): 
                if parameters["dust_model"] == "DC16": 
                    if T_r[i] < parameters["dust_thresh"]: 
                        self.rho_dust += (depl.compute_dust_to_gas_ratio(nH_r[i], 1) / dust_to_gas_saturated) * (Z_r[i] / 0.0129) * mass_r[i] * wk[i] / sum_wk_r[i] # Msol 
                elif parameters["dust_model"] == "constant": 
                    self.rho_dust += (Z_r[i] / 0.0129) * mass_r[i] * wk[i] / sum_wk_r[i] # Msol 

            for j in range(self.N_species): 
                if self.species[j] > 0.0: 
                    self.temperature_species[j] /= self.species[j] 
                    self.nH_species[j] /= self.species[j] 
                    self.velocity_species[j, :] /= self.species[j] 

            # Convert from masses to densities. 
            self.species /= (self.width ** 3.0) # Msol kpc^-3 
            self.rho_dust /= (self.width ** 3.0) # Msol kpc^-3 
        else: 
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        relative_pos = self.daughter_nodes[i, j, k].pos - particle_coords
                        overlap_size = particle_hsml + (self.daughter_nodes[i, j, k].width / 2.0) 
                        
                        ind_smooth = ((relative_pos[:, 0] > -overlap_size) & 
                                      (relative_pos[:, 0] < overlap_size) & 
                                      (relative_pos[:, 1] > -overlap_size) & 
                                      (relative_pos[:, 1] < overlap_size) & 
                                      (relative_pos[:, 2] > -overlap_size) & 
                                      (relative_pos[:, 2] < overlap_size)) 
                        
                        self.daughter_nodes[i, j, k].compute_cell_densities(particle_coords[ind_smooth], 
                                                                            particle_hsml[ind_smooth], 
                                                                            particle_mass[ind_smooth], 
                                                                            particle_mass_species[ind_smooth], 
                                                                            particle_temperature[ind_smooth], 
                                                                            particle_velocity[ind_smooth], 
                                                                            particle_metallicity[ind_smooth], 
                                                                            particle_nH[ind_smooth], 
                                                                            sum_wk[ind_smooth])

    def compute_cell_stellar_densities(self, particle_star_coords, particle_star_mass, age_index): 
        # Places star particles into the cell as point particles (i.e. no SPH smoothing) 
        if self.leaf == 1: 
            ind_cell = ((particle_star_coords[:, 0] > (self.pos[0] - (self.width / 2.0))) & 
                        (particle_star_coords[:, 0] < (self.pos[0] + (self.width / 2.0))) & 
                        (particle_star_coords[:, 1] > (self.pos[1] - (self.width / 2.0))) & 
                        (particle_star_coords[:, 1] < (self.pos[1] + (self.width / 2.0))) & 
                        (particle_star_coords[:, 2] > (self.pos[2] - (self.width / 2.0))) & 
                        (particle_star_coords[:, 2] < (self.pos[2] + (self.width / 2.0)))) 
            self.rho_star[age_index] += sum(particle_star_mass[ind_cell]) / (self.width ** 3.0)  # Msol kpc^-3 
        else: 
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        self.daughter_nodes[i, j, k].compute_cell_stellar_densities(particle_star_coords, particle_star_mass, age_index) 

    def compute_cell_smoothed_stellar_densities(self, 
                                                particle_star_coords, 
                                                particle_star_hsml, 
                                                particle_star_mass, 
                                                sum_wk_star, 
                                                age_index): 
        # Performs an SPH interpolation of stellar densities onto the 
        # centre of this cell. 
        if self.leaf == 1: 
            # We only pass particles within 
            # the smoothing kernel to this routine. 
            relative_pos = self.pos - particle_star_coords
                        
            ind_smooth = ((relative_pos[:, 0] > -particle_star_hsml) & 
                          (relative_pos[:, 0] < particle_star_hsml) & 
                          (relative_pos[:, 1] > -particle_star_hsml) & 
                          (relative_pos[:, 1] < particle_star_hsml) & 
                          (relative_pos[:, 2] > -particle_star_hsml) & 
                          (relative_pos[:, 2] < particle_star_hsml)) 
                        
            radii = np.sqrt(((relative_pos[ind_smooth, 0]) ** 2.0) + 
                            ((relative_pos[ind_smooth, 1]) ** 2.0) + 
                            ((relative_pos[ind_smooth, 2]) ** 2.0)) 

            hsml_smooth = particle_star_hsml[ind_smooth]
            sum_wk_smooth = sum_wk_star[ind_smooth] 
            mass_smooth = particle_star_mass[ind_smooth] 

            radii /= hsml_smooth 
            
            ind_r = (radii < 1.0)
            radii_r = radii[ind_r]
            hsml_r = hsml_smooth[ind_r]
            sum_wk_r = sum_wk_smooth[ind_r]
            mass_r = mass_smooth[ind_r]

            wk = np.zeros(len(mass_r), dtype = np.float64) 
            ind_low = (radii_r < 0.5) 
            wk[ind_low] += cubic_spline_low(radii_r[ind_low], hsml_r[ind_low]) 
            ind_hi = (radii_r >= 0.5) 
            wk[ind_hi] += cubic_spline_hi(radii_r[ind_hi], hsml_r[ind_hi]) 

            self.rho_star[age_index] += np.sum(mass_r * wk / sum_wk_r)   # Msol 

            # Convert from mass to density. 
            self.rho_star[age_index] /= (self.width ** 3.0) # Msol kpc^-3 
        else: 
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        relative_pos = self.daughter_nodes[i, j, k].pos - particle_star_coords
                        overlap_size = particle_star_hsml + (self.daughter_nodes[i, j, k].width / 2.0) 
            
                        ind_smooth = ((relative_pos[:, 0] > -overlap_size) & 
                                      (relative_pos[:, 0] < overlap_size) & 
                                      (relative_pos[:, 1] > -overlap_size) & 
                                      (relative_pos[:, 1] < overlap_size) & 
                                      (relative_pos[:, 2] > -overlap_size) & 
                                      (relative_pos[:, 2] < overlap_size)) 

                        self.daughter_nodes[i, j, k].compute_cell_smoothed_stellar_densities(particle_star_coords[ind_smooth], 
                                                                                             particle_star_hsml[ind_smooth], 
                                                                                             particle_star_mass[ind_smooth], 
                                                                                             sum_wk_star[ind_smooth], age_index)

    def compute_kernel_weights(self, particle_coords, particle_hsml): 
        # Computes the kernel weights of all particles that overlap 
        # with this cell. These will be used to compute sum_wk_i. 
        if self.leaf == 1: 
            relative_pos = self.pos - particle_coords

            ind_smooth = ((relative_pos[:, 0] > -particle_hsml) & 
                          (relative_pos[:, 0] < particle_hsml) & 
                          (relative_pos[:, 1] > -particle_hsml) & 
                          (relative_pos[:, 1] < particle_hsml) & 
                          (relative_pos[:, 2] > -particle_hsml) & 
                          (relative_pos[:, 2] < particle_hsml)) 
            
            radii = np.sqrt(((relative_pos[ind_smooth, 0]) ** 2.0) + 
                            ((relative_pos[ind_smooth, 1]) ** 2.0) + 
                            ((relative_pos[ind_smooth, 2]) ** 2.0)) 

            hsml_smooth = particle_hsml[ind_smooth] 
            radii /= hsml_smooth
            
            wk_outputs = np.zeros(len(particle_coords), dtype = np.float64) 
            wk_smooth = np.zeros(len(radii), dtype = np.float64) 

            ind_low = (radii < 0.5) 
            wk_smooth[ind_low] += cubic_spline_low(radii[ind_low], hsml_smooth[ind_low]) 

            ind_hi = ((radii >= 0.5) & (radii < 1.0)) 
            wk_smooth[ind_hi] += cubic_spline_hi(radii[ind_hi], hsml_smooth[ind_hi]) 

            wk_outputs[ind_smooth] += wk_smooth 
        
            return wk_outputs 
        else: 
            wk_outputs = np.zeros(len(particle_coords), dtype = np.float64) 
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        relative_pos = self.daughter_nodes[i, j, k].pos - particle_coords
                        overlap_size = particle_hsml + (self.daughter_nodes[i, j, k].width / 2.0) 
                        
                        ind_smooth = ((relative_pos[:, 0] > -overlap_size) & 
                                      (relative_pos[:, 0] < overlap_size) & 
                                      (relative_pos[:, 1] > -overlap_size) & 
                                      (relative_pos[:, 1] < overlap_size) & 
                                      (relative_pos[:, 2] > -overlap_size) & 
                                      (relative_pos[:, 2] < overlap_size)) 
                        wk_outputs[ind_smooth] += self.daughter_nodes[i, j, k].compute_kernel_weights(particle_coords[ind_smooth], particle_hsml[ind_smooth]) 
            return wk_outputs 
        
    def split_cells(self, particle_coords, particle_star_coords): 
        # If the number of particles in this cell exceeds the 
        # threshold, it is divided into 8 daughter cells. This 
        # routine is then recursively called on them as well, 
        # to continue splitting the cells until the criterion 
        # is met. 
        if (self.N_part > parameters["max_part_per_cell"]) or (self.N_star_part > parameters["max_part_per_cell"] and parameters["refinement_scheme"] == 1): 
            self.daughter_nodes = np.ndarray((2, 2, 2), dtype = np.object) 
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        self.daughter_nodes[i, j, k] = node(self, self.N_species, self.emitter_flag_list, self.atomic_mass_list) 
                        self.daughter_nodes[i, j, k].width = self.width / 2.0 
                        self.daughter_nodes[i, j, k].pos[0] = self.pos[0] - (self.width / 4.0) + (i * self.width / 2.0) 
                        self.daughter_nodes[i, j, k].pos[1] = self.pos[1] - (self.width / 4.0) + (j * self.width / 2.0) 
                        self.daughter_nodes[i, j, k].pos[2] = self.pos[2] - (self.width / 4.0) + (k * self.width / 2.0) 
                        part_ind, part_ind_star = self.daughter_nodes[i, j, k].find_particles(particle_coords, particle_star_coords) 
                        self.daughter_nodes[i, j, k].split_cells(particle_coords[part_ind, :], particle_star_coords[part_ind_star, :]) 
                        del part_ind 
                        del part_ind_star
        else: 
            self.leaf = 1 

    def determine_levels(self): 
        # Walks through the tree and adds up leaves and 
        # branches, and determines the max level 
        if self.leaf == 1: 
            return self.level, 0, 1    
        else: 
            max_level = 1 
            n_branch = 1 
            n_leaf = 0 
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        next_level, next_branch, next_leaf = self.daughter_nodes[i, j, k].determine_levels() 
                        n_branch += next_branch 
                        n_leaf += next_leaf 
                        if next_level > max_level: 
                            max_level = next_level 
            return max_level, n_branch, n_leaf 

    def compute_total_species_mass(self, species_index): 
        volume = self.width ** 3.0  # kpc^3 
        if self.leaf == 1: 
            return self.species[species_index] * volume  # Msol 
        else: 
            branch_species = 0.0 
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        branch_species += self.daughter_nodes[i, j, k].compute_total_species_mass(species_index) 
            return branch_species 

    def compute_total_stellar_mass(self, age_index): 
        volume = self.width ** 3.0  # kpc^3 
        if self.leaf == 1: 
            return self.rho_star[age_index] * volume  # Msol 
        else: 
            branch_mass = 0.0 
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        branch_mass += self.daughter_nodes[i, j, k].compute_total_stellar_mass(age_index)
            return branch_mass



    def compute_H_level_populations(self, elec_index, HI_index, HII_index): 
        # These aren't the real level populations. 
        # Instead, we create a 3-level atom, and 
        # set the n=3 level population to give the 
        # required H-alpha or H-beta emissivity. 
        # We then put the rest of it in n=1. 
        # NOTE: The RADMC-3D levelpop input files 
        # need the level populations in units of 
        # cm^-3, i.e. these are the densities of 
        # hydrogen in the given level, and not 
        # relative fractions. 

        if self.leaf == 1: 
            ne = self.species[elec_index]   # Msol kpc^-3 
            nHI = self.species[HI_index]  # Msol kpc^-3 
            nHII = self.species[HII_index]  # Msol kpc^-3 
            ne *= 6.77119485e-32 / (self.atomic_mass_list[elec_index] * 1.6726218e-24)  # Converts to cm^-3 
            nHI *= 6.77119485e-32 / (self.atomic_mass_list[HI_index] * 1.6726218e-24) # Converts to cm^-3 
            nHII *= 6.77119485e-32 / (self.atomic_mass_list[HII_index] * 1.6726218e-24) # Converts to cm^-3 
            T_HI = self.temperature_species[HI_index] 
            T_HII = self.temperature_species[HII_index] 

            h_nu_alpha = 3.0301602666779086e-12   # erg
            h_nu_beta = 4.090713693193314e-12     # erg

            A_alpha = 6.46e7  # s^-1 
            A_beta = 2.06e7   # s^-1 
            
            # First, using Osterbrock & Ferland (2006).

            # H-alpha 
            if T_HII > 0.0: 
                rec_alpha_OF06 = 7.86e-14 * ((T_HII / 1.0e4) ** -1.0)   # cm^3 s^-1 
            else: 
                rec_alpha_OF06 = 0.0 

            self.H_level_pop_OF06[0, 2] = nHII * ne * rec_alpha_OF06 / A_alpha 
            self.H_level_pop_OF06[0, 0] = nHII - self.H_level_pop_OF06[0, 2] 

            # H-beta 
            if T_HII > 0.0: 
                rec_beta_OF06 = 3.67e-14 * ((T_HII / 1.0e4) ** -0.91)    # cm^3 s^-1 
            else: 
                rec_beta_OF06 = 0.0 

            self.H_level_pop_OF06[1, 2] = nHII * ne * rec_beta_OF06 / A_beta 
            self.H_level_pop_OF06[1, 0] = nHII - self.H_level_pop_OF06[1, 2] 

            # Raga et al. (2015): recombination of HII  
            if T_HII > 0.0: 
                HII_rec_alpha_R15 = r15.raga15_Halpha_rec_caseB(T_HII) / h_nu_alpha  # cm^3 s^-1
                HII_rec_beta_R15 = r15.raga15_Hbeta_rec_caseB(T_HII) / h_nu_beta     # cm^3 s^-1
            else: 
                HII_rec_alpha_R15 = 0.0 
                HII_rec_beta_R15 = 0.0 

            self.HII_level_pop_R15[0, 2] = ne * nHII * HII_rec_alpha_R15 / A_alpha
            self.HII_level_pop_R15[0, 0] = nHII - self.HII_level_pop_R15[0, 2] 

            self.HII_level_pop_R15[1, 2] = ne * nHII * HII_rec_beta_R15 / A_beta
            self.HII_level_pop_R15[1, 0] = nHII - self.HII_level_pop_R15[1, 2] 


            # Raga et al. (2015): collisional excitation of HI
            if T_HI > 0.0: 
                HI_col_alpha_R15 = r15.raga15_Halpha_col_caseB(T_HI) / h_nu_alpha  # cm^3 s^-1
                HI_col_beta_R15 = r15.raga15_Hbeta_col_caseB(T_HI) / h_nu_beta     # cm^3 s^-1
            else: 
                HI_col_alpha_R15 = 0.0 
                HI_col_beta_R15 = 0.0 

            self.HI_level_pop_R15[0, 2] = ne * nHI * HI_col_alpha_R15 / A_alpha
            self.HI_level_pop_R15[0, 0] = nHI - self.HI_level_pop_R15[0, 2] 

            self.HI_level_pop_R15[1, 2] = ne * nHI * HI_col_beta_R15 / A_beta
            self.HI_level_pop_R15[1, 0] = nHI - self.HI_level_pop_R15[1, 2] 
        else: 
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        self.daughter_nodes[i, j, k].compute_H_level_populations(elec_index, HI_index, HII_index) 
            
    def walk_cells(self, species_file_list, T_file_list, nH_file_list, vel_file_list, turb_file, vol_file): 
        if self.leaf == 1: 
            emitter_index = 0 

            for i in range(self.N_species): 
                buf = struct.pack("d", self.species[i] * 6.77119485e-32 / (self.atomic_mass_list[i] * 1.6726218e-24)) # Converts to mol cm^-3 
                species_file_list[i].write(buf) 
                species_file_list[i].flush()

                if self.emitter_flag_list[i] == 1: 
                    # If an ion is absent from a cell, the corresponding T, nH and velocity will be zero. 
                    # We should set a minimum to the temperature, to avoid any problems. 
                    buf = struct.pack("d", max(self.temperature_species[i], 10.0)) # K 
                    T_file_list[emitter_index].write(buf) 
                    T_file_list[emitter_index].flush()

                    buf = struct.pack("d", self.nH_species[i])  # cm^-3 
                    nH_file_list[emitter_index].write(buf) 
                    nH_file_list[emitter_index].flush() 
                    
                    buf = struct.pack("3d", self.velocity_species[i, 0] * 1.0e5, self.velocity_species[i, 1] * 1.0e5, self.velocity_species[i, 2] * 1.0e5)  # cm s^-1 
                    vel_file_list[emitter_index].write(buf) 
                    vel_file_list[emitter_index].flush() 

                    emitter_index += 1 

            # Assume a constant microturbulence width b everywhere. 
            buf = struct.pack("d", parameters["b_turb"] * 1.0e5)   # cm s^-1 
            turb_file.write(buf)
            turb_file.flush()

            vol_file.write("%.6e \n" % (self.width ** 3.0, ))  # kpc^3 
            vol_file.flush() 
        else: 
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        self.daughter_nodes[i, j, k].walk_cells(species_file_list, T_file_list, nH_file_list, vel_file_list, turb_file, vol_file) 

    def walk_cells_dust(self, dust_file, dust_T_file, dust_species): 
        if self.leaf == 1: 
            if dust_species == 0: 
                # Graphite density 
                buf = struct.pack("d", self.rho_dust * dust_to_gas_graphite * 6.77119485e-32) # Converts to g cm^-3  
            elif dust_species == 1: 
                # Silicate density 
                buf = struct.pack("d", self.rho_dust * dust_to_gas_silicate * 6.77119485e-32) # Converts to g cm^-3  
            else: 
                print("ERROR: dust_species %d not recognised." % (dust_species, )) 
                sys.stdout.flush() 
                return 

            dust_file.write(buf)
            dust_T_file.write("0.0 \n") 
        else: 
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        self.daughter_nodes[i, j, k].walk_cells_dust(dust_file, dust_T_file, dust_species) 

    def walk_cells_stars(self, star_file, age_index): 
        if self.leaf == 1: 
            buf = struct.pack("d", self.rho_star[age_index] * 6.77119485e-32) # Converts to g cm^-3 
            star_file.write(buf)
        else: 
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        self.daughter_nodes[i, j, k].walk_cells_stars(star_file, age_index) 
                        
    def walk_cells_H_level_pop(self, pop_file_OF06, pop_file_R15_HII, pop_file_R15_HI, line_index): 
        if self.leaf == 1: 
            line = "%.6e %.6e %.6e \n" % (self.H_level_pop_OF06[line_index, 0], self.H_level_pop_OF06[line_index, 1], self.H_level_pop_OF06[line_index, 2]) 
            pop_file_OF06.write(line) 

            line = "%.6e %.6e %.6e \n" % (self.HII_level_pop_R15[line_index, 0], self.HII_level_pop_R15[line_index, 1], self.HII_level_pop_R15[line_index, 2]) 
            pop_file_R15_HII.write(line) 

            line = "%.6e %.6e %.6e \n" % (self.HI_level_pop_R15[line_index, 0], self.HI_level_pop_R15[line_index, 1], self.HI_level_pop_R15[line_index, 2]) 
            pop_file_R15_HI.write(line) 
        else: 
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        self.daughter_nodes[i, j, k].walk_cells_H_level_pop(pop_file_OF06, pop_file_R15_HII, pop_file_R15_HI, line_index) 

    def determine_amr_grid(self, amr_file): 
        if self.leaf == 1: 
            amr_file.write("0 \n") 
        else: 
            amr_file.write("1 \n") 
            for k in range(2): 
                for j in range(2): 
                    for i in range(2): 
                        k_index = k 
                        self.daughter_nodes[i, j, k_index].determine_amr_grid(amr_file) 
                        
def write_headers(amr_file, 
                  species_file_list, 
                  T_file_list, 
                  nH_file_list, 
                  vel_file_list, 
                  turb_file, 
                  dust_file, 
                  dust_T_file, 
                  star_file, 
                  grid_array_arg, 
                  grid_width, 
                  level_max, 
                  branch_max, 
                  leaf_max): 
    iformat = 1 
    precis = 8 
    nrcells = leaf_max 
    nrspec = 2 
    nr_star_spec = parameters["N_age_bins"] 

    amr_file.write(u"1 \n")  # iformat 
    amr_file.write(u"1 \n")  # grid style 
    amr_file.write(u"0 \n")  # coord system 
    amr_file.write(u"0 \n")  # grid info 
    amr_file.write(u"1 1 1 \n")  # incl x, y, z 
    output_line = u"%d %d %d \n" % (parameters["n_cell_base"], parameters["n_cell_base"], parameters["n_cell_base"]) 
    amr_file.write(output_line) 
    leaf_branch_max = leaf_max + branch_max 
    output_line = u"%d %d %d \n" % (level_max, leaf_branch_max, leaf_branch_max) 
    amr_file.write(output_line) 

    grid_array = grid_array_arg.copy() 

    grid_array -= grid_width / 2.0 
    grid_array -= grid_array[0] 
    grid_array *= 3.086e21         # cm 
    grid_width *= 3.086e21         # cm 
    for i in grid_array: 
        output_line = u"%.6e " % (i, ) 
        amr_file.write(output_line) 
    output_line = u"%.6e \n" % (grid_array[-1] + grid_width, ) 
    amr_file.write(output_line) 

    for i in grid_array: 
        output_line = u"%.6e " % (i, ) 
        amr_file.write(output_line) 
    output_line = u"%.6e \n" % (grid_array[-1] + grid_width, ) 
    amr_file.write(output_line) 

    for i in grid_array: 
        output_line = u"%.6e " % (i, ) 
        amr_file.write(output_line) 
    output_line = u"%.6e \n" % (grid_array[-1] + grid_width, ) 
    amr_file.write(output_line) 

    for my_file in species_file_list: 
        buf = struct.pack("3l", iformat, precis, nrcells) 
        my_file.write(buf) 
    
    for my_file in T_file_list: 
        buf = struct.pack("3l", iformat, precis, nrcells) 
        my_file.write(buf) 
    
    for my_file in nH_file_list: 
        buf = struct.pack("3l", iformat, precis, nrcells) 
        my_file.write(buf) 
    
    for my_file in vel_file_list: 
        buf = struct.pack("3l", iformat, precis, nrcells) 
        my_file.write(buf) 

    buf = struct.pack("3l", iformat, precis, nrcells)
    turb_file.write(buf)
    
    buf = struct.pack("4l", iformat, precis, nrcells, nrspec)
    dust_file.write(buf)

    dust_T_file.write("1 \n") 
    dust_T_file.write("%d \n" % (nrcells, )) 
    dust_T_file.write("%d \n" % (nrspec, )) 
    
    if parameters["include_stars"] == 1: 
        buf = struct.pack("4l", iformat, precis, nrcells, nr_star_spec)
        star_file.write(buf)

def write_restart_domain(task_id, data_int): 
    filename = "restart_domain_%d" % (task_id, )
    fd = open(filename, "wb") 
    
    for arr in data_int: 
        for idx in range(len(arr)): 
            buf = struct.pack("i", arr[idx]) 
            fd.write(buf) 
    
    fd.close() 

def read_restart_domain(task_id, N_task, N_chunk): 
    filename = "restart_domain_%d" % (task_id, )
    fd = open(filename, "rb") 

    output_list = [] 

    for idx in range(6): 
        buf = fd.read(N_task * 4) 
        output_list.append(np.array(struct.unpack("%di" % (N_task, ), buf)))

    for idx in range(3): 
        for i in range(2): 
            buf = fd.read(N_chunk[idx] * 4) 
            output_list.append(np.array(struct.unpack("%di" % (N_chunk[idx], ), buf)))
    
    buf = fd.read(N_task * 4) 
    output_list.append(np.array(struct.unpack("%di" % (N_task, ), buf)))

    buf = fd.read((parameters["n_cell_base"] ** 3) * 4) 
    output_list.append(np.array(struct.unpack("%di" % (parameters["n_cell_base"] ** 3, ), buf))) 

    if parameters["include_stars"] == 1: 
        buf = fd.read(N_task * 4) 
        output_list.append(np.array(struct.unpack("%di" % (N_task, ), buf)))

    return output_list 

def write_restart_kernel(task_id, data_dbl): 
    filename = "restart_kernel_%d" % (task_id, )
    fd = open(filename, "wb") 
    
    for arr in data_dbl: 
        for idx in range(len(arr)): 
            buf = struct.pack("d", arr[idx]) 
            fd.write(buf) 
    
    fd.close() 

def read_restart_kernel(task_id, N, N_star): 
    filename = "restart_kernel_%d" % (task_id, )
    fd = open(filename, "rb") 

    output_list = [] 

    buf = fd.read(N * 8) 
    output_list.append(np.array(struct.unpack("%dd" % (N, ), buf)))

    if parameters["include_stars"] == 1 and parameters["smooth_stars"] == 1: 
        buf = fd.read(N_star * 8) 
        output_list.append(np.array(struct.unpack("%dd" % (N_star, ), buf)))

    fd.close() 

    return output_list 


#nrianjan: Introducing rotation matrix to have a perfect face on view of the galaxy 
def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                    [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                    [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])




def main(): 
    parameter_file = sys.argv[1] 

    # Setup MPI 
    comm = MPI.COMM_WORLD 
    rank = comm.Get_rank() 
    N_task = comm.Get_size() 

    read_parameters(parameter_file) 

    if parameters["include_stars"] == 0 and parameters["refinement_scheme"] == 1: 
        raise Exception("ERROR: refinement_scheme == 1 but include_stars == 0. We need to include stars before we can refine on gas + stars. Aborting!") 

    if parameters["N_age_bins"] > len(log_stellar_age_Myr_bin_max) and parameters["include_stars"] == 1: 
        raise Exception("ERROR: N_age_bins = %d, but we cannot include more than %d age bins. Aborting!" % (parameters["N_age_bins"], len(log_stellar_age_Myr_bin_max))) 

    #niranjan: commenting out the following for now as it has been impemented
    #if parameters["cosmo_flag"] == 1: 
    #    raise Exception("ERROR: cosmo_flag = %d, but we have not fully implemented the correct unit conversions for cosmological runs. Aborting!" % (parameters["cosmo_flag"], )) 

    if parameters["dust_model"] != "DC16" and parameters["dust_model"] != "constant": 
        raise Exception("ERROR: dust_model %s not recognised. Aborting!" % (parameters["dust_model"], )) 

    if rank == 0: 
        print("Saturated dust to gas ratio: %.4e" % (dust_to_gas_saturated, )) 
        sys.stdout.flush() 

    # Determine coordinates of the base grid 
    # relative to the centre of the box. 
    delta_grid = parameters["box_size"] / parameters["n_cell_base"] 
    grid_array = np.arange(-parameters["box_size"] / 2.0, parameters["box_size"] / 2.0, delta_grid) 
    grid_array += delta_grid / 2.0     # position on centre of each cell 

    assert len(grid_array) == parameters["n_cell_base"] 
    
    if rank == 0: 
        print("Base AMR grid: %d x %d x %d cells" % (parameters["n_cell_base"], parameters["n_cell_base"], parameters["n_cell_base"])) 
        print("min(x) = %.3f kpc, max(x) = %.3f kpc" % (parameters["centre_x"] + grid_array[0] - (delta_grid / 2.0), parameters["centre_x"] + grid_array[-1] + (delta_grid / 2.0))) 
        print("min(y) = %.3f kpc, max(y) = %.3f kpc" % (parameters["centre_y"] + grid_array[0] - (delta_grid / 2.0), parameters["centre_y"] + grid_array[-1] + (delta_grid / 2.0))) 
        print("min(z) = %.3f kpc, max(z) = %.3f kpc" % (parameters["centre_z"] + grid_array[0] - (delta_grid / 2.0), parameters["centre_z"] + grid_array[-1] + (delta_grid / 2.0))) 
        sys.stdout.flush() 
    
    # Distribute chunks of the base AMR grid between MPI tasks 
    if parameters["automatic_domain_decomposition"] == 1: 
        N_chunk = decompose_3d_factors(int(N_task)) 
    else: 
        N_chunk = np.array([parameters["N_chunk_x"], parameters["N_chunk_y"], parameters["N_chunk_z"]]) 
        if N_task != (N_chunk[0] * N_chunk[1] * N_chunk[2]): 
            raise Exception("ERROR: N_chunk_x = %d, N_chunk_y = %d, N_chunk_z = %d, N_task = %d != %d. Aborting." % (N_chunk[0], N_chunk[1], N_chunk[2], N_task, N_chunk[0] * N_chunk[1] * N_chunk[2])) 

    if rank == 0: 
        print("Dividing base AMR grid between %d MPI tasks: %d x %d x %d" % (N_task, N_chunk[0], N_chunk[1], N_chunk[2])) 
        sys.stdout.flush() 

    # Parse species_file 
    f = open(parameters["species_file"], "r") 
    species_list = [] 
    atomic_mass_list = [] 
    emitter_flag_list = [] 
    
    for line in f: 
        values = line.split() 
        species_list.append(values[0]) 
        atomic_mass_list.append(float(values[1])) 
        emitter_flag_list.append(int(values[2])) 
    f.close() 
    N_species = len(species_list) 

    if rank == 0: 
        print("Extracting the following species:") 
        sys.stdout.flush() 
        for i in range(N_species): 
            print("%s, emitter: %d" % (species_list[i], emitter_flag_list[i])) 
            sys.stdout.flush() 

    # Root task reads in HDF5 snapshot file 
    if rank == 0: 
        if parameters['Multifiles'] == 0:
            try: 
                h5file = tables.openFile(parameters["infile"], "r") 
            except AttributeError: 
                h5file = tables.open_file(parameters["infile"], "r") 


            # Convert everything to single precision, for 
            # consistency with the send/recv buffers later on. 
            particle_coords = np.float64(h5file.root.PartType0.Coordinates.read())   # kpc
            particle_hsml = np.float64(h5file.root.PartType0.SmoothingLength.read()) # kpc 
            particle_mass = np.float64(h5file.root.PartType0.Masses.read() * 1.0e10) # Msol 
            particle_velocity = np.float64(h5file.root.PartType0.Velocities.read())  # km s^-1 
            
            if parameters["use_eqm_abundances"] == 1: 
                print("Reading equilibrium abundance array") 
                sys.stdout.flush() 
                particle_chem = np.float64(h5file.root.PartType0.EqmChimesAbundances.read()) 
            else: 
                print("Reading non-equilibrium abundance array") 
                sys.stdout.flush() 
                particle_chem = np.float64(h5file.root.PartType0.ChimesAbundances.read()) 
    
            particle_u = np.float64(h5file.root.PartType0.InternalEnergy.read() * 1.0e10)  # cgs 
            particle_mu = np.float64(h5file.root.PartType0.ChimesMu.read()) 
            particle_Z = np.float64(h5file.root.PartType0.Metallicity.read()) 
            particle_rho = np.float64(h5file.root.PartType0.Density.read())         # code units 
    
            if parameters["max_velocity"] > 0.0: 
                # If max_velocity is positive, limit the 
                # gas particle velocities in the Cartesian
                # directions to be no greater than max_velocity. 
                particle_velocity[(particle_velocity > parameters["max_velocity"])] = parameters["max_velocity"]
                particle_velocity[(particle_velocity < -parameters["max_velocity"])] = -parameters["max_velocity"]
    
            if parameters["include_stars"] == 1: 
                particle_star_form = np.float64(h5file.root.PartType4.StellarFormationTime.read()) 
                particle_star_mass = np.float64(h5file.root.PartType4.Masses.read() * 1.0e10)  # Msol 
                particle_star_coords = np.float64(h5file.root.PartType4.Coordinates.read())  # kpc 
    
                if parameters["cosmo_flag"] == 0: 
                    # Note that the star arrays are stored in the 
                    # order: Type 4, Type 2, Type 3 
                    try: 
                        particle_star_form_2 = np.float64(h5file.root.PartType2.StellarFormationTime.read()) 
                        particle_star_form = np.concatenate((particle_star_form, particle_star_form_2)) 
    
                        particle_star_mass_2 = np.float64(h5file.root.PartType2.Masses.read() * 1.0e10)  # Msol 
                        particle_star_mass = np.concatenate((particle_star_mass, particle_star_mass_2))  # Msol 
    
                        particle_star_coords_2 = np.float64(h5file.root.PartType2.Coordinates.read())  # kpc 
                        particle_star_coords = np.concatenate((particle_star_coords, particle_star_coords_2))  # kpc 
                    except tables.NoSuchNodeError:
                        print("Type 2 stars not present. Continuing.") 
    
                    try: 
                        particle_star_form_3 = np.float64(h5file.root.PartType3.StellarFormationTime.read()) 
                        particle_star_form = np.concatenate((particle_star_form, particle_star_form_3)) 
    
                        particle_star_mass_3 = np.float64(h5file.root.PartType3.Masses.read() * 1.0e10)  # Msol 
                        particle_star_mass = np.concatenate((particle_star_mass, particle_star_mass_3))  # Msol 
    
                        particle_star_coords_3 = np.float64(h5file.root.PartType3.Coordinates.read())  # kpc 
                        particle_star_coords = np.concatenate((particle_star_coords, particle_star_coords_3))  # kpc 
                    except tables.NoSuchNodeError:
                        print("Type 3 stars not present. Continuing.") 
            
                try: 
                    time = h5file.root.Header._f_getAttr('Time') 
                except AttributeError: 
                    time = h5file.root.Header._f_getattr('Time') 
    
                particle_star_age = (time - particle_star_form) * parameters["unit_time_Myr"] 
    
                if parameters["smooth_stars"] == 1: 
                    # Read in star smoothing lengths from a file. 
                    # They are stored in the order Type 4, 2, 3, 
                    # the same as above. 
                    if parameters["star_hsml_file"] == None: 
                        raise Exception("ERROR: smooth_stars = %d, but star_hsml_file has not been set. Please specify a file containing the stellar smoothing lengths. Aborting." % (parameters["smooth_stars"], )) 
    
                    try: 
                        star_file = tables.openFile(parameters["star_hsml_file"], "r") 
                    except AttributeError: 
                        star_file = tables.open_file(parameters["star_hsml_file"], "r") 
    
                    particle_star_hsml = np.float64(star_file.root.hsml.read()) 
    
                    # Impose maximum stellar smoothing length. 
                    if (parameters["max_star_hsml"] > 0.0): 
                        particle_star_hsml[(particle_star_hsml > parameters["max_star_hsml"])] = parameters["max_star_hsml"] 
    
                    star_file.close() 




        else: #If snapshot consists of multiple files #niranjan 2022
            #try:
            #    h5file = tables.openFile(parameters["infile"], "r")
            #except AttributeError:
            #    h5file = tables.open_file(parameters["infile"], "r")

            input_file = parameters['infile']

            LastSlash = input_file.rindex('/')
            input_dir = input_file[:LastSlash+1:]

            LastUnderscore = input_file.rindex('_')
            snapnum  = int(input_file[LastUnderscore+1:])

            #header = g.readsnap(input_dir, snapnum, 0, header_only=1)


            time = load_from_snapshot('Time', -1, input_dir, snapnum)

            # Convert everything to single precision, for 
            # consistency with the send/recv buffers later on. 
            #particle_coords = np.float64(h5file.root.PartType0.Coordinates.read())   # kpc 
            particle_coords = np.float64(load_from_snapshot( 'Coordinates', 0, input_dir, snapnum))

            #particle_hsml = np.float64(h5file.root.PartType0.SmoothingLength.read()) # kpc 
            particle_hsml = np.float64(load_from_snapshot( 'SmoothingLength', 0, input_dir, snapnum)) #kpc

            #particle_mass = np.float64(h5file.root.PartType0.Masses.read() * 1.0e10) # Msol 
            particle_mass = np.float64(load_from_snapshot( 'Masses', 0, input_dir, snapnum) * 1.0e10) #Msol

            #particle_velocity = np.float64(h5file.root.PartType0.Velocities.read())  # km s^-1 
            particle_velocity = np.float64(load_from_snapshot( 'Velocities', 0, input_dir, snapnum)) #km/s

            particle_star_velocity = np.float64(load_from_snapshot( 'Velocities', 4, input_dir, snapnum)) #km/s

            try:
                ChimesOutput = tables.openFile(parameters["ChimesOutputFile"], "r")
            except AttributeError:
                ChimesOutput = tables.open_file(parameters["ChimesOutputFile"], "r")


            if parameters["use_eqm_abundances"] == 1:
                print("Reading equilibrium abundance array")
                sys.stdout.flush()
                particle_chem = np.float64(ChimesOutput.root.EqmChemistryAbundances.read())
            else:
                print("Reading non-equilibrium abundance array")
                sys.stdout.flush()
                particle_chem = np.float64(ChimesOutput.root.ChimesAbundances.read())


            center = np.float64(ChimesOutput.root.Center.read())
            filtering_radius_cm = np.float64(ChimesOutput.root.FilteringRadiusCm.read())    

            #particle_u = np.float64(h5file.root.PartType0.InternalEnergy.read() * 1.0e10)  # cgs 
            particle_u = np.float64(load_from_snapshot( 'InternalEnergy', 0, input_dir, snapnum)) * 1.0e10 #cgs

            #particle_mu = np.float64(h5file.root.PartType0.ChimesMu.read())
            particle_mu = load_from_snapshot('ChimesMu', 0, input_dir, snapnum)
            particle_metallicity = load_from_snapshot( 'Metallicity', 0, input_dir, snapnum)

            if (np.size(particle_mu) == 1): #if 'ChimesMu' is not in the snapshot, load_from_snap will return 0, not an array
                print ('particle_mu not found in snapshot, calculating manually...')
                helium_mass_fraction = particle_metallicity[:,1]                   
                y_helium = helium_mass_fraction / (4*(1-helium_mass_fraction))
                ElectronAbundance = load_from_snapshot('ElectronAbundance', 0, input_dir, snapnum)                 
                particle_mu = (1.0 + 4*y_helium) / (1+y_helium+ElectronAbundance)

            
            #particle_Z = np.float64(h5file.root.PartType0.Metallicity.read())
            particle_Z = np.float64(load_from_snapshot( 'Metallicity', 0, input_dir, snapnum))
            particle_Z = particle_Z[:,0:11] #niranjan2022: taking only first 11 columns to check if mpi works

            #particle_rho = np.float64(h5file.root.PartType0.Density.read())         # code units 
            particle_rho = np.float64(load_from_snapshot( 'Density', 0, input_dir, snapnum))

            if parameters["max_velocity"] > 0.0:
                # If max_velocity is positive, limit the 
                # gas particle velocities in the Cartesian
                # directions to be no greater than max_velocity. 
                particle_velocity[(particle_velocity > parameters["max_velocity"])] = parameters["max_velocity"]
                particle_velocity[(particle_velocity < -parameters["max_velocity"])] = -parameters["max_velocity"]

            if parameters["include_stars"] == 1:
                #particle_star_form = np.float64(h5file.root.PartType4.StellarFormationTime.read())
                particle_star_form = np.float64(load_from_snapshot( 'StellarFormationTime', 4, input_dir, snapnum))

                #particle_star_mass = np.float64(h5file.root.PartType4.Masses.read() * 1.0e10)  # Msol 
                particle_star_mass = np.float64(load_from_snapshot( 'Masses', 4, input_dir, snapnum) * 1.0e10)

                #particle_star_coords = np.float64(h5file.root.PartType4.Coordinates.read())  # kpc 
                particle_star_coords = np.float64(load_from_snapshot( 'Coordinates', 4, input_dir, snapnum))

                if parameters["cosmo_flag"] == 0:
                    # Note that the star arrays are stored in the 
                    # order: Type 4, Type 2, Type 3 
                    try:
                        #particle_star_form_2 = np.float64(h5file.root.PartType2.StellarFormationTime.read())
                        particle_star_form_2 = np.float64(load_from_snapshot( 'StellarFormationTime', 2, input_dir, snapnum))
                        particle_star_form = np.concatenate((particle_star_form, particle_star_form_2))

                        #particle_star_mass_2 = np.float64(h5file.root.PartType2.Masses.read() * 1.0e10)  # Msol 
                        particle_star_mass_2 = np.float64(load_from_snapshot( 'Masses', 2, input_dir, snapnum))
                        particle_star_mass = np.concatenate((particle_star_mass, particle_star_mass_2))  # Msol 

                        #particle_star_coords_2 = np.float64(h5file.root.PartType2.Coordinates.read())  # kpc 
                        particle_star_coords_2 = np.float64(load_from_snapshot( 'Coordinates', 2, input_dir, snapnum))
                        particle_star_coords = np.concatenate((particle_star_coords, particle_star_coords_2))  # kpc 

                    #except tables.NoSuchNodeError:
                    except KeyError:
                        print("Type 2 stars not present. Continuing.")

                    try:
                        #particle_star_form_3 = np.float64(h5file.root.PartType3.StellarFormationTime.read())
                        particle_star_form_3 = np.float64(load_from_snapshot( 'StellarFormationTime', 3, input_dir, snapnum))
                        particle_star_form = np.concatenate((particle_star_form, particle_star_form_3))

                        #particle_star_mass_3 = np.float64(h5file.root.PartType3.Masses.read() * 1.0e10)  # Msol 
                        particle_star_mass_3 = np.float64(load_from_snapshot( 'Masses', 3, input_dir, snapnum))
                        particle_star_mass = np.concatenate((particle_star_mass, particle_star_mass_3))  # Msol 

                        #particle_star_coords_3 = np.float64(h5file.root.PartType3.Coordinates.read())  # kpc 
                        particle_star_coords_3 = np.float64(load_from_snapshot( 'Coordinates', 3, input_dir, snapnum))
                        particle_star_coords = np.concatenate((particle_star_coords, particle_star_coords_3))  # kpc                       
                      
                      
                     #except tables.NoSuchNodeError:
                    except KeyError:
                        print("Type 3 stars not present. Continuing.")

                    particle_star_age = (time - particle_star_form) * parameters["unit_time_Myr"]

                else:
                   try:
                     omega0 = load_from_snapshot( 'Omega0', -1, input_dir, snapnum)
                     hubble = load_from_snapshot('HubbleParam', -1, input_dir, snapnum)

                     H0_cgs = hubble * 3.2407789e-18 # Converting HubbleParam (in 100 km/s/Mpc) to s^-1 
                     seconds_in_a_Myr = 3.15576e13 

                     particle_star_coords = np.float64(load_from_snapshot('Coordinates', 4, input_dir, snapnum))  #kpc
                     particle_star_mass = np.float64(load_from_snapshot('Masses', 4, input_dir, snapnum) * 1.0e10) #Msol
                     particle_star_form = np.float64(load_from_snapshot( 'StellarFormationTime', 4, input_dir, snapnum))

                     expansion_factor = load_from_snapshot('Time', -1, input_dir, snapnum)

                     x_form = (omega0 / (1.0 - omega0)) / (particle_star_form ** 3.0)
                     x_now = (omega0 / (1.0 - omega0)) / (expansion_factor ** 3.0)
                     particle_star_age = (2. / (3. * np.sqrt(1 - omega0))) * np.log(np.sqrt(x_form * x_now)/ ((np.sqrt(1.0 + x_now) - 1.0) * (np.sqrt(1.0 + x_form) + 1.0)))
                     particle_star_age /= H0_cgs
                     particle_star_age /= seconds_in_a_Myr

                     
                   except KeyError:
                     print("Type 4 star particles are not present. Continuing.")
                     sys.stdout.flush()
                     particle_star_coords = np.empty((0, 3), dtype = np.float64)
                     particle_star_mass = np.empty(0, dtype = np.float64)
                     particle_star_age = np.empty(0, dtype = np.float64) 
                      

                #try:
                    #time = h5file.root.Header._f_getAttr('Time')
                    #time = header['time']
                #except AttributeError:
                    #time = h5file.root.Header._f_getattr('Time')

                #particle_star_age = (time - particle_star_form) * parameters["unit_time_Myr"]

                if parameters["smooth_stars"] == 1:
                    # Read in star smoothing lengths from a file. 
                    # They are stored in the order Type 4, 2, 3, 
                    # the same as above. 
                    if parameters["star_hsml_file"] == None:
                        raise Exception("ERROR: smooth_stars = %d, but star_hsml_file has not been set. Please specify a file containing the stellar smoothing lengths. Aborting." % (parameters["smooth_stars"], ))

                    try:
                        star_file = tables.openFile(parameters["star_hsml_file"], "r")
                    except AttributeError:
                        star_file = tables.open_file(parameters["star_hsml_file"], "r")

                    #particle_star_hsml = np.float64(star_file.root.hsml.read())
                    particle_star_hsml = np.float64(star_file.root.star_hsml.read()) / 3.086e21 #reading star_hsml saved during chimes calculations and converted to kpc
                    print("MIN AND MAX STAR_HSML = {},{}".format(np.min(particle_star_hsml), np.max(particle_star_hsml)))
                    print("SHAPE OF STAR_HSML IS = {}".format(np.shape(particle_star_hsml)))

                    # Impose maximum stellar smoothing length. 
                    if (parameters["max_star_hsml"] > 0.0):
                        particle_star_hsml[(particle_star_hsml > parameters["max_star_hsml"])] = parameters["max_star_hsml"]

                    star_file.close()

                #Filtering all data based on filtering_radius
           
                radius = filtering_radius_cm #/ 3.0857e21  #converting radius to kpc from cm
                #center /= 3.0857e21  #converting center to kpc units
                CM_vel_filtering_radius = 3.086e+21 #cm 

                print('RADIUS AND CENTER ARE {}, AND {}'.format(radius, center))      
                particle_coords_cm = particle_coords * 3.0857e21
                particle_star_coords_cm = particle_star_coords * 3.0857e21
     
                particle_coords_cm -= center
                particle_star_coords_cm -= center

                #shifting gas and star particle positions wrt to the center
                center /= 3.0857e21
                particle_coords -= center
                particle_star_coords -= center
               

                print('MIN AND MAX particle_coords after shift = {},{}'.format(np.min(particle_coords), np.max(particle_coords))) 
                
          
                #creating mask for gas particles
                R_gas = np.sqrt((particle_coords_cm * particle_coords_cm).sum(axis=1))
                print('MIN AND MAX R_GAS = {},{}'.format(np.min(R_gas), np.max(R_gas)))
                gas_mask = R_gas < radius
                #R_gas = R_gas[gas_mask]

                
                #creating mask for star particles
                R_star = np.sqrt((particle_star_coords_cm * particle_star_coords_cm).sum(axis=1))
                print('MIN AND MAX R_STAR = {},{}'.format(np.min(R_star), np.max(R_star)))
                star_mask = R_star < radius
                #R_star = R_star[star_mask]

                if (parameters["Subtract_CoMvelocity"]):
                    #calculating the velocity of Center of Mass using the vel of stars within 1 kpc
                    CM_vel_calc_mask = R_star < CM_vel_filtering_radius
                    particle_star_velocity_CM_calc = particle_star_velocity[CM_vel_calc_mask, :]
                    
                    vel_CM_x = (np.sum(particle_star_velocity_CM_calc[:,0] * particle_star_mass[CM_vel_calc_mask]))/(np.sum(particle_star_mass[CM_vel_calc_mask]))
                    vel_CM_y = (np.sum(particle_star_velocity_CM_calc[:,1] * particle_star_mass[CM_vel_calc_mask]))/(np.sum(particle_star_mass[CM_vel_calc_mask]))
                    vel_CM_z = (np.sum(particle_star_velocity_CM_calc[:,2] * particle_star_mass[CM_vel_calc_mask]))/(np.sum(particle_star_mass[CM_vel_calc_mask]))
                    vel_CM = np.array([vel_CM_x, vel_CM_y, vel_CM_z], dtype = float)

                print("NUMBER OF NON-ZERO ELEMENTS IN STAR_MASK IS = {}".format(np.count_nonzero(star_mask.astype(int))))
                print("NUMBER OF NON-ZERO ELEMENTS IN GAS_MASK IS = {}".format(np.count_nonzero(gas_mask.astype(int))))
                particle_coords = particle_coords[gas_mask,:]
                particle_hsml = particle_hsml[gas_mask]
                particle_mass = particle_mass[gas_mask]
                if (parameters["Subtract_CoMvelocity"]):
                    particle_velocity = particle_velocity[gas_mask,:] - vel_CM #niranjan: subtracting the velocity of center of mass from all gas velocities
                else:
                    particle_velocity = particle_velocity[gas_mask,:]
                particle_u = particle_u[gas_mask]
                particle_mu = particle_mu[gas_mask]
                particle_Z = particle_Z[gas_mask,:]
                particle_rho = particle_rho[gas_mask]

                print("SHAPE OF PARTICLE_STAR_COORDS BEFORE FILTERING IS = {}".format(np.shape(particle_star_coords)))
                if parameters["include_stars"] == 1:
                    particle_star_form = particle_star_form[star_mask]
                    particle_star_mass = particle_star_mass[star_mask]
                    particle_star_coords = particle_star_coords[star_mask,:]
                    particle_star_age = particle_star_age[star_mask]

                print("SHAPE OF PARTICLE_STAR_COORDS IS = {}".format(np.shape(particle_star_coords)))
                ChimesOutput.close()
           
            if (parameters["Rotate_to_faceon"] == 1):
                #Starting the rotation of coordinates and velocities to make the galaxy face on:
                zaxis = np.array([0.,0.,1.])
                L_gas = np.zeros(3)
                L_gas = np.sum(np.cross(particle_coords[:,:],particle_velocity[:,:])*particle_mass[:,np.newaxis],axis=0)
                L_dir = L_gas/np.linalg.norm(L_gas)  

                rotax = np.cross(L_dir,zaxis)
                rotangle = np.arccos(np.dot(L_dir,zaxis))
                rotmatrix = rotation_matrix(rotax,rotangle)   

                particle_coords = np.tensordot(rotmatrix, particle_coords, axes=(1,1)).T
                particle_velocity = np.tensordot(rotmatrix, particle_velocity, axes=(1,1)).T
                particle_star_coords = np.tensordot(rotmatrix, particle_star_coords, axes=(1,1)).T            


        if os.path.exists("restart_domain_%d" % (rank, )): 
            print("Reading domain data from restart file") 
            sys.stdout.flush() 

            restart_domain_data = read_restart_domain(rank, N_task, N_chunk) 
            
            x_ind_low_task = restart_domain_data[0] 
            x_ind_hi_task = restart_domain_data[1] 
            y_ind_low_task = restart_domain_data[2] 
            y_ind_hi_task = restart_domain_data[3] 
            z_ind_low_task = restart_domain_data[4] 
            z_ind_hi_task = restart_domain_data[5] 
            x_ind_low_chunk = restart_domain_data[6]
            x_ind_hi_chunk = restart_domain_data[7]
            y_ind_low_chunk = restart_domain_data[8]
            y_ind_hi_chunk = restart_domain_data[9]
            z_ind_low_chunk = restart_domain_data[10]
            z_ind_hi_chunk = restart_domain_data[11]
            N_parts = restart_domain_data[12]
            grid_task = restart_domain_data[13].reshape((parameters["n_cell_base"], parameters["n_cell_base"], parameters["n_cell_base"]), order = 'C') 
            if parameters["include_stars"] == 1: 
                N_parts_star = restart_domain_data[14]
        else: 
            # Determine how many particles to send to each task. Send any 
            # particles that overlap with a task's domain. 
            N_parts = np.zeros(N_task, dtype = np.int)  # Number of gas particles sent to each task 
            grid_task = np.ones((parameters["n_cell_base"], parameters["n_cell_base"], parameters["n_cell_base"]), dtype = np.int) * (-1)  # This array will store which task will take each cell in the base grid 

            if parameters["include_stars"] == 1: 
                N_parts_star = np.zeros(N_task, dtype = np.int)  # Number of star particles sent to each task 

            # The following arrays will store the indices of the 
            # base grid in the x, y and z directions of the 
            # boundaries of each task's domain 
            x_ind_low_task = np.zeros(N_task, dtype = np.int) 
            x_ind_hi_task = np.zeros(N_task, dtype = np.int) 
            y_ind_low_task = np.zeros(N_task, dtype = np.int) 
            y_ind_hi_task = np.zeros(N_task, dtype = np.int) 
            z_ind_low_task = np.zeros(N_task, dtype = np.int) 
            z_ind_hi_task = np.zeros(N_task, dtype = np.int) 

            # As above, but now for each chunk in the 
            # x, y and z directions 
            x_ind_low_chunk = np.zeros(N_chunk[0], dtype = np.int) 
            x_ind_hi_chunk = np.zeros(N_chunk[0], dtype = np.int) 
            y_ind_low_chunk = np.zeros(N_chunk[1], dtype = np.int) 
            y_ind_hi_chunk = np.zeros(N_chunk[1], dtype = np.int) 
            z_ind_low_chunk = np.zeros(N_chunk[2], dtype = np.int) 
            z_ind_hi_chunk = np.zeros(N_chunk[2], dtype = np.int) 

            # Calculate the initial boundaries of each chunk. 
            if parameters["initial_domain_spacing"] == 0: 
                # Uniform spacing.
                for i in range(N_chunk[0]): 
                    x_ind_low_chunk[i] = (i * parameters["n_cell_base"]) // N_chunk[0] 
                    x_ind_hi_chunk[i] = ((i + 1) * parameters["n_cell_base"]) // N_chunk[0] 
                for j in range(N_chunk[1]): 
                    y_ind_low_chunk[j] = (j * parameters["n_cell_base"]) // N_chunk[1] 
                    y_ind_hi_chunk[j] = ((j + 1) * parameters["n_cell_base"]) // N_chunk[1] 
                for k in range(N_chunk[2]): 
                    z_ind_low_chunk[k] = (k * parameters["n_cell_base"]) // N_chunk[2] 
                    z_ind_hi_chunk[k] = ((k + 1) * parameters["n_cell_base"]) // N_chunk[2] 
            elif parameters["initial_domain_spacing"] == 1: 
                # Centrally concentrated. 
                x_ind_low_chunk[0] = 0 
                x_ind_hi_chunk[0] = parameters["n_cell_base"] // 2 
                x_ind_low_chunk[-1] = parameters["n_cell_base"] // 2 
                x_ind_hi_chunk[-1] = parameters["n_cell_base"]
                for i in range((N_chunk[0] // 2) - 1): 
                    chunk_idx = (N_chunk[0] // 2) - i - 1
                    x_ind_low_chunk[chunk_idx] = (parameters["n_cell_base"] // 2) - (i + 1)
                    x_ind_hi_chunk[chunk_idx] = (parameters["n_cell_base"] // 2) - i 
                    x_ind_hi_chunk[0] -= 1
                    
                    chunk_idx = (N_chunk[0] // 2) + i 
                    x_ind_low_chunk[chunk_idx] = (parameters["n_cell_base"] // 2) + i 
                    x_ind_hi_chunk[chunk_idx] = (parameters["n_cell_base"] // 2) + (i + 1)
                    x_ind_low_chunk[-1] += 1

                y_ind_low_chunk[0] = 0 
                y_ind_hi_chunk[0] = parameters["n_cell_base"] // 2 
                y_ind_low_chunk[-1] = parameters["n_cell_base"] // 2 
                y_ind_hi_chunk[-1] = parameters["n_cell_base"]
                for j in range((N_chunk[1] // 2) - 1): 
                    chunk_idx = (N_chunk[1] // 2) - j - 1
                    y_ind_low_chunk[chunk_idx] = (parameters["n_cell_base"] // 2) - (j + 1)
                    y_ind_hi_chunk[chunk_idx] = (parameters["n_cell_base"] // 2) - j 
                    y_ind_hi_chunk[0] -= 1
                    
                    chunk_idx = (N_chunk[1] // 2) + j 
                    y_ind_low_chunk[chunk_idx] = (parameters["n_cell_base"] // 2) + j 
                    y_ind_hi_chunk[chunk_idx] = (parameters["n_cell_base"] // 2) + (j + 1)
                    y_ind_low_chunk[-1] += 1

                z_ind_low_chunk[0] = 0 
                z_ind_hi_chunk[0] = parameters["n_cell_base"] // 2 
                z_ind_low_chunk[-1] = parameters["n_cell_base"] // 2 
                z_ind_hi_chunk[-1] = parameters["n_cell_base"]
                for k in range((N_chunk[2] // 2) - 1): 
                    chunk_idx = (N_chunk[2] // 2) - k - 1
                    z_ind_low_chunk[chunk_idx] = (parameters["n_cell_base"] // 2) - (k + 1)
                    z_ind_hi_chunk[chunk_idx] = (parameters["n_cell_base"] // 2) - k 
                    z_ind_hi_chunk[0] -= 1
                    
                    chunk_idx = (N_chunk[2] // 2) + k 
                    z_ind_low_chunk[chunk_idx] = (parameters["n_cell_base"] // 2) + k 
                    z_ind_hi_chunk[chunk_idx] = (parameters["n_cell_base"] // 2) + (k + 1)
                    z_ind_low_chunk[-1] += 1            
            else: 
                raise Exception("The initial_domain_spacing parameter must be set to 0 or 1. Aborting.")

            # Initial domain decomposition 
            for i in range(N_chunk[0]): 
                for j in range(N_chunk[1]): 
                    for k in range(N_chunk[2]): 
                        task_id = (i * N_chunk[1] * N_chunk[2]) + (j * N_chunk[2]) + k 
                    
                        # Set the domain boundaries 
                        # from the chunk boundaries 
                        x_ind_low = x_ind_low_chunk[i]
                        x_ind_hi = x_ind_hi_chunk[i]
                        y_ind_low = y_ind_low_chunk[j]
                        y_ind_hi = y_ind_hi_chunk[j]
                        z_ind_low = z_ind_low_chunk[k]
                        z_ind_hi = z_ind_hi_chunk[k]
                        
                        # Store these boundaries 
                        x_ind_low_task[task_id] = x_ind_low 
                        x_ind_hi_task[task_id] = x_ind_hi 
                        y_ind_low_task[task_id] = y_ind_low 
                        y_ind_hi_task[task_id] = y_ind_hi 
                        z_ind_low_task[task_id] = z_ind_low 
                        z_ind_hi_task[task_id] = z_ind_hi 
                        
                        ind_slice = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                     (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                     (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                     (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                     (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                     (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)))

                        N_parts[task_id] = len(particle_mass[ind_slice]) 
                        
                        grid_task[x_ind_low:x_ind_hi, y_ind_low:y_ind_hi, z_ind_low:z_ind_hi] = task_id 

                        if parameters["include_stars"] == 1: 
                            if parameters["smooth_stars"] == 1: 
                                ind_slice_star = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                  (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                  (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                  (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                  (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                  (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)))
                            else: 
                                ind_slice_star = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0))) & 
                                                  (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                  (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0))) & 
                                                  (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                  (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0))) & 
                                                  (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0))))
                                
                            N_parts_star[task_id] = len(particle_star_mass[ind_slice_star]) 

            if parameters["adaptive_domain_decomposition"] == 1: 
                # Adjust the sizes of the domains to minimise the maximum 
                # number of gas or star particles in each domain. 
                print("Calculating adaptive domain decomposition") 
                sys.stdout.flush() 
                accept_adjustment = 1 
                domain_iter = 1
                while accept_adjustment == 1: 
                    accept_adjustment = 0 

                    # x-direction 
                    for i in range(N_chunk[0] - 1): 
                        # Determine whether the left or right 
                        # side of the boundary has the highest 
                        # maximum number of particles. 
                        N_max_left = 0 
                        N_max_right = 0 
                        for j in range(N_chunk[1]): 
                            for k in range(N_chunk[2]): 
                                task_id_left = (i * N_chunk[1] * N_chunk[2]) + (j * N_chunk[2]) + k 
                                task_id_right = ((i + 1) * N_chunk[1] * N_chunk[2]) + (j * N_chunk[2]) + k 
                                if N_parts[task_id_left] > N_max_left: 
                                    N_max_left = N_parts[task_id_left] 
                                if N_parts[task_id_right] > N_max_right: 
                                    N_max_right = N_parts[task_id_right] 
                                if parameters["include_stars"] == 1: 
                                    if N_parts_star[task_id_left] > N_max_left: 
                                        N_max_left = N_parts_star[task_id_left] 
                                    if N_parts_star[task_id_right] > N_max_right: 
                                        N_max_right = N_parts_star[task_id_right] 

                        if N_max_left < N_max_right: 
                            # Shift boundary right, if able 
                            if x_ind_hi_chunk[i + 1] - x_ind_low_chunk[i + 1] <= 1: 
                                continue 
                            boundary_shift = 1 
                        else: 
                            # Shift boundary left, if able 
                            if x_ind_hi_chunk[i] - x_ind_low_chunk[i] <= 1: 
                                continue 
                            boundary_shift = -1 

                        # Test whether shifting the boundary 
                        # reduces the max particles. 
                        N_max_new_left = 0 
                        N_max_new_right = 0
                        for j in range(N_chunk[1]): 
                            for k in range(N_chunk[2]): 
                                y_ind_low = y_ind_low_chunk[j] 
                                y_ind_hi = y_ind_hi_chunk[j] 
                                z_ind_low = z_ind_low_chunk[k] 
                                z_ind_hi = z_ind_hi_chunk[k] 

                                # Left side 
                                x_ind_low = x_ind_low_chunk[i] 
                                x_ind_hi = x_ind_hi_chunk[i] + boundary_shift

                                ind_slice = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                             (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                             (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)))
                            
                                N_parts_left = len(particle_mass[ind_slice]) 
                                if N_parts_left > N_max_new_left: 
                                    N_max_new_left = N_parts_left 
                        
                                if parameters["include_stars"] == 1: 
                                    if parameters["smooth_stars"] == 1: 
                                        ind_slice_star = ((particle_star_coords[:, 0] > 
                                                           (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 0] < 
                                                           (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                          (particle_star_coords[:, 1] > 
                                                           (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 1] < 
                                                           (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                          (particle_star_coords[:, 2] > 
                                                           (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 2] < 
                                                           (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)))
                                    else: 
                                        ind_slice_star = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0))))

                                    N_parts_star_left = len(particle_star_mass[ind_slice_star]) 

                                    if N_parts_star_left > N_max_new_left: 
                                        N_max_new_left = N_parts_star_left 

                                # Right side 
                                x_ind_low = x_ind_low_chunk[i + 1] + boundary_shift
                                x_ind_hi = x_ind_hi_chunk[i + 1] 

                                ind_slice = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                             (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                             (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)))
                                
                                N_parts_right = len(particle_mass[ind_slice]) 
                                if N_parts_right > N_max_new_right: 
                                    N_max_new_right = N_parts_right 

                                if parameters["include_stars"] == 1: 
                                    if parameters["smooth_stars"] == 1: 
                                        ind_slice_star = ((particle_star_coords[:, 0] > 
                                                           (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 0] < 
                                                           (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                          (particle_star_coords[:, 1] > 
                                                           (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 1] < 
                                                           (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                          (particle_star_coords[:, 2] > 
                                                           (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 2] < 
                                                           (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)))
                                    else: 
                                        ind_slice_star = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0))))

                                    N_parts_star_right = len(particle_star_mass[ind_slice_star]) 

                                    if N_parts_star_right > N_max_new_right: 
                                        N_max_new_right = N_parts_star_right 

                        if max([N_max_new_left, N_max_new_right]) < max([N_max_left, N_max_right]): 
                            # This shift has indeed reduced the 
                            # maximum particle load. 
                            accept_adjustment = 1 

                            # Update boundaries and particles numbers 
                            x_ind_hi_chunk[i] += boundary_shift 
                            x_ind_low_chunk[i + 1] += boundary_shift 
                            for j in range(N_chunk[1]): 
                                for k in range(N_chunk[2]): 
                                    y_ind_low = y_ind_low_chunk[j] 
                                    y_ind_hi = y_ind_hi_chunk[j] 
                                    z_ind_low = z_ind_low_chunk[k] 
                                    z_ind_hi = z_ind_hi_chunk[k] 
                                    
                                    # Left side 
                                    x_ind_low = x_ind_low_chunk[i] 
                                    x_ind_hi = x_ind_hi_chunk[i] 
                                    task_id = (i * N_chunk[1] * N_chunk[2]) + (j * N_chunk[2]) + k 
                                    x_ind_low_task[task_id] = x_ind_low 
                                    x_ind_hi_task[task_id] = x_ind_hi 
                                    
                                    ind_slice = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                                 (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                                 (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)))
                                    
                                    N_parts[task_id] = len(particle_mass[ind_slice]) 
                                    grid_task[x_ind_low:x_ind_hi, y_ind_low:y_ind_hi, z_ind_low:z_ind_hi] = task_id 
                                    
                                    if parameters["include_stars"] == 1: 
                                        if parameters["smooth_stars"] == 1: 
                                            ind_slice_star = ((particle_star_coords[:, 0] > 
                                                               (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 0] < 
                                                               (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                              (particle_star_coords[:, 1] > 
                                                               (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 1] < 
                                                               (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                              (particle_star_coords[:, 2] > 
                                                               (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 2] < 
                                                               (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)))
                                        else: 
                                            ind_slice_star = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0))))
                                        
                                        N_parts_star[task_id] = len(particle_star_mass[ind_slice_star]) 

                                    # Right side 
                                    x_ind_low = x_ind_low_chunk[i + 1] 
                                    x_ind_hi = x_ind_hi_chunk[i + 1] 
                                    task_id = ((i + 1) * N_chunk[1] * N_chunk[2]) + (j * N_chunk[2]) + k 
                                    x_ind_low_task[task_id] = x_ind_low 
                                    x_ind_hi_task[task_id] = x_ind_hi 
                                    
                                    ind_slice = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                                 (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                                 (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)))
                                
                                    N_parts[task_id] = len(particle_mass[ind_slice]) 
                                    grid_task[x_ind_low:x_ind_hi, y_ind_low:y_ind_hi, z_ind_low:z_ind_hi] = task_id 
                                    
                                    if parameters["include_stars"] == 1: 
                                        if parameters["smooth_stars"] == 1: 
                                            ind_slice_star = ((particle_star_coords[:, 0] > 
                                                               (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 0] < 
                                                               (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                              (particle_star_coords[:, 1] > 
                                                               (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 1] < 
                                                               (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                              (particle_star_coords[:, 2] > 
                                                               (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 2] < 
                                                               (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)))
                                        else: 
                                            ind_slice_star = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0))))

                                        N_parts_star[task_id] = len(particle_star_mass[ind_slice_star]) 

                    # y-direction 
                    for j in range(N_chunk[1] - 1): 
                        # Determine whether the left or right 
                        # side of the boundary has the highest 
                        # maximum number of particles. 
                        N_max_left = 0 
                        N_max_right = 0 
                        for i in range(N_chunk[0]): 
                            for k in range(N_chunk[2]): 
                                task_id_left = (i * N_chunk[1] * N_chunk[2]) + (j * N_chunk[2]) + k 
                                task_id_right = (i * N_chunk[1] * N_chunk[2]) + ((j + 1) * N_chunk[2]) + k 
                                if N_parts[task_id_left] > N_max_left: 
                                    N_max_left = N_parts[task_id_left] 
                                if N_parts[task_id_right] > N_max_right: 
                                    N_max_right = N_parts[task_id_right] 
                                if parameters["include_stars"] == 1: 
                                    if N_parts_star[task_id_left] > N_max_left: 
                                        N_max_left = N_parts_star[task_id_left] 
                                    if N_parts_star[task_id_right] > N_max_right: 
                                        N_max_right = N_parts_star[task_id_right] 

                        if N_max_left < N_max_right: 
                            # Shift boundary right, if able 
                            if y_ind_hi_chunk[j + 1] - y_ind_low_chunk[j + 1] <= 1: 
                                continue 
                            boundary_shift = 1 
                        else: 
                            # Shift boundary left, if able 
                            if y_ind_hi_chunk[j] - y_ind_low_chunk[j] <= 1: 
                                continue 
                            boundary_shift = -1 

                        # Test whether shifting the boundary 
                        # reduces the max particles. 
                        N_max_new_left = 0 
                        N_max_new_right = 0
                        for i in range(N_chunk[0]): 
                            for k in range(N_chunk[2]): 
                                x_ind_low = x_ind_low_chunk[i] 
                                x_ind_hi = x_ind_hi_chunk[i] 
                                z_ind_low = z_ind_low_chunk[k] 
                                z_ind_hi = z_ind_hi_chunk[k] 
                                
                                # Left side 
                                y_ind_low = y_ind_low_chunk[j] 
                                y_ind_hi = y_ind_hi_chunk[j] + boundary_shift
                                
                                ind_slice = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                             (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                             (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)))
                            
                                N_parts_left = len(particle_mass[ind_slice]) 
                                if N_parts_left > N_max_new_left: 
                                    N_max_new_left = N_parts_left 
                        
                                if parameters["include_stars"] == 1: 
                                    if parameters["smooth_stars"] == 1: 
                                        ind_slice_star = ((particle_star_coords[:, 0] > 
                                                           (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 0] < 
                                                           (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                          (particle_star_coords[:, 1] > 
                                                           (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 1] < 
                                                           (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                          (particle_star_coords[:, 2] > 
                                                           (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 2] < 
                                                           (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)))
                                    else: 
                                        ind_slice_star = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0))))

                                    N_parts_star_left = len(particle_star_mass[ind_slice_star]) 

                                    if N_parts_star_left > N_max_new_left: 
                                        N_max_new_left = N_parts_star_left 

                                # Right side 
                                y_ind_low = y_ind_low_chunk[j + 1] + boundary_shift
                                y_ind_hi = y_ind_hi_chunk[j + 1] 
                                
                                ind_slice = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                             (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                             (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)))
                                
                                N_parts_right = len(particle_mass[ind_slice]) 
                                if N_parts_right > N_max_new_right: 
                                    N_max_new_right = N_parts_right 

                                if parameters["include_stars"] == 1: 
                                    if parameters["smooth_stars"] == 1: 
                                        ind_slice_star = ((particle_star_coords[:, 0] > 
                                                           (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 0] < 
                                                           (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                          (particle_star_coords[:, 1] > 
                                                           (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 1] < 
                                                           (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                          (particle_star_coords[:, 2] > 
                                                           (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 2] < 
                                                           (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)))
                                    else: 
                                        ind_slice_star = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0))))
                                        
                                    N_parts_star_right = len(particle_star_mass[ind_slice_star]) 

                                    if N_parts_star_right > N_max_new_right: 
                                        N_max_new_right = N_parts_star_right 

                        if max([N_max_new_left, N_max_new_right]) < max([N_max_left, N_max_right]): 
                            # This shift has indeed reduced the 
                            # maximum particle load. 
                            accept_adjustment = 1 
                            
                            # Update boundaries and particles numbers 
                            y_ind_hi_chunk[j] += boundary_shift 
                            y_ind_low_chunk[j + 1] += boundary_shift 
                            for i in range(N_chunk[0]): 
                                for k in range(N_chunk[2]): 
                                    x_ind_low = x_ind_low_chunk[i] 
                                    x_ind_hi = x_ind_hi_chunk[i] 
                                    z_ind_low = z_ind_low_chunk[k] 
                                    z_ind_hi = z_ind_hi_chunk[k] 
                                    
                                    # Left side 
                                    y_ind_low = y_ind_low_chunk[j] 
                                    y_ind_hi = y_ind_hi_chunk[j] 
                                    task_id = (i * N_chunk[1] * N_chunk[2]) + (j * N_chunk[2]) + k 
                                    y_ind_low_task[task_id] = y_ind_low 
                                    y_ind_hi_task[task_id] = y_ind_hi 
                                    
                                    ind_slice = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                                 (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                                 (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)))
                                    
                                    N_parts[task_id] = len(particle_mass[ind_slice]) 
                                    grid_task[x_ind_low:x_ind_hi, y_ind_low:y_ind_hi, z_ind_low:z_ind_hi] = task_id 
                                    
                                    if parameters["include_stars"] == 1: 
                                        if parameters["smooth_stars"] == 1: 
                                            ind_slice_star = ((particle_star_coords[:, 0] > 
                                                               (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 0] < 
                                                               (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                              (particle_star_coords[:, 1] > 
                                                               (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 1] < 
                                                               (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                              (particle_star_coords[:, 2] > 
                                                               (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 2] < 
                                                               (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)))
                                        else: 
                                            ind_slice_star = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0))))
                                            
                                        N_parts_star[task_id] = len(particle_star_mass[ind_slice_star]) 

                                    # Right side 
                                    y_ind_low = y_ind_low_chunk[j + 1] 
                                    y_ind_hi = y_ind_hi_chunk[j + 1] 
                                    task_id = (i * N_chunk[1] * N_chunk[2]) + ((j + 1) * N_chunk[2]) + k 
                                    y_ind_low_task[task_id] = y_ind_low 
                                    y_ind_hi_task[task_id] = y_ind_hi 
                                    
                                    ind_slice = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                                 (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                                 (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)))
                                
                                    N_parts[task_id] = len(particle_mass[ind_slice]) 
                                    grid_task[x_ind_low:x_ind_hi, y_ind_low:y_ind_hi, z_ind_low:z_ind_hi] = task_id 

                                    if parameters["include_stars"] == 1: 
                                        if parameters["smooth_stars"] == 1: 
                                            ind_slice_star = ((particle_star_coords[:, 0] > 
                                                               (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 0] < 
                                                               (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                              (particle_star_coords[:, 1] > 
                                                               (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 1] < 
                                                               (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                              (particle_star_coords[:, 2] > 
                                                               (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 2] < 
                                                               (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)))
                                        else: 
                                            ind_slice_star = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0))))

                                        N_parts_star[task_id] = len(particle_star_mass[ind_slice_star]) 

                    # z-direction 
                    for k in range(N_chunk[2] - 1): 
                        # Determine whether the left or right 
                        # side of the boundary has the highest 
                        # maximum number of particles. 
                        N_max_left = 0 
                        N_max_right = 0 
                        for i in range(N_chunk[0]): 
                            for j in range(N_chunk[1]): 
                                task_id_left = (i * N_chunk[1] * N_chunk[2]) + (j * N_chunk[2]) + k 
                                task_id_right = (i * N_chunk[1] * N_chunk[2]) + (j * N_chunk[2]) + k + 1
                                if N_parts[task_id_left] > N_max_left: 
                                    N_max_left = N_parts[task_id_left] 
                                if N_parts[task_id_right] > N_max_right: 
                                    N_max_right = N_parts[task_id_right] 
                                if parameters["include_stars"] == 1: 
                                    if N_parts_star[task_id_left] > N_max_left: 
                                        N_max_left = N_parts_star[task_id_left] 
                                    if N_parts_star[task_id_right] > N_max_right: 
                                        N_max_right = N_parts_star[task_id_right] 

                        if N_max_left < N_max_right: 
                            # Shift boundary right, if able 
                            if z_ind_hi_chunk[k + 1] - z_ind_low_chunk[k + 1] <= 1: 
                                continue 
                            boundary_shift = 1 
                        else: 
                            # Shift boundary left, if able 
                            if z_ind_hi_chunk[k] - z_ind_low_chunk[k] <= 1: 
                                continue 
                            boundary_shift = -1 

                        # Test whether shifting the boundary 
                        # reduces the max particles. 
                        N_max_new_left = 0 
                        N_max_new_right = 0
                        for i in range(N_chunk[0]): 
                            for j in range(N_chunk[1]): 
                                x_ind_low = x_ind_low_chunk[i] 
                                x_ind_hi = x_ind_hi_chunk[i] 
                                y_ind_low = y_ind_low_chunk[j] 
                                y_ind_hi = y_ind_hi_chunk[j] 
                                
                                # Left side 
                                z_ind_low = z_ind_low_chunk[k] 
                                z_ind_hi = z_ind_hi_chunk[k] + boundary_shift
                                
                                ind_slice = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                             (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                             (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)))
                            
                                N_parts_left = len(particle_mass[ind_slice]) 
                                if N_parts_left > N_max_new_left: 
                                    N_max_new_left = N_parts_left 
                        
                                if parameters["include_stars"] == 1: 
                                    if parameters["smooth_stars"] == 1: 
                                        ind_slice_star = ((particle_star_coords[:, 0] > 
                                                           (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 0] < 
                                                           (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                          (particle_star_coords[:, 1] > 
                                                           (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 1] < 
                                                           (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                          (particle_star_coords[:, 2] > 
                                                           (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 2] < 
                                                           (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)))
                                    else: 
                                        ind_slice_star = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0))))
                                        
                                    N_parts_star_left = len(particle_star_mass[ind_slice_star]) 

                                    if N_parts_star_left > N_max_new_left: 
                                        N_max_new_left = N_parts_star_left 

                                # Right side 
                                z_ind_low = z_ind_low_chunk[k + 1] + boundary_shift
                                z_ind_hi = z_ind_hi_chunk[k + 1] 
                                
                                ind_slice = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                             (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                             (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                             (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)))
                                
                                N_parts_right = len(particle_mass[ind_slice]) 
                                if N_parts_right > N_max_new_right: 
                                    N_max_new_right = N_parts_right 

                                if parameters["include_stars"] == 1: 
                                    if parameters["smooth_stars"] == 1: 
                                        ind_slice_star = ((particle_star_coords[:, 0] > 
                                                           (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 0] < 
                                                           (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                          (particle_star_coords[:, 1] > 
                                                           (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 1] < 
                                                         (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                          (particle_star_coords[:, 2] > 
                                                           (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                          (particle_star_coords[:, 2] < 
                                                           (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)))
                                    else: 
                                        ind_slice_star = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0))) & 
                                                          (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0))))
                                        
                                    N_parts_star_right = len(particle_star_mass[ind_slice_star]) 

                                    if N_parts_star_right > N_max_new_right: 
                                        N_max_new_right = N_parts_star_right 

                        if max([N_max_new_left, N_max_new_right]) < max([N_max_left, N_max_right]): 
                            # This shift has indeed reduced the 
                            # maximum particle load. 
                            accept_adjustment = 1 
                            
                            # Update boundaries and particles numbers 
                            z_ind_hi_chunk[k] += boundary_shift 
                            z_ind_low_chunk[k + 1] += boundary_shift 
                            for i in range(N_chunk[0]): 
                                for j in range(N_chunk[1]): 
                                    x_ind_low = x_ind_low_chunk[i] 
                                    x_ind_hi = x_ind_hi_chunk[i] 
                                    y_ind_low = y_ind_low_chunk[j] 
                                    y_ind_hi = y_ind_hi_chunk[j] 
                                    
                                    # Left side 
                                    z_ind_low = z_ind_low_chunk[k] 
                                    z_ind_hi = z_ind_hi_chunk[k] 
                                    task_id = (i * N_chunk[1] * N_chunk[2]) + (j * N_chunk[2]) + k 
                                    z_ind_low_task[task_id] = z_ind_low 
                                    z_ind_hi_task[task_id] = z_ind_hi 
                                
                                    ind_slice = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                                 (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                                 (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)))
                                    
                                    N_parts[task_id] = len(particle_mass[ind_slice]) 
                                    grid_task[x_ind_low:x_ind_hi, y_ind_low:y_ind_hi, z_ind_low:z_ind_hi] = task_id 

                                    if parameters["include_stars"] == 1: 
                                        if parameters["smooth_stars"] == 1: 
                                            ind_slice_star = ((particle_star_coords[:, 0] > 
                                                               (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 0] < 
                                                               (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                              (particle_star_coords[:, 1] > 
                                                               (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 1] < 
                                                               (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                              (particle_star_coords[:, 2] > 
                                                               (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 2] < 
                                                               (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)))
                                        else: 
                                            ind_slice_star = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0))))
                                        
                                        N_parts_star[task_id] = len(particle_star_mass[ind_slice_star]) 

                                    # Right side 
                                    z_ind_low = z_ind_low_chunk[k + 1] 
                                    z_ind_hi = z_ind_hi_chunk[k + 1] 
                                    task_id = (i * N_chunk[1] * N_chunk[2]) + (j * N_chunk[2]) + k + 1 
                                    z_ind_low_task[task_id] = z_ind_low 
                                    z_ind_hi_task[task_id] = z_ind_hi 
                                    
                                    ind_slice = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                                 (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                                 (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                                 (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)))
                                    
                                    N_parts[task_id] = len(particle_mass[ind_slice]) 
                                    grid_task[x_ind_low:x_ind_hi, y_ind_low:y_ind_hi, z_ind_low:z_ind_hi] = task_id 
                                    
                                    if parameters["include_stars"] == 1: 
                                        if parameters["smooth_stars"] == 1: 
                                            ind_slice_star = ((particle_star_coords[:, 0] > 
                                                               (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 0] < 
                                                               (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                              (particle_star_coords[:, 1] > 
                                                               (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 1] < 
                                                               (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                                              (particle_star_coords[:, 2] > 
                                                               (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                                              (particle_star_coords[:, 2] < 
                                                               (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)))
                                        else: 
                                            ind_slice_star = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0))) & 
                                                              (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0))))

                                        N_parts_star[task_id] = len(particle_star_mass[ind_slice_star]) 

                    print("Adaptive domain iteration %d:" % (domain_iter, )) 
                    print("Domain sizes in x: min = %d cells, max = %d cells." % (min(x_ind_hi_chunk - x_ind_low_chunk), max(x_ind_hi_chunk - x_ind_low_chunk))) 
                    print("Domain sizes in y: min = %d cells, max = %d cells." % (min(y_ind_hi_chunk - y_ind_low_chunk), max(y_ind_hi_chunk - y_ind_low_chunk))) 
                    print("Domain sizes in z: min = %d cells, max = %d cells." % (min(z_ind_hi_chunk - z_ind_low_chunk), max(z_ind_hi_chunk - z_ind_low_chunk))) 
                    sys.stdout.flush() 
                    domain_iter += 1 
                                
                print("Adaptive domain decomposition complete") 
                print("Domain sizes in x: min = %d cells, max = %d cells." % (min(x_ind_hi_chunk - x_ind_low_chunk), max(x_ind_hi_chunk - x_ind_low_chunk))) 
                print("Domain sizes in y: min = %d cells, max = %d cells." % (min(y_ind_hi_chunk - y_ind_low_chunk), max(y_ind_hi_chunk - y_ind_low_chunk))) 
                print("Domain sizes in z: min = %d cells, max = %d cells." % (min(z_ind_hi_chunk - z_ind_low_chunk), max(z_ind_hi_chunk - z_ind_low_chunk))) 
                sys.stdout.flush() 

            # Write domain decomposition 
            # data to restart file. 
            restart_domain_data = []
            restart_domain_data.append(x_ind_low_task) 
            restart_domain_data.append(x_ind_hi_task) 
            restart_domain_data.append(y_ind_low_task) 
            restart_domain_data.append(y_ind_hi_task) 
            restart_domain_data.append(z_ind_low_task) 
            restart_domain_data.append(z_ind_hi_task) 
            restart_domain_data.append(x_ind_low_chunk)
            restart_domain_data.append(x_ind_hi_chunk)
            restart_domain_data.append(y_ind_low_chunk)
            restart_domain_data.append(y_ind_hi_chunk)
            restart_domain_data.append(z_ind_low_chunk)
            restart_domain_data.append(z_ind_hi_chunk)
            restart_domain_data.append(N_parts)
            restart_domain_data.append(grid_task.ravel(order = 'C'))
            if parameters["include_stars"] == 1: 
                restart_domain_data.append(N_parts_star)
        
            write_restart_domain(rank, restart_domain_data) 

        # Determine numbers of particles in the 
        # box, and the min and max in each domain. 
        ind_box = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[0] - (delta_grid / 2.0) - particle_hsml)) & 
                   (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[-1] + (delta_grid / 2.0) + particle_hsml)) &
                   (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[0] - (delta_grid / 2.0) - particle_hsml)) & 
                   (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[-1] + (delta_grid / 2.0) + particle_hsml)) &
                   (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[0] - (delta_grid / 2.0) - particle_hsml)) & 
                   (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[-1] + (delta_grid / 2.0) + particle_hsml)))

        print("%d gas particles total, %d gas particles within the box, with %d to %d gas particle per MPI task" % (len(particle_mass), len(particle_mass[ind_box]), min(N_parts), max(N_parts))) 

        if parameters["include_stars"] == 1: 
            ind_star_box = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[0] - (delta_grid / 2.0) - particle_star_hsml)) & 
                            (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[-1] + (delta_grid / 2.0) + particle_star_hsml)) &
                            (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[0] - (delta_grid / 2.0) - particle_star_hsml)) & 
                            (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[-1] + (delta_grid / 2.0) + particle_star_hsml)) &
                            (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[0] - (delta_grid / 2.0) - particle_star_hsml)) & 
                            (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[-1] + (delta_grid / 2.0) + particle_star_hsml)))

            print("%d star particles total, %d star particles within the box, with %d to %d star particle per MPI task" % (len(particle_star_mass), len(particle_star_mass[ind_star_box]), min(N_parts_star), max(N_parts_star))) 

        sys.stdout.flush()

        if parameters['Multifiles'] == 0: #niranjan: adding this criterion
            h5file.close() 
    else: 
        N_parts = np.empty(N_task, dtype = np.int) 
        x_ind_low_task = np.empty(N_task, dtype = np.int) 
        x_ind_hi_task = np.empty(N_task, dtype = np.int) 
        y_ind_low_task = np.empty(N_task, dtype = np.int) 
        y_ind_hi_task = np.empty(N_task, dtype = np.int) 
        z_ind_low_task = np.empty(N_task, dtype = np.int) 
        z_ind_hi_task = np.empty(N_task, dtype = np.int) 

        if parameters["include_stars"] == 1: 
            N_parts_star = np.empty(N_task, dtype = np.int) 
        
    comm.Bcast(N_parts, root = 0) 
    comm.Bcast(x_ind_low_task, root = 0) 
    comm.Bcast(x_ind_hi_task, root = 0) 
    comm.Bcast(y_ind_low_task, root = 0) 
    comm.Bcast(y_ind_hi_task, root = 0) 
    comm.Bcast(z_ind_low_task, root = 0) 
    comm.Bcast(z_ind_hi_task, root = 0) 

    if parameters["include_stars"] == 1: 
        comm.Bcast(N_parts_star, root = 0) 

    comm.Barrier() 

    if rank == 0: 
        for i in range(1, N_task):
            comm.Send((grid_task.ravel(order = 'C')).copy(), dest = i, tag = i) 
    else: 
        grid_task = np.empty(parameters["n_cell_base"] * parameters["n_cell_base"] * parameters["n_cell_base"], dtype = np.int) 
        comm.Recv(grid_task, source = 0, tag = rank) 

    comm.Barrier() 

    if rank != 0: 
        grid_task = grid_task.reshape((parameters["n_cell_base"], parameters["n_cell_base"], parameters["n_cell_base"]), order = 'C') 

    if rank == 0: 
        print("Broadcasting particle data") 
        sys.stdout.flush() 

    if parameters["include_stars"] == 1: 
        N_data_arrays = 13 
    else: 
        N_data_arrays = 9 

    # Send particles to their respective tasks 
    if rank == 0:  
        for i in range(N_chunk[0]): 
            for j in range(N_chunk[1]): 
                for k in range(N_chunk[2]): 
                    task_id = (i * N_chunk[1] * N_chunk[2]) + (j * N_chunk[2]) + k

                    x_ind_low = x_ind_low_task[task_id] 
                    x_ind_hi = x_ind_hi_task[task_id] 
                    y_ind_low = y_ind_low_task[task_id] 
                    y_ind_hi = y_ind_hi_task[task_id] 
                    z_ind_low = z_ind_low_task[task_id] 
                    z_ind_hi = z_ind_hi_task[task_id] 

                    ind_slice = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                 (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                 (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                 (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                 (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                 (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)))

                    if parameters["include_stars"] == 1: 
                        ind_slice_star = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                          (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                          (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                          (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) & 
                                          (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                          (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)))
                    
                    if task_id == 0: 
                        particle_coords_task = particle_coords[ind_slice, :].copy() 
                        particle_hsml_task = particle_hsml[ind_slice].copy() 
                        particle_mass_task = particle_mass[ind_slice].copy() 
                        particle_velocity_task = particle_velocity[ind_slice, :].copy() 
                        particle_chem_task = particle_chem[ind_slice, :].copy() 
                        particle_u_task = particle_u[ind_slice].copy() 
                        particle_mu_task = particle_mu[ind_slice].copy() 
                        particle_Z_task = particle_Z[ind_slice, :].copy() #niranjan 2022: adding [,:], 
                        particle_rho_task = particle_rho[ind_slice].copy() 

                        if parameters["include_stars"] == 1: 
                            particle_star_age_task = particle_star_age[ind_slice_star].copy() 
                            particle_star_coords_task = particle_star_coords[ind_slice_star, :].copy()
                            particle_star_mass_task = particle_star_mass[ind_slice_star].copy() 
                            particle_star_hsml_task = particle_star_hsml[ind_slice_star].copy() 
                    else: 
                        comm.Send((particle_coords[ind_slice, :].copy()).ravel(order = 'C'), dest = task_id, tag = N_data_arrays * task_id) 
                        comm.Send(particle_hsml[ind_slice].copy(), dest = task_id, tag = (N_data_arrays * task_id) + 1) 
                        comm.Send(particle_mass[ind_slice].copy(), dest = task_id, tag = (N_data_arrays * task_id) + 2) 
                        comm.Send((particle_velocity[ind_slice, :].copy()).ravel(order = 'C'), dest = task_id, tag = (N_data_arrays * task_id) + 3) 
                        comm.Send((particle_chem[ind_slice, :].copy()).ravel(order = 'C'), dest = task_id, tag = (N_data_arrays * task_id) + 4) 
                        comm.Send(particle_u[ind_slice].copy(), dest = task_id, tag = (N_data_arrays * task_id) + 5) 
                        comm.Send(particle_mu[ind_slice].copy(), dest = task_id, tag = (N_data_arrays * task_id) + 6) 
                        comm.Send((particle_Z[ind_slice, :].copy()).ravel(order = 'C'), dest = task_id, tag = (N_data_arrays * task_id) + 7) 
                        comm.Send(particle_rho[ind_slice].copy(), dest = task_id, tag = (N_data_arrays * task_id) + 8) 

                        if parameters["include_stars"] == 1: 
                            comm.Send(particle_star_age[ind_slice_star].copy(), dest = task_id, tag = (N_data_arrays * task_id) + 9) 
                            comm.Send((particle_star_coords[ind_slice_star, :].copy()).ravel(order = 'C'), dest = task_id, tag = (N_data_arrays * task_id) + 10) 
                            comm.Send(particle_star_mass[ind_slice_star].copy(), dest = task_id, tag = (N_data_arrays * task_id) + 11) 
                            comm.Send(particle_star_hsml[ind_slice_star].copy(), dest = task_id, tag = (N_data_arrays * task_id) + 12) 
    else: 
        particle_coords_task = np.empty((N_parts[rank] * 3), dtype = np.float64) 
        particle_hsml_task = np.empty(N_parts[rank], dtype = np.float64) 
        particle_mass_task = np.empty(N_parts[rank], dtype = np.float64) 
        particle_velocity_task = np.empty((N_parts[rank] * 3), dtype = np.float64) 
        particle_chem_task = np.empty((N_parts[rank] * 157), dtype = np.float64) 
        particle_u_task = np.empty(N_parts[rank], dtype = np.float64) 
        particle_mu_task = np.empty(N_parts[rank], dtype = np.float64) 
        particle_Z_task = np.empty((N_parts[rank] * 11), dtype = np.float64) 
        particle_rho_task = np.empty(N_parts[rank], dtype = np.float64) 

        if parameters["include_stars"] == 1: 
            particle_star_age_task = np.empty(N_parts_star[rank], dtype = np.float64) 
            particle_star_coords_task = np.empty(N_parts_star[rank] * 3, dtype = np.float64) 
            particle_star_mass_task = np.empty(N_parts_star[rank], dtype = np.float64) 
            particle_star_hsml_task = np.empty(N_parts_star[rank], dtype = np.float64) 

        comm.Recv(particle_coords_task, source = 0, tag = N_data_arrays * rank) 
        comm.Recv(particle_hsml_task, source = 0, tag = (N_data_arrays * rank) + 1) 
        comm.Recv(particle_mass_task, source = 0, tag = (N_data_arrays * rank) + 2) 
        comm.Recv(particle_velocity_task, source = 0, tag = (N_data_arrays * rank) + 3) 
        comm.Recv(particle_chem_task, source = 0, tag = (N_data_arrays * rank) + 4) 
        comm.Recv(particle_u_task, source = 0, tag = (N_data_arrays * rank) + 5) 
        comm.Recv(particle_mu_task, source = 0, tag = (N_data_arrays * rank) + 6) 
        comm.Recv(particle_Z_task, source = 0, tag = (N_data_arrays * rank) + 7) 
        comm.Recv(particle_rho_task, source = 0, tag = (N_data_arrays * rank) + 8) 

        if parameters["include_stars"] == 1: 
            comm.Recv(particle_star_age_task, source = 0, tag = (N_data_arrays * rank) + 9) 
            comm.Recv(particle_star_coords_task, source = 0, tag = (N_data_arrays * rank) + 10) 
            comm.Recv(particle_star_mass_task, source = 0, tag = (N_data_arrays * rank) + 11) 
            comm.Recv(particle_star_hsml_task, source = 0, tag = (N_data_arrays * rank) + 12) 
        
    comm.Barrier() 

    if rank == 0: 
        print("Finished broadcasting particle data") 
        sys.stdout.flush() 
    else: 
        particle_coords_task = particle_coords_task.reshape((-1, 3), order = 'C') 
        particle_velocity_task = particle_velocity_task.reshape((-1, 3), order = 'C') 
        particle_chem_task = particle_chem_task.reshape((-1, 157), order = 'C') 
        particle_Z_task = particle_Z_task.reshape((-1, 11), order = 'C') 

        if parameters["include_stars"] == 1: 
            particle_star_coords_task = particle_star_coords_task.reshape((-1, 3), order = 'C') 

    # Derived quantities 
    XH_task = 1.0 - (particle_Z_task[:, 0] + particle_Z_task[:, 1]) 
    particle_temperature_task = (2.0 / 3.0) * particle_mu_task * 1.67e-24 * particle_u_task / 1.38e-16  # K 
    particle_nH_task = particle_rho_task * XH_task * (parameters["unit_density"] / 1.67e-24) 

    if parameters["include_stars"] == 1: 
        particle_star_age_task[(particle_star_age_task < 0.001)] = 0.001 
        log_star_age_task = np.log10(particle_star_age_task) 

    # Species to extract 
    particle_mass_species_task = np.zeros((N_parts[rank], N_species), dtype = np.float64) 
    for i in range(N_species): 
        if species_list[i] == "pH2": 
            particle_mass_species_task[:, i] = particle_mass_task * particle_chem_task[:, rc.chimes_dict["H2"]] * 0.25 * XH_task * atomic_mass_list[i] 
        elif species_list[i] == "oH2": 
            particle_mass_species_task[:, i] = particle_mass_task * particle_chem_task[:, rc.chimes_dict["H2"]] * 0.75 * XH_task * atomic_mass_list[i] 
        elif species_list[i] == "tot": 
            particle_mass_species_task[:, i] = particle_mass_task * XH_task 
        else: 
            particle_mass_species_task[:, i] = particle_mass_task * particle_chem_task[:, rc.chimes_dict[species_list[i]]] * XH_task * atomic_mass_list[i] 

    if parameters["verbose_log_files"] == 1: 
        write_task_log(rank, "%d gas particles" % (N_parts[rank], ))
        if parameters["include_stars"] == 1: 
            write_task_log(rank, "%d star particles" % (N_parts_star[rank], ))

    # Determine which chunk of the base AMR grid is this 
    # task's domain. 
    for i in range(N_chunk[0]): 
        for j in range(N_chunk[1]): 
            for k in range(N_chunk[2]): 
                task_id = (i * N_chunk[1] * N_chunk[2]) + (j * N_chunk[2]) + k

                if rank == task_id: 
                    nx_task = x_ind_hi_task[task_id] - x_ind_low_task[task_id] 
                    ny_task = y_ind_hi_task[task_id] - y_ind_low_task[task_id] 
                    nz_task = z_ind_hi_task[task_id] - z_ind_low_task[task_id] 
                    
                    break 
                    

    if os.path.exists("restart_amr_grid_%d" % (rank, )): 
        if rank == 0: 
            print("Reading AMR grid of nodes from restart file") 
            sys.stdout.flush() 
        if parameters["verbose_log_files"] == 1: 
            write_task_log(rank, "Reading AMR grid of nodes from restart file")

        fd = open("restart_amr_grid_%d" % (rank, ), "rb") 
        amr_nodes_task = np.ndarray((nx_task, ny_task, nz_task), dtype = np.object) 
        for k in range(nz_task): 
            for j in range(ny_task): 
                for i in range(nx_task): 
                    amr_nodes_task[i, j, k] = node(0, N_species, emitter_flag_list, atomic_mass_list) 
                    amr_nodes_task[i, j, k].read_restart_node(fd) 
    else: 
        # Create AMR grid of nodes for this task's domain 
        if rank == 0: 
            print("creating AMR grid of nodes") 
            sys.stdout.flush() 
        if parameters["verbose_log_files"] == 1: 
            write_task_log(rank, "creating AMR grid of nodes")

        if parameters["refinement_scheme"] == 1: 
            # The N_age_bins_refine parameter determines 
            # which stellar age bins are included when 
            # refining the AMR grid. 
            ind_star_age = (log_star_age_task < log_stellar_age_Myr_bin_max[parameters["N_age_bins_refine"] - 1])

        amr_nodes_task = np.ndarray((nx_task, ny_task, nz_task), dtype = np.object) 
        count = 1 
        for k in range(nz_task): 
            for j in range(ny_task): 
                for i in range(nx_task): 
                    if rank == 0: 
                        print("%d of %d" % (count, nx_task * ny_task * nz_task)) 
                        sys.stdout.flush() 
                    if parameters["verbose_log_files"] == 1: 
                        write_task_log(rank, "%d of %d" % (count, nx_task * ny_task * nz_task))
                    count += 1 
                    amr_nodes_task[i, j, k] = node(0, N_species, emitter_flag_list, atomic_mass_list) 
                    amr_nodes_task[i, j, k].pos[0] = parameters["centre_x"] + grid_array[i + x_ind_low_task[rank]] 
                    amr_nodes_task[i, j, k].pos[1] = parameters["centre_y"] + grid_array[j + y_ind_low_task[rank]] 
                    amr_nodes_task[i, j, k].pos[2] = parameters["centre_z"] + grid_array[k + z_ind_low_task[rank]] 
                    amr_nodes_task[i, j, k].width = parameters["box_size"] / parameters["n_cell_base"]

                    if parameters["refinement_scheme"] == 0: 
                        part_ind, part_ind_star = amr_nodes_task[i, j, k].find_particles(particle_coords_task, None) 
                        amr_nodes_task[i, j, k].split_cells(particle_coords_task[part_ind, :], None) 
                    elif parameters["refinement_scheme"] == 1: 
                        part_ind, part_ind_star = amr_nodes_task[i, j, k].find_particles(particle_coords_task, particle_star_coords_task[ind_star_age, :])
                        amr_nodes_task[i, j, k].split_cells(particle_coords_task[part_ind, :], particle_star_coords_task[ind_star_age, :][part_ind_star, :]) 
                    else: 
                        raise Exception("refinement scheme %d not recognised. Aborting" % (parameters["refinement_scheme"], )) 

        # Write AMR grid to restart files 
        filename = "restart_amr_grid_%d" % (rank, ) 
        fd = open(filename, "wb") 
        for k in range(nz_task): 
            for j in range(ny_task): 
                for i in range(nx_task): 
                    amr_nodes_task[i, j, k].write_restart_node(fd) 
        fd.close() 

    if os.path.exists("restart_kernel_%d" % (rank, )): 
        if rank == 0: 
            print("Reading kernel weights from restart file") 
            sys.stdout.flush() 
        if parameters["verbose_log_files"] == 1: 
            write_task_log(rank, "Reading kernel weights from restart file")

        restart_kernel_data = read_restart_kernel(rank, N_parts[rank], N_parts_star[rank]) 
        sum_wk_task = restart_kernel_data[0] 
        if parameters["include_stars"] == 1 and parameters["smooth_stars"] == 1: 
            sum_wk_star_task = restart_kernel_data[1] 
    else: 
        # Now go through and compute kernel weights for each particle             
        if rank == 0: 
            print("Computing kernel weights for gas particles") 
            sys.stdout.flush() 
        if parameters["verbose_log_files"] == 1: 
            write_task_log(rank, "Computing kernel weights for gas particles")
        
        wk_task = np.zeros(len(particle_mass_task), np.float64) 

        count = 1 
        for k in range(nz_task): 
            for j in range(ny_task): 
                for i in range(nx_task): 
                    if rank == 0: 
                        print("%d of %d" % (count, nx_task * ny_task * nz_task)) 
                        sys.stdout.flush() 
                    if parameters["verbose_log_files"] == 1: 
                        write_task_log(rank, "%d of %d" % (count, nx_task * ny_task * nz_task))
                    count += 1 

                    relative_pos = amr_nodes_task[i, j, k].pos - particle_coords_task
                    overlap_size = particle_hsml_task + (amr_nodes_task[i, j, k].width / 2.0) 
                    
                    ind_smooth = ((relative_pos[:, 0] > -overlap_size) & 
                                  (relative_pos[:, 0] < overlap_size) & 
                                  (relative_pos[:, 1] > -overlap_size) & 
                                  (relative_pos[:, 1] < overlap_size) & 
                                  (relative_pos[:, 2] > -overlap_size) & 
                                  (relative_pos[:, 2] < overlap_size)) 
                    wk_task[ind_smooth] += amr_nodes_task[i, j, k].compute_kernel_weights(particle_coords_task[ind_smooth], particle_hsml_task[ind_smooth]) 

        if parameters["include_stars"] == 1 and parameters["smooth_stars"] == 1: 
            if rank == 0: 
                print("Computing kernel weights for star particles") 
                sys.stdout.flush() 
            if parameters["verbose_log_files"] == 1: 
                write_task_log(rank, "Computing kernel weights for star particles")
                    
            wk_star_task = np.zeros(len(particle_star_mass_task), np.float64) 

            count = 1 
            for k in range(nz_task): 
                for j in range(ny_task): 
                    for i in range(nx_task): 
                        if rank == 0: 
                            print("%d of %d" % (count, nx_task * ny_task * nz_task)) 
                            sys.stdout.flush() 
                        if parameters["verbose_log_files"] == 1: 
                            write_task_log(rank, "%d of %d" % (count, nx_task * ny_task * nz_task))
                        count += 1 

                        relative_pos = amr_nodes_task[i, j, k].pos - particle_star_coords_task
                        overlap_size = particle_star_hsml_task + (amr_nodes_task[i, j, k].width / 2.0) 
                    
                        ind_smooth = ((relative_pos[:, 0] > -overlap_size) & 
                                      (relative_pos[:, 0] < overlap_size) & 
                                      (relative_pos[:, 1] > -overlap_size) & 
                                      (relative_pos[:, 1] < overlap_size) & 
                                      (relative_pos[:, 2] > -overlap_size) & 
                                      (relative_pos[:, 2] < overlap_size)) 

                        wk_star_task[ind_smooth] += amr_nodes_task[i, j, k].compute_kernel_weights(particle_star_coords_task[ind_smooth], particle_star_hsml_task[ind_smooth]) 

        if parameters["verbose_log_files"] == 1: 
            write_task_log(rank, "Waiting for other tasks.")

        comm.Barrier() 

        # Now collate all kernel weights on root task 
        if rank == 0: 
            print("Summing kernel weights") 
            sys.stdout.flush() 

        if rank == 0: 
            wk_buffer = [wk_task.copy()] 
            for i in range(1, N_task): 
                wk_buffer.append(np.empty(N_parts[i], dtype = np.float64)) 

        comm.Barrier() 

        if rank == 0: 
            for i in range(1, N_task): 
                comm.Recv(wk_buffer[i], source = i, tag = i) 
        else: 
            comm.Send(wk_task, dest = 0, tag = rank) 

        comm.Barrier() 

        if parameters["include_stars"] == 1 and parameters["smooth_stars"] == 1: 
            if rank == 0: 
                wk_star_buffer = [wk_star_task.copy()] 
                for i in range(1, N_task): 
                    wk_star_buffer.append(np.empty(N_parts_star[i], dtype = np.float64)) 

            comm.Barrier() 

            if rank == 0: 
                for i in range(1, N_task): 
                    comm.Recv(wk_star_buffer[i], source = i, tag = i) 
            else: 
                comm.Send(wk_star_task, dest = 0, tag = rank) 

            comm.Barrier() 

        if rank == 0: 
            # Sum the kernel weights from each task. 
            sum_wk = np.zeros(len(particle_mass), dtype = np.float64) 
        
            for i in range(N_chunk[0]): 
                for j in range(N_chunk[1]): 
                    for k in range(N_chunk[2]): 
                        task_id = (i * N_chunk[1] * N_chunk[2]) + (j * N_chunk[2]) + k
                
                        x_ind_low = x_ind_low_task[task_id] 
                        x_ind_hi = x_ind_hi_task[task_id] 
                        y_ind_low = y_ind_low_task[task_id] 
                        y_ind_hi = y_ind_hi_task[task_id] 
                        z_ind_low = z_ind_low_task[task_id] 
                        z_ind_hi = z_ind_hi_task[task_id] 
                    
                        ind_slice = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                     (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                     (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                     (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                     (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                     (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml))) 

                        sum_wk[ind_slice] += wk_buffer[task_id] 

            if parameters["include_stars"] == 1 and parameters["smooth_stars"] == 1: 
                sum_wk_star = np.zeros(len(particle_star_mass), dtype = np.float64) 
                
                for i in range(N_chunk[0]): 
                    for j in range(N_chunk[1]): 
                        for k in range(N_chunk[2]): 
                            task_id = (i * N_chunk[1] * N_chunk[2]) + (j * N_chunk[2]) + k
        
                            x_ind_low = x_ind_low_task[task_id] 
                            x_ind_hi = x_ind_hi_task[task_id] 
                            y_ind_low = y_ind_low_task[task_id] 
                            y_ind_hi = y_ind_hi_task[task_id] 
                            z_ind_low = z_ind_low_task[task_id] 
                            z_ind_hi = z_ind_hi_task[task_id] 
                            
                            ind_slice = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                         (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) &
                                         (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                         (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) &
                                         (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                         (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml))) 
                        
                            sum_wk_star[ind_slice] += wk_star_buffer[task_id] 

        # Send summed kernel weights back to other tasks 
        if rank == 0: 
            for i in range(N_chunk[0]): 
                for j in range(N_chunk[1]): 
                    for k in range(N_chunk[2]): 
                        task_id = (i * N_chunk[1] * N_chunk[2]) + (j * N_chunk[2]) + k

                        x_ind_low = x_ind_low_task[task_id] 
                        x_ind_hi = x_ind_hi_task[task_id] 
                        y_ind_low = y_ind_low_task[task_id] 
                        y_ind_hi = y_ind_hi_task[task_id] 
                        z_ind_low = z_ind_low_task[task_id] 
                        z_ind_hi = z_ind_hi_task[task_id] 
                    
                        ind_slice = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                     (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                     (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                     (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml)) &
                                     (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_hsml)) & 
                                     (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_hsml))) 
        
                        if task_id == 0: 
                            sum_wk_task = sum_wk[ind_slice].copy() 
                        else: 
                            comm.Send(sum_wk[ind_slice].copy(), dest = task_id, tag = task_id) 
        else: 
            sum_wk_task = np.empty(N_parts[rank], dtype = np.float64) 
            comm.Recv(sum_wk_task, source = 0, tag = rank) 

        comm.Barrier() 

        if parameters["include_stars"] == 1 and parameters["smooth_stars"] == 1: 
            if rank == 0: 
                for i in range(N_chunk[0]): 
                    for j in range(N_chunk[1]): 
                        for k in range(N_chunk[2]): 
                            task_id = (i * N_chunk[1] * N_chunk[2]) + (j * N_chunk[2]) + k
                            
                            x_ind_low = x_ind_low_task[task_id] 
                            x_ind_hi = x_ind_hi_task[task_id] 
                            y_ind_low = y_ind_low_task[task_id] 
                            y_ind_hi = y_ind_hi_task[task_id] 
                            z_ind_low = z_ind_low_task[task_id] 
                            z_ind_hi = z_ind_hi_task[task_id] 
                            
                            ind_slice = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[x_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                         (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[x_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) &
                                         (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[y_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                         (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[y_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml)) &
                                         (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[z_ind_low] - (delta_grid / 2.0) - particle_star_hsml)) & 
                                         (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[z_ind_hi - 1] + (delta_grid / 2.0) + particle_star_hsml))) 
                        
                            if task_id == 0: 
                                sum_wk_star_task = sum_wk_star[ind_slice].copy() 
                            else: 
                                comm.Send(sum_wk_star[ind_slice].copy(), dest = task_id, tag = task_id) 
            else: 
                sum_wk_star_task = np.empty(N_parts_star[rank], dtype = np.float64) 
                comm.Recv(sum_wk_star_task, source = 0, tag = rank) 

            comm.Barrier() 

        # Write summed kernel weights 
        # to restart file. 
        restart_kernel_data = []
        restart_kernel_data.append(sum_wk_task) 
        if parameters["include_stars"] == 1 and parameters["smooth_stars"] == 1: 
            restart_kernel_data.append(sum_wk_star_task) 
        write_restart_kernel(rank, restart_kernel_data) 

    if os.path.exists("restart_densities_%d" % (rank, )): 
        # Read densities from restart file 
        if rank == 0: 
            print("Reading densities from restart file") 
            sys.stdout.flush() 
        if parameters["verbose_log_files"] == 1: 
            write_task_log(rank, "Reading densities from restart file")

        fd = open("restart_densities_%d" % (rank, ), "rb") 
        for k in range(nz_task): 
            for j in range(ny_task): 
                for i in range(nx_task): 
                    amr_nodes_task[i, j, k].read_restart_densities(fd) 
    else: 
        # Compute SPH and stellar densities 
        if rank == 0: 
            if parameters["include_stars"] == 1: 
                print("Computing SPH and stellar densities.") 
                sys.stdout.flush() 
            else: 
                print("Computing SPH densities.") 
                sys.stdout.flush() 
            sys.stdout.flush() 
        if parameters["verbose_log_files"] == 1: 
            if parameters["include_stars"] == 1: 
                write_task_log(rank, "Computing SPH and stellar densities.")
            else: 
                write_task_log(rank, "Computing SPH densities.")

        if parameters["include_H_level_pop"] == 1: 
            elec_index = -1 
            HI_index = -1 
            HII_index = -1 
            for i in range(len(species_list)): 
                if species_list[i] == "elec": 
                    elec_index = i 
                if species_list[i] == "HI": 
                    HI_index = i 
                if species_list[i] == "HII": 
                    HII_index = i 
        
            if elec_index < 0: 
                raise Exception("H level populations enabled, but electrons not included in species list. Aborting.") 
            if HI_index < 0: 
                raise Exception("H level populations enabled, but HI not included in species list. Aborting.") 
            if HII_index < 0: 
                raise Exception("H level populations enabled, but HII not included in species list. Aborting.") 
        
        count = 1 
        for k in range(nz_task): 
            for j in range(ny_task): 
                for i in range(nx_task): 
                    if rank == 0: 
                        print("%d of %d" % (count, nx_task * ny_task * nz_task)) 
                        sys.stdout.flush() 
                    if parameters["verbose_log_files"] == 1: 
                        write_task_log(rank, "%d of %d" % (count, nx_task * ny_task * nz_task))

                    count += 1 

                    relative_pos = amr_nodes_task[i, j, k].pos - particle_coords_task
                    overlap_size = particle_hsml_task + amr_nodes_task[i, j, k].width / 2.0
                    
                    ind_smooth = ((relative_pos[:, 0] > -overlap_size) & 
                                  (relative_pos[:, 0] < overlap_size) & 
                                  (relative_pos[:, 1] > -overlap_size) & 
                                  (relative_pos[:, 1] < overlap_size) & 
                                  (relative_pos[:, 2] > -overlap_size) & 
                                  (relative_pos[:, 2] < overlap_size) & 
                                  (sum_wk_task > 0.0)) 

                    amr_nodes_task[i, j, k].compute_cell_densities(particle_coords_task[ind_smooth], 
                                                                   particle_hsml_task[ind_smooth], 
                                                                   particle_mass_task[ind_smooth], 
                                                                   particle_mass_species_task[ind_smooth], 
                                                                   particle_temperature_task[ind_smooth], 
                                                                   particle_velocity_task[ind_smooth], 
                                                                   particle_Z_task[ind_smooth, 0], 
                                                                   particle_nH_task[ind_smooth], 
                                                                   sum_wk_task[ind_smooth]) 
                    if parameters["include_H_level_pop"] == 1:
                        amr_nodes_task[i, j, k].compute_H_level_populations(elec_index, HI_index, HII_index) 

                    if parameters["include_stars"] == 1: 
                        for bin_idx in range(parameters["N_age_bins"], ): 
                            if bin_idx == 0: 
                                ind_age_bin = (log_star_age_task < log_stellar_age_Myr_bin_max[0])
                            else: 
                                ind_age_bin = ((log_star_age_task >= log_stellar_age_Myr_bin_max[bin_idx - 1]) & (log_star_age_task < log_stellar_age_Myr_bin_max[bin_idx]))
                            
                            if parameters["smooth_stars"] == 1: 
                                relative_pos = amr_nodes_task[i, j, k].pos - particle_star_coords_task[ind_age_bin]
                                overlap_size = particle_star_hsml_task[ind_age_bin] + (amr_nodes_task[i, j, k].width / 2.0)
            
                                ind_smooth = ((relative_pos[:, 0] > -overlap_size) & 
                                              (relative_pos[:, 0] < overlap_size) & 
                                              (relative_pos[:, 1] > -overlap_size) & 
                                              (relative_pos[:, 1] < overlap_size) & 
                                              (relative_pos[:, 2] > -overlap_size) & 
                                              (relative_pos[:, 2] < overlap_size) & 
                                              (sum_wk_star_task[ind_age_bin] > 0.0)) 

                                amr_nodes_task[i, j, k].compute_cell_smoothed_stellar_densities(particle_star_coords_task[ind_age_bin, :][ind_smooth], particle_star_hsml_task[ind_age_bin][ind_smooth], particle_star_mass_task[ind_age_bin][ind_smooth], sum_wk_star_task[ind_age_bin][ind_smooth], bin_idx)

                                # Check for stars that don't overlap the centres
                                # of any cells (i.e. sum_wk_star is zero) and
                                # add them as points into individual cells.
                                ind_zero_wk = (sum_wk_star_task[ind_age_bin] == 0.0) 
                                if sum(ind_zero_wk) > 0: 
                                    amr_nodes_task[i, j, k].compute_cell_stellar_densities(particle_star_coords_task[ind_age_bin, :][ind_zero_wk, :], particle_star_mass_task[ind_age_bin][ind_zero_wk], bin_idx) 

                            else: 
                                amr_nodes_task[i, j, k].compute_cell_stellar_densities(particle_star_coords_task[ind_age_bin, :], particle_star_mass_task[ind_age_bin], bin_idx)

        # Write densities to restart files 
        filename = "restart_densities_%d" % (rank, ) 
        fd = open(filename, "wb") 
        for k in range(nz_task): 
            for j in range(ny_task): 
                for i in range(nx_task): 
                    amr_nodes_task[i, j, k].write_restart_densities(fd) 
        try: 
            fd.close() 
        except OSError: 
            print("Encountered an OS error when trying to close file restart_densities_%d. Continuing." % (rank, )) 
    
    if parameters["verbose_log_files"] == 1: 
        write_task_log(rank, "Waiting for other tasks.")

    comm.Barrier() 

    # Collating the amr_nodes on the root MPI task takes a long time 
    # (I think it is because the node structures are complicated). 
    # Instead, each task will write out its own nodes, but we will go 
    # through each cell of the base grid one at a time.     

    # First, determine number of levels, branches and leaves from each 
    # task, then combine them 
    level_max_task = 1
    branch_max_task = 0 
    leaf_max_task = 0 
    for k in range(nz_task): 
        for j in range(ny_task): 
            for i in range(nx_task): 
                next_level, next_branch, next_leaf = amr_nodes_task[i, j, k].determine_levels() 
                branch_max_task += next_branch 
                leaf_max_task += next_leaf 
                if next_level > level_max_task: 
                    level_max_task = next_level 

    if rank == 0: 
        level_max_buffer = [] 
        branch_max_buffer = [] 
        leaf_max_buffer = [] 
        
        for i in range(1, N_task): 
            level_max_buffer.append(np.empty(1, dtype = np.int)) 
            branch_max_buffer.append(np.empty(1, dtype = np.int)) 
            leaf_max_buffer.append(np.empty(1, dtype = np.int)) 
        
        for i in range(1, N_task):
            comm.Recv(level_max_buffer[i - 1], source = i, tag = (i * 3)) 
            comm.Recv(branch_max_buffer[i - 1], source = i, tag = (i * 3) + 1) 
            comm.Recv(leaf_max_buffer[i - 1], source = i, tag = (i * 3) + 2) 
    else: 
        comm.Send(np.array([level_max_task]), dest = 0, tag = (rank * 3)) 
        comm.Send(np.array([branch_max_task]), dest = 0, tag = (rank * 3) + 1) 
        comm.Send(np.array([leaf_max_task]), dest = 0, tag = (rank * 3) + 2) 

    comm.Barrier() 
            
    if rank == 0: 
        level_max = level_max_task 
        branch_max = branch_max_task 
        leaf_max = leaf_max_task 

        for i in range(1, N_task): 
            branch_max += branch_max_buffer[i - 1][0] 
            leaf_max += leaf_max_buffer[i - 1][0] 
            if level_max_buffer[i - 1][0] > level_max: 
                level_max = level_max_buffer[i - 1][0] 

        print("level_max = %d, branch_max = %d, leaf_max = %d" % (level_max, branch_max, leaf_max)) 
        sys.stdout.flush() 

        print("Writing file headers") 
        sys.stdout.flush() 

        # Open output files and write the first few 'header' lines 
        f_amr = open("amr_grid.inp", "w") 
        f_turb = open("microturbulence.binp", "wb") 
        f_dust = open("dust_density.binp", "wb") 
        f_dust_T = open("dust_temperature_zero.dat", "w") 

        if parameters["include_stars"] == 1: 
            f_star = open("stellarsrc_density.binp", "wb") 
        else: 
            f_star = None 

        f_species_list = [] 
        f_T_list = [] 
        f_nH_list = [] 
        f_vel_list = [] 
        for i in range(N_species): 
            species_filename = "numberdens_%s.binp" % (species_list[i], ) 
            f_species_list.append(open(species_filename, "wb")) 

            if emitter_flag_list[i] == 1: 
                T_filename = "gas_temperature_%s.binp" % (species_list[i], ) 
                f_T_list.append(open(T_filename, "wb")) 

                nH_filename = "gas_nHtot_%s.binp" % (species_list[i], )
                f_nH_list.append(open(nH_filename, "wb")) 

                vel_filename = "gas_velocity_%s.binp" % (species_list[i], ) 
                f_vel_list.append(open(vel_filename, "wb")) 

        write_headers(f_amr, 
                      f_species_list, 
                      f_T_list, 
                      f_nH_list, 
                      f_vel_list, 
                      f_turb, 
                      f_dust, 
                      f_dust_T, 
                      f_star, 
                      grid_array, 
                      grid_array[1] - grid_array[0], 
                      level_max, branch_max, leaf_max)

        # Close output files 
        f_amr.close() 
        f_turb.close() 
        f_dust.close() 
        f_dust_T.close() 

        for my_file in f_species_list: 
            my_file.close() 
        for my_file in f_T_list: 
            my_file.close() 
        for my_file in f_nH_list: 
            my_file.close() 
        for my_file in f_vel_list: 
            my_file.close() 

        if parameters["include_stars"] == 1: 
            f_star.close() 

    comm.Barrier() 

    # Test that total mass of each species in the AMR grid agrees 
    # with the total mass in the particles 
    if rank == 0: 
        print("Testing total species masses in grid versus particles") 
        sys.stdout.flush() 

    grid_species_mass_task = np.zeros(N_species, dtype = np.float64) 
    for species_index in range(N_species): 
        for k in range(nz_task): 
            for j in range(ny_task): 
                for i in range(nx_task): 
                    grid_species_mass_task[species_index] += amr_nodes_task[i, j, k].compute_total_species_mass(species_index)  

    if rank == 0: 
        grid_species_mass_buffer = [] 
        for i in range(1, N_task): 
            grid_species_mass_buffer.append(np.empty(N_species, dtype = np.float64)) 
            
        for i in range(1, N_task): 
            comm.Recv(grid_species_mass_buffer[i - 1], source = i, tag = i) 
    else: 
        comm.Send(grid_species_mass_task, dest = 0, tag = rank) 

    comm.Barrier() 
    
    if rank == 0: 
        grid_species_total = grid_species_mass_task.copy() 
        for i in range(1, N_task): 
            grid_species_total += grid_species_mass_buffer[i - 1] 
        
        ind_box = ((particle_coords[:, 0] > (parameters["centre_x"] + grid_array[0] - (delta_grid / 2.0) - particle_hsml)) & 
                   (particle_coords[:, 0] < (parameters["centre_x"] + grid_array[-1] + (delta_grid / 2.0) + particle_hsml)) &
                   (particle_coords[:, 1] > (parameters["centre_y"] + grid_array[0] - (delta_grid / 2.0) - particle_hsml)) & 
                   (particle_coords[:, 1] < (parameters["centre_y"] + grid_array[-1] + (delta_grid / 2.0) + particle_hsml)) &
                   (particle_coords[:, 2] > (parameters["centre_z"] + grid_array[0] - (delta_grid / 2.0) - particle_hsml)) & 
                   (particle_coords[:, 2] < (parameters["centre_z"] + grid_array[-1] + (delta_grid / 2.0) + particle_hsml)))

        for i in range(N_species): 
            if species_list[i] == "pH2": 
                particle_species_total = np.sum(particle_mass[ind_box] * particle_chem[ind_box, rc.chimes_dict["H2"]] * 0.25 * (1.0 - (particle_Z[ind_box, 0] + particle_Z[ind_box, 1])) * atomic_mass_list[i]) 
            elif species_list[i] == "oH2": 
                particle_species_total = np.sum(particle_mass[ind_box] * particle_chem[ind_box, rc.chimes_dict["H2"]] * 0.75 * (1.0 - (particle_Z[ind_box, 0] + particle_Z[ind_box, 1])) * atomic_mass_list[i]) 
            elif species_list[i] == "tot": 
                particle_species_total = np.sum(particle_mass[ind_box] * (1.0 - (particle_Z[ind_box, 0] + particle_Z[ind_box, 1]))) 
            else: 
                particle_species_total = np.sum(particle_mass[ind_box] * particle_chem[ind_box, rc.chimes_dict[species_list[i]]] * (1.0 - (particle_Z[ind_box, 0] + particle_Z[ind_box, 1])) * atomic_mass_list[i]) 

            print("Species %s: M_grid = %.4e Msol, M_particles = %.4e" % (species_list[i], grid_species_total[i], particle_species_total)) 
            sys.stdout.flush() 
            
    comm.Barrier() 

    if parameters["include_stars"] == 1: 
        # Test that the total stellar mass in the AMR grid agrees 
        # with the total mass in the star particles 
        if rank == 0: 
            print("Testing total stellar masses in grid versus particles") 
            sys.stdout.flush() 

        grid_stellar_mass_task = np.zeros(parameters["N_age_bins"], dtype = np.float64) 
        for age_index in range(parameters["N_age_bins"]): 
            for k in range(nz_task): 
                for j in range(ny_task): 
                    for i in range(nx_task): 
                        grid_stellar_mass_task[age_index] += amr_nodes_task[i, j, k].compute_total_stellar_mass(age_index)  

        if rank == 0: 
            grid_stellar_mass_buffer = [] 
            for i in range(1, N_task): 
                grid_stellar_mass_buffer.append(np.empty(parameters["N_age_bins"], dtype = np.float64)) 
            
            for i in range(1, N_task): 
                comm.Recv(grid_stellar_mass_buffer[i - 1], source = i, tag = i) 
        else: 
            comm.Send(grid_stellar_mass_task, dest = 0, tag = rank) 

        comm.Barrier() 
    
        if rank == 0: 
            grid_stellar_total = grid_stellar_mass_task.copy() 
            for i in range(1, N_task): 
                grid_stellar_total += grid_stellar_mass_buffer[i - 1] 
        
            ind_box = ((particle_star_coords[:, 0] > (parameters["centre_x"] + grid_array[0] - (delta_grid / 2.0) - particle_star_hsml)) & 
                       (particle_star_coords[:, 0] < (parameters["centre_x"] + grid_array[-1] + (delta_grid / 2.0) + particle_star_hsml)) &
                       (particle_star_coords[:, 1] > (parameters["centre_y"] + grid_array[0] - (delta_grid / 2.0) - particle_star_hsml)) & 
                       (particle_star_coords[:, 1] < (parameters["centre_y"] + grid_array[-1] + (delta_grid / 2.0) + particle_star_hsml)) &
                       (particle_star_coords[:, 2] > (parameters["centre_z"] + grid_array[0] - (delta_grid / 2.0) - particle_star_hsml)) & 
                       (particle_star_coords[:, 2] < (parameters["centre_z"] + grid_array[-1] + (delta_grid / 2.0) + particle_star_hsml)))
            
            particle_star_age_box = particle_star_age[ind_box] 
            particle_star_age_box[(particle_star_age_box < 0.001)] = 0.001 
            log_star_age = np.log10(particle_star_age_box) 
            for bin_idx in range(parameters["N_age_bins"]): 
                if bin_idx == 0: 
                    ind_age_bin = (log_star_age < log_stellar_age_Myr_bin_max[0])
                else: 
                    ind_age_bin = ((log_star_age >= log_stellar_age_Myr_bin_max[bin_idx - 1]) & (log_star_age < log_stellar_age_Myr_bin_max[bin_idx])) 
            
                particle_stellar_total = np.sum(particle_star_mass[ind_box][ind_age_bin]) 

                print("Stellar age bin %d: M_grid = %.4e Msol, M_particles = %.4e" % (bin_idx, grid_stellar_total[bin_idx], particle_stellar_total)) 
                sys.stdout.flush() 
            
        comm.Barrier() 

    # Walk through cells and write out the densities etc., in the correct order. 
    if rank == 0: 
        print("Writing outputs") 
        sys.stdout.flush() 

    for k in range(parameters["n_cell_base"]): 
        for j in range(parameters["n_cell_base"]): 
            for i in range(parameters["n_cell_base"]): 
                if grid_task[i, j, k] == rank: 
                    f_turb = open("microturbulence.binp", "ab") 
                    f_vol = open("cell_volume_kpc3.dat", "a") 

                    f_species_list = [] 
                    f_T_list = [] 
                    f_nH_list = [] 
                    f_vel_list = [] 
                    for species_index in range(N_species): 
                        species_filename = "numberdens_%s.binp" % (species_list[species_index], ) 
                        f_species_list.append(open(species_filename, "ab")) 

                        if emitter_flag_list[species_index] == 1: 
                            T_filename = "gas_temperature_%s.binp" % (species_list[species_index], ) 
                            f_T_list.append(open(T_filename, "ab")) 

                            nH_filename = "gas_nHtot_%s.binp" % (species_list[species_index], ) 
                            f_nH_list.append(open(nH_filename, "ab")) 
                            
                            vel_filename = "gas_velocity_%s.binp" % (species_list[species_index], ) 
                            f_vel_list.append(open(vel_filename, "ab")) 

                    amr_nodes_task[i - x_ind_low_task[rank], j - y_ind_low_task[rank], k - z_ind_low_task[rank]].walk_cells(f_species_list, 
                                                                                                                            f_T_list, 
                                                                                                                            f_nH_list, 
                                                                                                                            f_vel_list, 
                                                                                                                            f_turb, 
                                                                                                                            f_vol) 
 
                    f_turb.close() 
                    f_vol.close() 
                    for my_file in f_species_list: 
                        my_file.close() 
                    for my_file in f_T_list: 
                        my_file.close() 
                    for my_file in f_vel_list: 
                        my_file.close() 

                comm.Barrier() 

    # Dust densities need to walk through AMR 
    # grid twice (once for each species) 
    for dust_species in range(2): 
        for k in range(parameters["n_cell_base"]): 
            for j in range(parameters["n_cell_base"]): 
                for i in range(parameters["n_cell_base"]): 
                    if grid_task[i, j, k] == rank: 
                        f_dust = open("dust_density.binp", "ab") 
                        f_dust_T = open("dust_temperature_zero.dat", "a") 

                        amr_nodes_task[i - x_ind_low_task[rank], j - y_ind_low_task[rank], k - z_ind_low_task[rank]].walk_cells_dust(f_dust, f_dust_T, dust_species) 

                        f_dust.close() 
                        f_dust_T.close() 

                    comm.Barrier() 

    # Star densities need to walk through 
    # AMR grid multiple times (once for 
    # each age bin).
    if parameters["include_stars"] == 1: 
        for age_index in range(parameters["N_age_bins"]): 
            for k in range(parameters["n_cell_base"]): 
                for j in range(parameters["n_cell_base"]): 
                    for i in range(parameters["n_cell_base"]): 
                        if grid_task[i, j, k] == rank: 
                            f_star = open("stellarsrc_density.binp", "ab") 

                            amr_nodes_task[i - x_ind_low_task[rank], j - y_ind_low_task[rank], k - z_ind_low_task[rank]].walk_cells_stars(f_star, age_index) 

                            f_star.close() 

                        comm.Barrier() 

    for k in range(parameters["n_cell_base"]): 
        for j in range(parameters["n_cell_base"]): 
            for i in range(parameters["n_cell_base"]): 
                if grid_task[i, j, k] == rank: 
                    f_amr = open("amr_grid.inp", "a") 
                    
                    amr_nodes_task[i - x_ind_low_task[rank], j - y_ind_low_task[rank], k - z_ind_low_task[rank]].determine_amr_grid(f_amr) 

                    f_amr.close() 
                    
                comm.Barrier() 

    # rank 0 will now write out the stellarsrc_templates.inp file 
    if parameters["include_stars"] == 1: 
        if rank == 0: 
            # Read in SB99 template spectra 
            wavelength_list = [] 
            flux_list = [] 
            for i in range(parameters["N_age_bins"]): 
                template_file = "%s/spectrum_kroupa_Gv40_Zsol_age%d.dat" % (parameters["SB99_dir"], i + 1) 
                f_SB99 = open(template_file, "r") 
            
                wavelength = [] 
                flux = [] 
            
                for line in f_SB99: 
                    values = line.split() 
                    wavelength.append(float(values[0])) 
                    flux.append(float(values[1])) 
                
                wavelength_list.append(wavelength) 
                flux_list.append(flux) 
            
                f_SB99.close() 

            # Now create stellarsrc_templates.inp, 
            # for the smooth stellar distribution 
            f_star_src = open("stellarsrc_templates.inp", "w") 
            f_star_src.write("2 \n") 
        
            line = "%d \n" % (parameters["N_age_bins"], ) 
            f_star_src.write(line) 
        
            line = "%d \n" % (len(wavelength_list[0]), ) 
            f_star_src.write(line) 

            for i in wavelength_list[0]: 
                line = "%.6f \n" % (i, ) 
                f_star_src.write(line) 
            
            for age_index in range(0, parameters["N_age_bins"]): 
                for j in flux_list[age_index]: 
                    line = "%.6e \n" % (j * 4.0 * np.pi * ((3.086e18 ** 2.0) / 1.99e33), )  # erg/s/Hz per g of stars 
                    f_star_src.write(line) 

            f_star_src.close() 

        comm.Barrier() 

    if parameters["include_H_level_pop"] == 1: 
        if rank == 0: 
            print("Writing H level populations") 
            sys.stdout.flush() 

            # Write first few header lines 
            f_pop_Halpha_OF06 = open("levelpop_HII_Halpha_OF06.dat", "w") 
            f_pop_Halpha_OF06.write("1 \n") 

            line = "%d \n" % (leaf_max, ) 
            f_pop_Halpha_OF06.write(line) 
            
            f_pop_Halpha_OF06.write("3 \n") 
            f_pop_Halpha_OF06.write("1 2 3 \n") 
            
            f_pop_Halpha_OF06.close() 

            f_pop_Halpha_R15_HII = open("levelpop_HII_Halpha_R15.dat", "w") 
            f_pop_Halpha_R15_HII.write("1 \n") 

            line = "%d \n" % (leaf_max, ) 
            f_pop_Halpha_R15_HII.write(line) 
            
            f_pop_Halpha_R15_HII.write("3 \n") 
            f_pop_Halpha_R15_HII.write("1 2 3 \n") 
            
            f_pop_Halpha_R15_HII.close() 

            f_pop_Halpha_R15_HI = open("levelpop_HI_Halpha_R15.dat", "w") 
            f_pop_Halpha_R15_HI.write("1 \n") 

            line = "%d \n" % (leaf_max, ) 
            f_pop_Halpha_R15_HI.write(line) 
            
            f_pop_Halpha_R15_HI.write("3 \n") 
            f_pop_Halpha_R15_HI.write("1 2 3 \n") 
            
            f_pop_Halpha_R15_HI.close() 

            f_pop_Hbeta_OF06 = open("levelpop_HII_Hbeta_OF06.dat", "w") 
            f_pop_Hbeta_OF06.write("1 \n") 

            line = "%d \n" % (leaf_max, ) 
            f_pop_Hbeta_OF06.write(line) 
            
            f_pop_Hbeta_OF06.write("3 \n") 
            f_pop_Hbeta_OF06.write("1 2 3 \n") 
            
            f_pop_Hbeta_OF06.close() 

            f_pop_Hbeta_R15_HII = open("levelpop_HII_Hbeta_R15.dat", "w") 
            f_pop_Hbeta_R15_HII.write("1 \n") 

            line = "%d \n" % (leaf_max, ) 
            f_pop_Hbeta_R15_HII.write(line) 
            
            f_pop_Hbeta_R15_HII.write("3 \n") 
            f_pop_Hbeta_R15_HII.write("1 2 3 \n") 
            
            f_pop_Hbeta_R15_HII.close() 

        comm.Barrier() 
                
        for k in range(parameters["n_cell_base"]): 
            for j in range(parameters["n_cell_base"]): 
                for i in range(parameters["n_cell_base"]): 
                    if grid_task[i, j, k] == rank: 
                        f_pop_Halpha_OF06 = open("levelpop_HII_Halpha_OF06.dat", "a") 
                        f_pop_Halpha_R15_HII = open("levelpop_HII_Halpha_R15.dat", "a")  
                        f_pop_Halpha_R15_HI = open("levelpop_HI_Halpha_R15.dat", "a") 
                        f_pop_Hbeta_OF06 = open("levelpop_HII_Hbeta_OF06.dat", "a") 
                        f_pop_Hbeta_R15_HII = open("levelpop_HII_Hbeta_R15.dat", "a") 
                        f_pop_Hbeta_R15_HI = open("levelpop_HI_Hbeta_R15.dat", "a") 

                        amr_nodes_task[i - x_ind_low_task[rank], j - y_ind_low_task[rank], k - z_ind_low_task[rank]].walk_cells_H_level_pop(f_pop_Halpha_OF06, f_pop_Halpha_R15_HII, f_pop_Halpha_R15_HI, 0) 
                        amr_nodes_task[i - x_ind_low_task[rank], j - y_ind_low_task[rank], k - z_ind_low_task[rank]].walk_cells_H_level_pop(f_pop_Hbeta_OF06, f_pop_Hbeta_R15_HII, f_pop_Hbeta_R15_HI, 1) 

                        f_pop_Halpha_OF06.close() 
                        f_pop_Halpha_R15_HII.close() 
                        f_pop_Halpha_R15_HI.close() 
                        f_pop_Hbeta_OF06.close() 
                        f_pop_Hbeta_R15_HII.close() 
                        f_pop_Hbeta_R15_HI.close() 

                    comm.Barrier() 

    if rank == 0: 
        print("Finished") 
        sys.stdout.flush() 

    return 

if __name__ == "__main__": 
    main() 
    
