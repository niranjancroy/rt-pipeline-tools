import h5py 
import numpy as np 

def read_chimes(filename, chimes_species, chimes_dataset = "PartType0/ChimesAbundances"): 
    """ Reads the abundances of a given species from the CHIMES 
        abundance array in an HDF5 snapshot. This uses the full 
        CHIMES network. 

        Parameters: 
        filename - string containing the name of the HDF5 file to 
                   read in. 
        chimes_species - string giving the name of the ion/molecule 
                         to extract.
        chimes_dataset - OPTIONAL, gives the name of the dataset 
                         containing the CHIMES array. Defaults to 
                         PartType0/ChimesAbundances.
        """

    h5file = h5py.File(filename, "r") 

    try: 
        chimes_index = chimes_dict[chimes_species] 
    except KeyError: 
        print("Error: species %s is not recognised in the CHIMES abundance array. Aborting." % (chimes_species, )) 
        return 

    output_array = h5file[chimes_dataset][:, chimes_index] 
    h5file.close() 

    return output_array 

def read_reduced_chimes(filename, chimes_species, chimes_dataset = "PartType0/ChimesAbundances", element_flags = np.zeros(9)): 
    """ Reads the abundances of a given species from the CHIMES 
        abundance array in an HDF5 snapshot. This uses the 
        reduced CHIMES network, i.e. when individual elements 
        have been switched off. 

        Parameters: 
        filename - string containing the name of the HDF5 file to 
                   read in. 
        chimes_species - string giving the name of the ion/molecule 
                         to extract.
        chimes_dataset - OPTIONAL, gives the name of the dataset 
                         containing the CHIMES array. Defaults to 
                         PartType0/ChimesAbundances.
        element_flags - OPTIONAL, array of 9 integers, each either 
                        0 or 1, indicating whether each metal in 
                        the CHIMES network is included in the 
                        reduced network. Defaults to the primordial 
                        network, i.e. only H and He included. 
        """

    h5file = h5py.File(filename, "r") 

    # Create reduced chimes dictionary 
    reduced_chimes_dict = create_reduced_chimes_dictionary(element_flags = element_flags) 

    try: 
        chimes_index = reduced_chimes_dict[chimes_species] 
        if chimes_index == -1: 
            raise Exception("ERROR: in read_reduced_chimes(), species %s is not included in the reduced network." % (chimes_species, )) 
    except KeyError: 
        raise Exception("ERROR: species %s is not recognised in the CHIMES abundance array. Aborting." % (chimes_species, )) 

    output_array = h5file[chimes_dataset][:, chimes_index] 
    h5file.close() 

    return output_array 

def create_reduced_chimes_dictionary(element_flags = np.zeros(9)): 
    """ Creates a dictionary mapping species names to their 
        position in the CHIMES abundance array when using a 
        reduced CHIMES network, i.e. when some of the elements 
        have been switched off. 

        Parameters: 
        element_flags - OPTIONAL, array of 9 integers, each either 
                        0 or 1, indicating whether each metal in 
                        the CHIMES network is included in the 
                        reduced network. Defaults to the primordial 
                        network, i.e. only H and He included. 
        """ 
    # Check that element_flags is the correct size 
    if len(element_flags) != 9: 
        raise Exception("ERROR: In create_reduced_chimes_dictionary(), len(element_flags) = %d, it needs to be of length 9." % (len(element_flags), )) 

    # Check that all values in element_flags 
    # are either 0 or 1 
    for idx in range(len(element_flags)): 
        if element_flags[idx] != 0 and element_flags[idx] != 1: 
            raise Exception("ERROR: In create_reduced_chimes_dictionary(), element_flags[%d] == %d; only values of 0 or 1 are allowed." % (idx, element_flags[idx])) 

    # Create a copy of chimes_dict 
    reduced_chimes_dict = chimes_dict.copy() 

    # Keep track of how many species have 
    # been excluded in the reduced network. 
    N_excluded = 0 
    
    # Carbon 
    carbon_ions = ["CI", "CII", "CIII", "CIV", "CV", "CVI", "CVII", "Cm"]
    if element_flags[0] == 0: 
        for species in carbon_ions: 
            reduced_chimes_dict[species] = -1 
            N_excluded += 1 

    # Nitrogen 
    nitrogen_ions = ["NI", "NII", "NIII", "NIV", "NV", "NVI", "NVII", "NVIII"]
    if element_flags[1] == 0: 
        for species in nitrogen_ions: 
            reduced_chimes_dict[species] = -1 
            N_excluded += 1 
    else: 
        for species in nitrogen_ions: 
            reduced_chimes_dict[species] -= N_excluded 

    # Oxygen 
    oxygen_ions = ["OI", "OII", "OIII", "OIV", "OV", "OVI", "OVII", "OVIII", "OIX", "Om"]
    if element_flags[2] == 0: 
        for species in oxygen_ions: 
            reduced_chimes_dict[species] = -1 
            N_excluded += 1 
    else: 
        for species in oxygen_ions: 
            reduced_chimes_dict[species] -= N_excluded 

    # Neon 
    neon_ions = ["NeI", "NeII", "NeIII", "NeIV", "NeV", "NeVI", "NeVII", "NeVIII", "NeIX", "NeX", "NeXI"]
    if element_flags[3] == 0: 
        for species in neon_ions: 
            reduced_chimes_dict[species] = -1 
            N_excluded += 1 
    else: 
        for species in neon_ions: 
            reduced_chimes_dict[species] -= N_excluded 

    # Magnesium 
    magnesium_ions = ["MgI", "MgII", "MgIII", "MgIV", "MgV", "MgVI", "MgVII", "MgVIII", "MgIX", "MgX", "MgXI", "MgXII", "MgXIII"]
    if element_flags[4] == 0: 
        for species in magnesium_ions: 
            reduced_chimes_dict[species] = -1 
            N_excluded += 1 
    else: 
        for species in magnesium_ions: 
            reduced_chimes_dict[species] -= N_excluded 

    # Silicon 
    silicon_ions = ["SiI", "SiII", "SiIII", "SiIV", "SiV", "SiVI", "SiVII", "SiVIII", "SiIX", "SiX", "SiXI", "SiXII", "SiXIII", "SiXIV", "SiXV"]
    if element_flags[5] == 0: 
        for species in silicon_ions: 
            reduced_chimes_dict[species] = -1 
            N_excluded += 1 
    else: 
        for species in silicon_ions: 
            reduced_chimes_dict[species] -= N_excluded 

    # Sulphur 
    sulphur_ions = ["SI", "SII", "SIII", "SIV", "SV", "SVI", "SVII", "SVIII", "SIX", "SX", "SXI", "SXII", "SXIII", "SXIV", "SXV", "SXVI", "SXVII"]
    if element_flags[6] == 0: 
        for species in sulphur_ions: 
            reduced_chimes_dict[species] = -1 
            N_excluded += 1 
    else: 
        for species in sulphur_ions: 
            reduced_chimes_dict[species] -= N_excluded 

    # Calcium 
    calcium_ions = ["CaI", "CaII", "CaIII", "CaIV", "CaV", "CaVI", "CaVII", "CaVIII", "CaIX", "CaX", "CaXI", "CaXII", "CaXIII", "CaXIV", "CaXV", "CaXVI", "CaXVII", "CaXVIII", "CaXIX", "CaXX", "CaXXI"]
    if element_flags[7] == 0: 
        for species in calcium_ions: 
            reduced_chimes_dict[species] = -1 
            N_excluded += 1 
    else: 
        for species in calcium_ions: 
            reduced_chimes_dict[species] -= N_excluded 

    # Iron 
    iron_ions = ["FeI", "FeII", "FeIII", "FeIV", "FeV", "FeVI", "FeVII", "FeVIII", "FeIX", "FeX", "FeXI", "FeXII", "FeXIII", "FeXIV", "FeXV", "FeXVI", "FeXVII", "FeXVIII", "FeXIX", "FeXX", "FeXXI", "FeXXII", "FeXXIII", "FeXXIV", "FeXXV", "FeXXVI", "FeXXVII"]
    if element_flags[8] == 0: 
        for species in iron_ions: 
            reduced_chimes_dict[species] = -1 
            N_excluded += 1 
    else: 
        for species in iron_ions: 
            reduced_chimes_dict[species] -= N_excluded 

    # Molecules 
    if N_excluded > 0: 
        reduced_chimes_dict["H2"] -= N_excluded 
        reduced_chimes_dict["H2p"] -= N_excluded 
        reduced_chimes_dict["H3p"] -= N_excluded 

        if element_flags[2] == 0: 
            reduced_chimes_dict["OH"] = -1
            reduced_chimes_dict["H2O"] = -1 
            N_excluded += 2 
        else: 
            reduced_chimes_dict["OH"] -= N_excluded 
            reduced_chimes_dict["H2O"] -= N_excluded 

        if element_flags[0] == 0: 
            reduced_chimes_dict["C2"] = -1 
            N_excluded += 1 
        else: 
            reduced_chimes_dict["C2"] -= N_excluded 

        if element_flags[2] == 0: 
            reduced_chimes_dict["O2"] = -1 
            N_excluded += 1
        else: 
            reduced_chimes_dict["O2"] -= N_excluded 

        if element_flags[0] == 0 or element_flags[2] == 0: 
            reduced_chimes_dict["HCOp"] = -1 
            N_excluded += 1 
        else: 
            reduced_chimes_dict["HCOp"] -= N_excluded 
            
        if element_flags[0] == 0: 
            reduced_chimes_dict["CH"] = -1 
            reduced_chimes_dict["CH2"] = -1 
            reduced_chimes_dict["CH3p"] = -1 
            N_excluded += 3 
        else: 
            reduced_chimes_dict["CH"] -= N_excluded 
            reduced_chimes_dict["CH2"] -= N_excluded 
            reduced_chimes_dict["CH3p"] -= N_excluded 

        if element_flags[0] == 0 or element_flags[2] == 0: 
            reduced_chimes_dict["CO"] = -1 
            N_excluded += 1 
        else: 
            reduced_chimes_dict["CO"] -= N_excluded 
            
        if element_flags[0] == 0: 
            reduced_chimes_dict["CHp"] = -1 
            reduced_chimes_dict["CH2p"] = -1 
            N_excluded += 2 
        else: 
            reduced_chimes_dict["CHp"] -= N_excluded 
            reduced_chimes_dict["CH2p"] -= N_excluded 

        if element_flags[2] == 0: 
            reduced_chimes_dict["OHp"] = -1 
            reduced_chimes_dict["H2Op"] = -1 
            reduced_chimes_dict["H3Op"] = -1 
            N_excluded += 3 
        else: 
            reduced_chimes_dict["OHp"] -= N_excluded 
            reduced_chimes_dict["H2Op"] -= N_excluded 
            reduced_chimes_dict["H3Op"] -= N_excluded 
            
        if element_flags[0] == 0 or element_flags[2] == 0: 
            reduced_chimes_dict["COp"] = -1 
            reduced_chimes_dict["HOCp"] = -1 
            N_excluded += 2 
        else: 
            reduced_chimes_dict["COp"] -= N_excluded 
            reduced_chimes_dict["HOCp"] -= N_excluded 

        if element_flags[2] == 0: 
            reduced_chimes_dict["O2p"] = -1 
            N_excluded += 1 
        else: 
            reduced_chimes_dict["O2p"] -= N_excluded 

    return reduced_chimes_dict 

def create_reduced_inverse_chimes_dictionary(element_flags = np.zeros(9)): 
    """ Returns the inverse dictionary, i.e. that maps the 
        species position in the reduced CHIMES abundance 
        array to the species name. 

        Parameters: 
        element_flags - OPTIONAL, array of 9 integers, each either 
                        0 or 1, indicating whether each metal in 
                        the CHIMES network is included in the 
                        reduced network. Defaults to the primordial 
                        network, i.e. only H and He included. 
        """ 

    # Create dictionary for reduced network 
    reduced_chimes_dict = create_reduced_chimes_dictionary(element_flags = element_flags) 
    
    # Take inverse of the dictionary 
    reduced_chimes_dict_inv = {ind: label for label, ind in reduced_chimes_dict.items()}

    # Remove the -1 entry -- not included 
    # in the reduced network. 
    if len(reduced_chimes_dict_inv) < 157: 
        del reduced_chimes_dict_inv[-1] 

    return reduced_chimes_dict_inv 

## CHIMES species dictionary. 
# 
#  This dictionary maps the species 
#  names to their position in the 
#  full CHIMES abundance array. 
chimes_dict = {"elec": 0,
               "HI": 1,
               "HII": 2,
               "Hm": 3,
               "HeI": 4,
               "HeII": 5,
               "HeIII": 6,
               "CI": 7,
               "CII": 8,
               "CIII": 9,
               "CIV": 10,
               "CV": 11,
               "CVI": 12,
               "CVII": 13,
               "Cm": 14,
               "NI": 15,
               "NII": 16,
               "NIII": 17,
               "NIV": 18,
               "NV": 19,
               "NVI": 20,
               "NVII": 21,
               "NVIII": 22,
               "OI": 23,
               "OII": 24,
               "OIII": 25,
               "OIV": 26,
               "OV": 27,
               "OVI": 28,
               "OVII": 29,
               "OVIII": 30,
               "OIX": 31,
               "Om": 32,
               "NeI": 33,
               "NeII": 34,
               "NeIII": 35,
               "NeIV": 36,
               "NeV": 37,
               "NeVI": 38,
               "NeVII": 39,
               "NeVIII": 40,
               "NeIX": 41,
               "NeX": 42,
               "NeXI": 43,
               "MgI": 44,
               "MgII": 45,
               "MgIII": 46,
               "MgIV": 47,
               "MgV": 48,
               "MgVI": 49,
               "MgVII": 50,
               "MgVIII": 51,
               "MgIX": 52,
               "MgX": 53,
               "MgXI": 54,
               "MgXII": 55,
               "MgXIII": 56,
               "SiI": 57,
               "SiII": 58,
               "SiIII": 59,
               "SiIV": 60,
               "SiV": 61,
               "SiVI": 62,
               "SiVII": 63,
               "SiVIII": 64,
               "SiIX": 65,
               "SiX": 66,
               "SiXI": 67,
               "SiXII": 68,
               "SiXIII": 69,
               "SiXIV": 70,
               "SiXV": 71,
               "SI": 72,
               "SII": 73,
               "SIII": 74,
               "SIV": 75,
               "SV": 76,
               "SVI": 77,
               "SVII": 78,
               "SVIII": 79,
               "SIX": 80,
               "SX": 81,
               "SXI": 82,
               "SXII": 83,
               "SXIII": 84,
               "SXIV": 85,
               "SXV": 86,
               "SXVI": 87,
               "SXVII": 88,
               "CaI": 89,
               "CaII": 90,
               "CaIII": 91,
               "CaIV": 92,
               "CaV": 93,
               "CaVI": 94,
               "CaVII": 95,
               "CaVIII": 96,
               "CaIX": 97,
               "CaX": 98,
               "CaXI": 99,
               "CaXII": 100,
               "CaXIII": 101,
               "CaXIV": 102,
               "CaXV": 103,
               "CaXVI": 104,
               "CaXVII": 105,
               "CaXVIII": 106,
               "CaXIX": 107,
               "CaXX": 108,
               "CaXXI": 109,
               "FeI": 110,
               "FeII": 111,
               "FeIII": 112,
               "FeIV": 113,
               "FeV": 114,
               "FeVI": 115,
               "FeVII": 116,
               "FeVIII": 117,
               "FeIX": 118,
               "FeX": 119,
               "FeXI": 120,
               "FeXII": 121,
               "FeXIII": 122,
               "FeXIV": 123,
               "FeXV": 124,
               "FeXVI": 125,
               "FeXVII": 126,
               "FeXVIII": 127,
               "FeXIX": 128,
               "FeXX": 129,
               "FeXXI": 130,
               "FeXXII": 131,
               "FeXXIII": 132,
               "FeXXIV": 133,
               "FeXXV": 134,
               "FeXXVI": 135,
               "FeXXVII": 136,
               "H2": 137,
               "H2p": 138,
               "H3p": 139,
               "OH": 140,
               "H2O": 141,
               "C2": 142,
               "O2": 143,
               "HCOp": 144,
               "CH": 145,
               "CH2": 146,
               "CH3p": 147,
               "CO": 148,
               "CHp": 149,
               "CH2p": 150,
               "OHp": 151,
               "H2Op": 152,
               "H3Op": 153,
               "COp": 154,
               "HOCp": 155,
               "O2p": 156}

## Inverse CHIMES species dictionary. 
# 
#  This dictionary gives the inverse 
#  of the chimes_dict mapping, i.e. 
#  given a position in the full CHIMES 
#  abundance array it will return the 
#  name of the species at that position. 
chimes_dict_inv = {ind: label for label, ind in chimes_dict.items()}
