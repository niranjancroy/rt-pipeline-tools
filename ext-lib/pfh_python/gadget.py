from __future__ import division,print_function

######################################################################

def snap_ext(snum,four_char=0):
	ext='00'+str(snum);
	if (snum>=10): ext='0'+str(snum)
	if (snum>=100): ext=str(snum)
	if (four_char==1): ext='0'+ext
	if (snum>=1000): ext=str(snum)
	return ext;

######################################################################

import gadget_lib.readsnap
def readsnap(sdir,snum,ptype,h0=0,cosmological=0,skip_bh=0,four_char=0,header_only=0,loud=0):
    return gadget_lib.readsnap.readsnap(sdir,snum,ptype,h0=h0,\
            cosmological=cosmological,skip_bh=skip_bh,four_char=four_char,header_only=header_only,loud=loud);

######################################################################

## constants
import gadget_lib.constants 
mass_in_msun = gadget_lib.constants.mass_in_msun
G = gadget_lib.constants.G
KB = gadget_lib.constants.KB
c = gadget_lib.constants.c
msol = gadget_lib.constants.msol
rsol = gadget_lib.constants.rsol
lsol = gadget_lib.constants.lsol
G = gadget_lib.constants.G
ev2erg = gadget_lib.constants.ev2erg
pc = gadget_lib.constants.pc
kpc = gadget_lib.constants.kpc
unit_m = gadget_lib.constants.units_mass
unit_l = gadget_lib.constants.units_length
unit_v = gadget_lib.constants.units_velocity

######################################################################

import gadget_lib.gas_temperature

def gas_mu(num_e):
    return gadget_lib.gas_temperature.gas_mu(num_e);

def gas_temperature(u, num_e, keV=0):
    return gadget_lib.gas_temperature.gas_temperature(u, num_e, keV=keV);

def gas_cs_effective_eos(u, q_eos=1.0):
    return gadget_lib.gas_temperature.gas_cs_effective_eos(u, q_eos=q_eos);

def gas_xray_brems(mass_in_gadget_units, u_in_gadget_units, rho_in_gadget_units, num_e, num_h):
    return gadget_lib.gas_temperature.gas_xray_brems(mass_in_gadget_units, \
    u_in_gadget_units, rho_in_gadget_units, num_e, num_h);

######################################################################

import gadget_lib.load_stellar_hsml

def get_particle_hsml( x, y, z, DesNgb=32, Hmax=0.):
    return gadget_lib.load_stellar_hsml.get_particle_hsml( x, y, z, DesNgb=DesNgb, Hmax=Hmax);
    
def load_allstars_hsml(snapdir,snapnum,cosmo=0,use_rundir=0,four_char=0,use_h0=1,filename_prefix=''):
    return gadget_lib.load_stellar_hsml.load_allstars_hsml(snapdir,snapnum,cosmo=cosmo,\
        use_rundir=use_rundir,four_char=four_char,use_h0=use_h0,filename_prefix=filename_prefix);
        
######################################################################

import gadget_lib.cosmo

def get_stellar_ages(ppp,ppp_head,cosmological=1):
    return gadget_lib.cosmo.get_stellar_ages(ppp,ppp_head,cosmological=cosmological);
    
def calculate_zoom_center(sdir,snum,cen=[0.,0.,0.],clip_size=2.e10,\
        rho_cut=1.0e-5, h0=1, four_char=0, cosmological=1, skip_bh=1):
    return gadget_lib.cosmo.calculate_zoom_center(sdir,snum,cen=cen,clip_size=clip_size,\
        rho_cut=rho_cut,h0=h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh);
