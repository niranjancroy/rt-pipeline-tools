from __future__ import division,print_function

import numpy as np
import ctypes
#import pfh_utils as util
import os.path
import h5py
#import struct
#import array

def checklen(x):
    return len(np.array(x,ndmin=1));
def fcor(x):
    return np.array(x,dtype='f',ndmin=1)
def vfloat(x):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_float));

def ok_scan(input,xmax=1.0e10,pos=0):
    if (pos==1):
        return (np.isnan(input)==False) & (abs(input)<=xmax) & (input > 0.);
    else:
        return (np.isnan(input)==False) & (abs(input)<=xmax);

def snap_ext(snum,four_char=0):
	ext='00'+str(snum);
	if (snum>=10): ext='0'+str(snum)
	if (snum>=100): ext=str(snum)
	if (four_char==1): ext='0'+ext
	if (snum>=1000): ext=str(snum)
	return ext;


def get_particle_hsml( x, y, z, DesNgb=32, Hmax=0.):
    x=fcor(x); y=fcor(y); z=fcor(z); N=checklen(x); 
    ok=(ok_scan(x) & ok_scan(y) & ok_scan(z)); x=x[ok]; y=y[ok]; z=z[ok];
    if(Hmax==0.):
        dx=np.max(x)-np.min(x); dy=np.max(y)-np.min(y); dz=np.max(z)-np.min(z); ddx=np.max([dx,dy,dz]); 
        Hmax=5.*ddx*(np.float(N)**(-1./3.)); ## mean inter-particle spacing

    ## load the routine we need
    exec_call=util.return_python_routines_cdir()+'/StellarHsml/starhsml.so'
    h_routine=ctypes.cdll[exec_call];

    h_out_cast=ctypes.c_float*N; H_OUT=h_out_cast();
    ## main call to the hsml-finding routine
    h_routine.stellarhsml( ctypes.c_int(N), \
        vfloat(x), vfloat(y), vfloat(z), ctypes.c_int(DesNgb), \
        ctypes.c_float(Hmax), ctypes.byref(H_OUT) )

    ## now put the output arrays into a useful format 
    h = np.ctypeslib.as_array(np.copy(H_OUT));
    return h;



def load_allstars_hsml(snapdir,snapnum,cosmo=0,use_rundir=0,
        four_char=0,use_h0=1,filename_prefix='', DesNgb=62):
    import gadget

    ## do we already have a file of these pre-calculated with the right DesNgb?
    #rootdir='/work/01799/phopkins/stellar_hsml/'
    rootdir=snapdir

    exts=snap_ext(snapnum,four_char=four_char);
    s0=snapdir.split("/"); ss=s0[len(s0)-1]; 
    if(len(ss)==0): ss=s0[len(s0)-2];
    if(filename_prefix==''):
        hsmlfile_r=rootdir+ss+'_allstars_hsml_'+exts
    else:
        hsmlfile_r=rootdir+filename_prefix+'_allstars_hsml_'+exts
        while hsmlfile_r.split('/')[-1].startswith('_'):
            temp = hsmlfile_r.split('/')
            temp[-1] = temp[-1][1:]
            hsmlfile_r = '/'.join(temp)
        temp = hsmlfile_r.split('/')
        if len(temp) > 1:
            if temp[-2] == temp[-1].split('_')[0]:
                temp[-1] = '_'.join(temp[-1].split('_')[1:])
                hsmlfile_r = '/'.join(temp)

    if (use_rundir==1): hsmlfile_r=snapdir+'/allstars_hsml_'+exts
    hsmlfile=hsmlfile_r+'.hdf5' ## check if hdf5 file exists
    grpname = 'DesNgb'+str(int(DesNgb))

    if os.path.exists(hsmlfile): ## it exists! 
        with h5py.File(hsmlfile, 'r') as lut:
            if grpname in lut:    ## it has the data!
                grp = lut['DesNgb'+str(int(DesNgb))]
                nstars = grp.attrs['nstars']
                h_in = grp['h'][:]
                print("Loaded stellar smoothing lengths for snap {} with DesNgb = {} from {}".format(snapnum, DesNgb, hsmlfile))
                return h_in

    # else: ## no pre-computed file, need to do it ourselves
    #if we didn't return above, then we either don't have the pre-computed file or we don't have the right neighbor count
    have=0; ptype=4;
    properties_to_read = ['position']
    ppp=gadget.readsnap(snapdir,snapnum,ptype,h0=use_h0,
                        cosmological=cosmo, wetzel=True, properties=properties_to_read);

    if(ppp['k']==1):
        if(ppp['p'].shape[0]>1):
            have=1; pos=ppp['p']; x=pos[:,0]; y=pos[:,1]; z=pos[:,2];
    if (cosmo==0):
        for ptype in [2,3]:
            ppp=gadget.readsnap(snapdir,snapnum,ptype,h0=use_h0,cosmological=0, wetzel=True, properties=properties_to_read);
            if(ppp['k']==1):
                if(ppp['p'].shape[0]>1):
                    pos=ppp['p']
                    if (have==1):
                        x=np.concatenate((x,pos[:,0]));
                        y=np.concatenate((y,pos[:,1]));
                        z=np.concatenate((z,pos[:,2]));
                    else:
                        x=pos[:,0]; y=pos[:,1]; z=pos[:,2];
                    have=1;
    ## ok now we have the compiled positions
    if (have==1):
        h = get_particle_hsml(x,y,z,DesNgb=DesNgb);
        ## great now we've got the stars, lets write this to a file for next time
        nstars = checklen(h);
        if (nstars>1):
            try:
                with h5py.File(hsmlfile, 'a') as lut:
                    #check again whether the data is there, cause another process may have updated it in the interim...
                    if grpname not in lut:
                        grp = lut.create_group(grpname)
                        grp.attrs.create('nstars', nstars)
                        grp.create_dataset('h', data=h)
                        print("Computed and saved smoothing lengths for {} stars in snap {} with DesNgb = {} to {}".format(nstars, snapnum, DesNgb, hsmlfile))
                    else:
                        print("Computed softening lengths for {} stars in snap {} with DesNgb = {}, but it looks like they were saved to {} by another process in the interim".format(nstars, snapnum, DesNgb, hsmlfile))
            except IOError:
                print("!!! -- unable to save smoothing lengths with DesNgb = {} in {}; recommend checking your permissions, but continuing for now anyway".format(DesNgb, hsmlfile))
            return h;
        else:
            return 0;
    else:
        return 0;
    return 0; ## failed to find stars

