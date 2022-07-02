from __future__ import division,print_function

import numpy as np

def checklen(x):
    return len(np.array(x,ndmin=1));

def quick_lookback_time(z,h=0.71,Omega_M=0.27):
    ## exact solution for a flat universe
    a=1./(1.+z); x=Omega_M/(1.-Omega_M) / (a*a*a);
    t=(2./(3.*np.sqrt(1.-Omega_M))) * np.log( np.sqrt(x) / (-1. + np.sqrt(1.+x)) );
    t *= 13.777 * (0.71/h); ## in Gyr
    return t;
    
def get_stellar_ages(ppp,ppp_head,cosmological=1):
    ### ppp = loaded 'brick' of new star particle quantities, 
    ###        i.e. ppp=gadget.readsnap(directory,number,4,cosmological=1)
    ### ppp_head = loaded 'header' for the snapshot, 
    ###        i.e. ppp_head=gadget_readsnap(directory,number,4,cosmological=1,header_only=1)
	if (ppp['k'] != 1): return -1;
	a_form=ppp['age'];
	a_now=ppp_head['time'];
	
	if (cosmological==1):
	    z_form=1./a_form-1.; t_form=quick_lookback_time(z_form); 
	    z_now=1./a_now-1.; t_now=quick_lookback_time(z_now);
	    ages = (t_now - t_form); # should be in gyr
	else:
	    ages = a_now - a_form; # code units are gyr already!
	    
	return ages;

def calculate_zoom_center(sdir,snum,cen=[0.,0.,0.],clip_size=2.e10,\
        rho_cut=1.0e-5, h0=1, four_char=0, cosmological=1, skip_bh=1):
    import gadget
    
    rgrid=np.array([1.0e10,1000.,700.,500.,300.,200.,100.,70.,50.,30.,20.,10.,5.,2.5,1.]);
    rgrid=rgrid[rgrid <= clip_size];
    ps=gadget.readsnap( sdir, snum, 4, \
        h0=h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh,
        wetzel=True, properties=['mass', 'position'], force_float32=True)
    if (ps['k']==1):
        n_new=ps['m'].shape[0];
        if (n_new > 1):
            pos=ps['p']; x0s=pos[:,0]; y0s=pos[:,1]; z0s=pos[:,2];
    else:
        n_new=0
    pg=gadget.readsnap( sdir, snum, 0, \
        h0=h0,four_char=four_char,cosmological=cosmological,skip_bh=skip_bh,
        wetzel=True, properties=['density', 'position'], force_float32=True)
    rho=np.array(pg['rho'])*407.5;
    if (rho.shape[0] > 0):
        pos=pg['p']; x0g=pos[:,0]; y0g=pos[:,1]; z0g=pos[:,2];
    cen=np.array(cen);

    for i_rcut in range(checklen(rgrid)):
        for j_looper in range(5):
            if (n_new > 1000):
                x=x0s; y=y0s; z=z0s;
            else:
                ok=(rho > rho_cut);
                x=x0g[ok]; y=y0g[ok]; z=z0g[ok];
            x=x-cen[0]; y=y-cen[1]; z=z-cen[2];
            r = np.sqrt(x*x + y*y + z*z);
            ok = (r < rgrid[i_rcut]);
            if (checklen(r[ok]) > 1000):
                x=x[ok]; y=y[ok]; z=z[ok];
                if (i_rcut <= checklen(rgrid)-5):
                    cen+=np.array([np.median(x),np.median(y),np.median(z)]);
                else:
                    cen+=np.array([np.mean(x),np.mean(y),np.mean(z)]);
            else:
                if (checklen(r[ok]) > 200):
                    x=x[ok]; y=y[ok]; z=z[ok];
                    cen+=np.array([np.mean(x),np.mean(y),np.mean(z)]);
    return cen;
