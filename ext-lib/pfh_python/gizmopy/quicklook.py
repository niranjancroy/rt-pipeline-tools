import numpy as np
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plot
from gizmopy.load_from_snapshot import *
from gizmopy.load_fire_snap import *
#matplotlib.use('Agg') ## this calls matplotlib without a X-windows GUI


def quicklook(sdir,snum,rcut,cen=np.zeros(0),xz=False,yz=False,
    key='stardmgastemp',numpart=1e9,figsize=(8.,8.),rasterized=True,marker=',',
    dt_Myr=100.,weight='B',quiet=False,pdf=False):
    '''
    master routine to quickly examine various snapshot properties
    '''    
    numpart = np.round(numpart).astype('int')
    pylab.close('all'); cen=np.array(cen); plot.figure(1,figsize=figsize);
    if(cen.size<=0):
        xyz=load_fire_snap('Coordinates',4,sdir,snum)
        print(xyz.shape)
        cen=np.median(xyz,axis=0); cn=xyz-cen; r2=np.sum(cn*cn,axis=1); ok=np.where(r2<rcut*rcut)[0]; xyz=xyz.take(ok,axis=0); 
        if(xyz.size > 3): cen=np.median(xyz,axis=0); 
        print('no center provided, centering on ',cen)
        
    if('massprofile' in key):
        quicklook_mass_profile(sdir,snum,cen=cen,rcut=rcut,key=key)
        if(pdf): pylab.savefig('qlook_'+key+'.pdf',transparent=True)
        return
    if(key=='sfr'): 
        quicklook_sfr_time(sdir,snum,cen=cen,rcut=rcut,dt_Myr=dt_Myr)
        if(pdf): pylab.savefig('qlook_'+key+'.pdf',transparent=True)
        return
    if(key=='phase'): 
        quicklook_phase(sdir,snum,cen=cen,rcut=rcut)
        if(pdf): pylab.savefig('qlook_'+key+'.pdf',transparent=True)
        return
    if(key=='vc'):
        quicklook_vc_profile(sdir,snum,cen=cen,rcut=rcut)
        if(pdf): pylab.savefig('qlook_'+key+'.pdf',transparent=True)
        return
    if('angmom' in key):
        p,j = quicklook_j_distrib(sdir,snum,cen=cen,rcut=rcut,key=key,weight=weight)
        if(pdf): pylab.savefig('qlook_'+key+'.pdf',transparent=True)
        return

    xcoord=0; ycoord=1; zcoord=2; 
    if(xz):
        xcoord=0; ycoord=2; zcoord=1;
    if(yz):
        xcoord=1; ycoord=2; zcoord=0;

    m_dm=0; m_star=0; m_gas=0; m_cold=0; m_warm=0; m_hot=0;
    if('dm' in key):
        xyz,mask=center_and_clip(load_fire_snap('Coordinates',1,sdir,snum),cen,rcut,numpart)
        if not (quiet): 
            m_dm=np.sum(load_fire_snap('Masses',1,sdir,snum,particle_mask=mask))
            print('DM mass=',m_dm)
        pylab.plot(xyz[:,xcoord],xyz[:,ycoord],marker=marker,linestyle='',color='red',rasterized=rasterized)

        if(2==0):
            xyz,mask=center_and_clip(load_fire_snap('Coordinates',2,sdir,snum),cen,rcut,numpart)
            if not (quiet): 
                m_dm=np.sum(load_fire_snap('Masses',1,sdir,snum,particle_mask=mask))
                print('DM mass=',m_dm)
            pylab.plot(xyz[:,xcoord],xyz[:,ycoord],marker=marker,linestyle='',color='purple',rasterized=rasterized)

    if('star' in key):
        xyz,mask=center_and_clip(load_fire_snap('Coordinates',4,sdir,snum),cen,rcut,numpart)
        pylab.plot(xyz[:,xcoord],xyz[:,ycoord],marker=marker,linestyle='',color='black',rasterized=rasterized)
        if not (quiet): 
            m_star=np.sum(load_fire_snap('Masses',4,sdir,snum,particle_mask=mask))
            print('Stellar mass=',m_star)
            print('Stellar center in plot=',np.median(xyz,axis=0)+cen)
    if('bh' in key):
        xyz,mask=center_and_clip(load_fire_snap('Coordinates',5,sdir,snum),cen,rcut,numpart)
        if(xyz.size > 0):
            pylab.plot(xyz[:,xcoord],xyz[:,ycoord],marker='x',markersize=20,linestyle='',color='magenta',rasterized=rasterized)
            if not (quiet): 
                m_bh=load_fire_snap('Masses',5,sdir,snum,particle_mask=mask)
                print('Total BH particle mass=',np.sum(m_bh),' Number of SMBH in plot=',m_bh.size)
                print('BH center in plot=',np.median(xyz,axis=0)+cen)
                print('BH particles largest-to-smallest: ',np.log10(1.e10*m_bh[np.argsort(m_bh)][::-1]))
                m_bh_t=load_fire_snap('BH_Mass',5,sdir,snum,particle_mask=mask)
                print('BH_Mass largest-to-smallest : ',np.log10(1.e10*m_bh_t[np.argsort(m_bh_t)][::-1]))
    if('gas' in key):
        xyz,mask=center_and_clip(load_fire_snap('Coordinates',0,sdir,snum),cen,rcut,numpart)
        if('temp' in key):
            t=load_fire_snap('Temperature',0,sdir,snum).take(mask,axis=0)
            if not (quiet): 
                m = load_fire_snap('Masses',0,sdir,snum,particle_mask=mask)
                m_gas=np.sum(m)
                print('Gas mass=',m_gas)
            tcold = 0.5e4; thot = 1.e6;
            tcold = 8000.; thot = 3.e5;
            ok=np.where(t > thot)[0]
            if(ok.shape[0] > 0):
                pylab.plot(xyz[ok,xcoord],xyz[ok,ycoord],marker=marker,linestyle='',color='orange',rasterized=rasterized)
                if not (quiet): 
                    m_hot = np.sum(m[ok])
                    print(' Gas mass [hot]=',m_hot)
            ok=np.where((t > tcold)&(t < thot))[0]
            if(ok.shape[0] > 0):
                pylab.plot(xyz[ok,xcoord],xyz[ok,ycoord],marker=marker,linestyle='',color='lime',rasterized=rasterized)
                if not (quiet): 
                    m_warm = np.sum(m[ok])
                    print(' Gas mass [warm]=',m_warm)
            ok=np.where(t < tcold)[0]
            if(ok.shape[0] > 0):
                pylab.plot(xyz[ok,xcoord],xyz[ok,ycoord],marker=marker,linestyle='',color='purple',rasterized=rasterized)
                if not (quiet): 
                    m_cold = np.sum(m[ok])
                    print(' Gas mass [cold]=',m_cold)
        else:
            pylab.plot(xyz[:,xcoord],xyz[:,ycoord],marker=marker,linestyle='',color='green',rasterized=rasterized)
            if not (quiet): 
                m_gas=np.sum(load_fire_snap('Masses',0,sdir,snum,particle_mask=mask))
                print('Gas mass=',m_gas)
    if not (quiet): 
        if(m_dm>0):
            print('Baryon fraction=',(m_gas+m_star)/(m_gas+m_star+m_dm) / (0.162))

    if(pdf): pylab.savefig('qlook_'+key+'.pdf',transparent=True)
    return



def quicklook_sfr_time(sdir,snum,cen=np.zeros(0),rcut=100.,dt_Myr=100.):
    '''
    subroutine to quickly estimate and plot the SF history
    '''
    xyz,mask = center_and_clip(load_fire_snap('Coordinates',4,sdir,snum),cen,rcut,1e9)
    age = load_fire_snap('StellarAgeGyr',4,sdir,snum).take(mask,axis=0)
    m = load_fire_snap('Masses',4,sdir,snum).take(mask,axis=0)
    bins = np.linspace(0.,13.8,13.8*1000./dt_Myr)
    y,xb = np.histogram(age,bins=bins,weights=m)
    y *= 1.e10 / (dt_Myr*1.e6)
    pylab.plot(xb[0:-1]+0.5*np.diff(xb),y,linestyle='-')
    pylab.axis([0.,13.8,np.min(y[y>0])/1.5,np.max(y)*1.5])
    pylab.yscale('log')
    return



def quicklook_phase(sdir,snum,cen=np.zeros(0),rcut=100.):
    '''
    subroutine to plot temperature-density diagram for gas in snapshot
    '''
    xyz,mask = center_and_clip(load_fire_snap('Coordinates',0,sdir,snum),cen,rcut,1e9)
    nH = np.log10( 406.4 * load_fire_snap('Density',0,sdir,snum).take(mask,axis=0) )
    T = np.log10( load_fire_snap('Temperature',0,sdir,snum).take(mask,axis=0) )
    pylab.plot(nH,T,marker=',',linestyle='',color='black',rasterized=True);
    return
    


def quicklook_vc_profile(sdir,snum,cen=np.zeros(0),rcut=100.):
    '''
    subroutine to quickly estimate and plot the circular velocity curve
    '''
    r2_all=np.zeros(0); m_all=np.zeros(0);
    for ptype in [0,1,4]:
        xyz,mask = center_and_clip(load_fire_snap('Coordinates',ptype,sdir,snum),cen,rcut,1e9)
        m = load_fire_snap('Masses',ptype,sdir,snum).take(mask,axis=0)
        r2 = np.sum(xyz*xyz,axis=1)
        m_all = np.concatenate([m_all,m],axis=0)
        r2_all = np.concatenate([r2_all,r2],axis=0)
    s=np.argsort(r2_all);
    r2=r2_all[s]; mt=np.cumsum(m_all[s]); r=np.sqrt(r2); vc=np.sqrt(6.67e-8*mt*1.e10*2.e33/(r*3.086e21))/1.e5;
    pylab.plot(r,vc,linestyle='-')
    pylab.axis([0.,rcut,0.,np.max(vc)*1.05])
    return
    


def quicklook_mass_profile(sdir,snum,cen=np.zeros(0),rcut=100.,key='massprofile_dm'):
    '''
    subroutine to quickly estimate and plot the spherically-averaged mass profile
    '''
    ptypes=np.zeros(0).astype('int')
    if('dm' in key): ptypes=np.append(ptypes,1)
    if('gas' in key): ptypes=np.append(ptypes,0)
    if('star' in key): ptypes=np.append(ptypes,4)
    for ptype in ptypes:
        xyz,mask = center_and_clip(load_fire_snap('Coordinates',ptype,sdir,snum),cen,rcut,1e9)
        m = load_fire_snap('Masses',ptype,sdir,snum).take(mask,axis=0)
        r2 = np.sum(xyz*xyz,axis=1)
        rmax=np.max(r2); 
        if(rcut<rmax): rmax=rcut;
        log_r = 0.5*np.log(r2)
        log_rg = np.linspace(np.min(log_r),np.log(rmax),100)
        y,xb=np.histogram(log_r,bins=log_rg,weights=m)
        r0 = np.exp(xb[0:-1]+0.5*np.diff(xb))
        y *= 1.e10 / ((xb[1]-xb[0]) * 4.*np.pi*r0*r0*r0)
        pylab.plot(r0,y,linestyle='-')
    pylab.yscale('log'); pylab.xscale('log')    
    return
 
    

def quicklook_j_distrib(sdir,snum,cen=np.zeros(0),rcut=100.,key='gas_angmom',
    weight='B',plot=True):
    '''
    subroutine to quickly estimate and plot the distribution of angular momentum
    '''
    ptype=0;
    if('gas' in key): 
        ptype=0; 
        if(weight=='B'): weight='Masses';
    if('dm' in key): 
        ptype=1; weight='Masses';
    if('star' in key): 
        ptype=4;
        weight='StellarLuminosity_'+weight
    if(('mass' in weight) or ('Mass' in weight) or (weight=='m')): weight='Masses';
    xyz,mask = center_and_clip(load_fire_snap('Coordinates',ptype,sdir,snum),cen,rcut,1e9) # load positions
    m = load_fire_snap('Masses',ptype,sdir,snum).take(mask,axis=0) # load masses [or weights, more generically]
    vxyz = load_fire_snap('Velocities',ptype,sdir,snum).take(mask,axis=0) # load velocities
    vxyz = vxyz - np.median(vxyz,axis=0) # reset to median local velocity about r=0
    v2_mag = np.sum(vxyz*vxyz,axis=1);
    jvec=np.cross(vxyz,xyz); m_jvec=(m*(jvec.transpose())).transpose(); # compute j vector
    j_tot=np.sum(m_jvec,axis=0); j_hat=j_tot/np.sqrt(np.sum(j_tot*j_tot)); j_z=np.sum(j_hat*jvec,axis=1); # compute total and project on it
    r20 = np.sum(xyz*xyz,axis=1);  m0=m; # get positions for interpolation below
    if(weight != 'Masses'): 
        if(weight=='NeutralHydrogenAbundance' or weight=='Z' or weight=='Temperature'):
            m0=m*load_fire_snap(weight,ptype,sdir,snum).take(mask,axis=0) # actually compute weights to use below for distribution function        
        else:
            m0=load_fire_snap(weight,ptype,sdir,snum).take(mask,axis=0) # actually compute weights to use below for distribution function

    
    r2_all=r20; m_all=m; # now compute the j for a circular orbit
    for ptype_t in [0,1,4]:
        if(ptype_t!=ptype):
            xyz,mask = center_and_clip(load_fire_snap('Coordinates',ptype_t,sdir,snum),cen,rcut,1e9)
            m = load_fire_snap('Masses',ptype_t,sdir,snum).take(mask,axis=0)
            r2 = np.sum(xyz*xyz,axis=1)
            m_all = np.concatenate([m_all,m],axis=0)
            r2_all = np.concatenate([r2_all,r2],axis=0)
    r_all=np.sqrt(r2_all); s=np.argsort(r_all); r=r_all[s]; m=m_all[s]; mr=m/r; r2=r*r;
    unit_vc2 = (6.67e-8 * 1.989e43 / 3.086e21) / 1.e10 # converts to (km/s)^2
    Gmr_inner=unit_vc2*np.cumsum(m)/r; Gmr_outer=unit_vc2*np.cumsum(mr[::-1])[::-1];
    vcirc_r = np.sqrt(Gmr_inner)
    phi_r = -(Gmr_inner + Gmr_outer)
    ecirc_r = -(0.5*Gmr_inner + Gmr_outer)
    jcirc_r = r * vcirc_r
    jc_interp_element = np.interp(r20,r2,jcirc_r)
    phi_element = np.interp(r20,r2,phi_r)
    e_element = phi_element + 0.5*v2_mag
    s=np.argsort(ecirc_r)
    r2_of_e = np.interp(e_element,ecirc_r[s],r2[s])
    jc_of_e = np.interp(r2_of_e,r2,jcirc_r)
    jz_i_r = j_z / jc_interp_element
    jz_i_e = j_z / jc_of_e
    nbins=100
    if(np.median(jz_i_e) < 0.5): nbins=50
    bins = np.linspace(-1.05,1.15,nbins)
    
    
    y_r,xb = np.histogram(jz_i_r,bins=bins,weights=m0,density=True)
    y_e,xb = np.histogram(jz_i_e,bins=bins,weights=m0,density=True)
    x = xb[0:-1]+0.5*np.diff(xb)
    if(plot): 
        #pylab.plot(x,y_r,linestyle='-',color='blue')
        pylab.plot(x,y_e,linestyle='-',color='black')
        pylab.plot([0.,0.],[0.,np.max(y_e)*1.05],color='black',linestyle=':')
        pylab.plot([1.,1.],[0.,np.max(y_e)*1.05],color='black',linestyle=':')
        pylab.axis([-0.9,1.6,0.,np.max(y_e)*1.05])
    
    return y_e, x


def center_and_clip(xyz,center,r_cut,n_target):
    '''
    trim vector, re-center it, and clip keeping particles of interest
    '''    
    xyz -= center;
    d = np.amax(np.abs(xyz), axis=1)
    ok = np.where(d < r_cut)[0]
    xyz = xyz.take(ok, axis=0)
    if(xyz.shape[0] > n_target):
        ok_n = np.random.choice(xyz.shape[0], n_target)
        xyz = xyz.take(ok_n, axis=0)
        ok = ok.take(ok_n, axis=0)
    return xyz, ok


def estimate_zoom_center(sdir, snum, # snapshot directory and number
    ptype=1, # particle type to search for density peak in 
    min_searchsize=0.01, # minimum radius in code units to search iteratively
    search_deviation_tolerance=0.0005, # tolerance for change in center on subsequent iterations
    search_stepsize=1.25, # typical step-size for zoom-in iteration around density peak
    max_searchsize=1000., # initial maximum radius around median point to use for iteration
    cen_guess=[0.,0.,0.], # initial guess for centroid, to use for iterations (0,0,0) = no guess
    quiet=True, # suppress output from iterations
    **kwargs): # additional keyword arguments for check_if_filename_exists
    '''
    Quick routine to estimate the central mass concentration of a zoom-in snapshot
    '''
    
    fname,fname_base,fname_ext = check_if_filename_exists(sdir,snum,**kwargs)
    cen = np.array(cen_guess)
    if(fname=='NULL'): return cen, 0
    if(fname_ext!='.hdf5'): return cen, 0
    file = h5py.File(fname,'r') # Open hdf5 snapshot file
    header_toparse = file["Header"].attrs # Load header dictionary (to parse below)
    numfiles = header_toparse["NumFilesPerSnapshot"]
    npartTotal = header_toparse["NumPart_Total"]
    if(npartTotal[ptype]<1): return cen, 0    
    boxsize = header_toparse["BoxSize"]
    time = header_toparse["Time"]
    z = header_toparse["Redshift"]
    hubble = header_toparse["HubbleParam"]
    file.close()
    ascale=1.0; cosmological=False;
    if(np.abs(time*(1.+z)-1.) < 1.e-6): 
        cosmological=True; ascale=time;
    rcorr = hubble / ascale; # scale to code units
    max_searchsize *= rcorr; min_searchsize *= rcorr; search_deviation_tolerance *= rcorr;
    if(max_searchsize > boxsize): max_searchsize=boxsize
    d_thold = max_searchsize
    if(cen[0]==0.): d_thold=1.e10
    niter = 0
    xyz = np.zeros((0,3))
    while(1):
        xyz_m=np.zeros(3); n_t=np.zeros(1);
        if(niter == 0):
            for i_file in range(numfiles):
                if (numfiles>1): fname = fname_base+'.'+str(i_file)+fname_ext  
                if(os.stat(fname).st_size>0):
                    file = h5py.File(fname,'r') # Open hdf5 snapshot file
                    npart = file["Header"].attrs["NumPart_ThisFile"]
                    if(npart[ptype] > 1):
                        xyz_all = np.array(file['PartType'+str(ptype)+'/Coordinates/'])
                        d = np.amax(np.abs(xyz_all-cen),axis=1)
                        ok = np.where(d < d_thold)
                        n0 = ok[0].shape[0]
                        xyz_all = xyz_all.take(ok[0],axis=0)
                        if(xyz_all.size > 0): 
                            xyz = np.concatenate([xyz,xyz_all])
                            xyz_m += np.sum(xyz_all,axis=0)
                            n_t += n0
                    file.close()
            xyz_prev = xyz
            if xyz.size == 0:
                ## didn't load any particles -- go back and use a bigger threshold on initial load
                d_thold *= 3
                niter = 0
                continue
        else:
            d = np.amax(np.abs(xyz_prev-cen),axis=1)
            ok = np.where(d < d_thold)
            n0 = ok[0].shape[0]
            xyz = xyz_prev.take(ok[0],axis=0)
            if(n0 > 1):
                xyz_m += np.sum(xyz,axis=0)
                n_t += n0
        niter += 1;
        if(n_t[0] <= 0): 
            d_thold *= 1.5
        else:
            xyz_m /= n_t[0]
            d_cen = np.sqrt(np.sum((cen-xyz_m)**2))
            if(quiet==False): print('cen_o=',cen[0],cen[1],cen[2],' cen_n=',xyz_m[0],xyz_m[1],xyz_m[2],' in box=',d_thold,' cen_diff=',d_cen,' min_search/dev_tol=',min_searchsize,search_deviation_tolerance)
            if(niter > 100): break
            if(niter > 3):
                if(d_thold <= min_searchsize): break
                if(d_cen <= search_deviation_tolerance): break
            if(n_t[0] <= 10): break
            cen = xyz_m
            d_thold /= search_stepsize
            if(d_thold < 2.*d_cen): d_thold = 2.*d_cen
            if(max_searchsize < d_thold): d_thold=max_searchsize
            xyz_prev = xyz
    return xyz_m / rcorr, npartTotal[ptype]

