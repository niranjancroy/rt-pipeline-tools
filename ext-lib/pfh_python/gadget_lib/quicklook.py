import numpy as np



def quicklook( P, x0, cen=[0.,0.,0.], xy=1,yz=0,xz=0, point_num_cut=1.e5 ):
    pylab.close('all')
    x=P['p'][:,0]-cen[0]; y=P['p'][:,1]-cen[1]; z=P['p'][:,2]-cen[2];
    r2=x*x+y*y+z*z; ok=(r2 lt 5.*x0); 
    if (x[ok].size > point_num_cut):
        ok = ok & (np.random.rand(x.size) < point_num_cut/(1.*x[ok].size))
    x=x[ok]; y=y[ok]; z=z[ok];
    xx=x; yy=y; 
    if(yz==1):
        xx=y; yy=z;
    if(xz==1):
        xx=x; yy=z;
    pylab.axis([-x0,x0,-x0,x0])
    pylab.plot(xx,yy,marker=',',linestyle='',rasterized=True,color='black')
    