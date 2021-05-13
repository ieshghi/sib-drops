import immersed_fluid as code1
import matplotlib.pyplot as plt
import numpy as np
N = 20
T = 300*1.38*10**(-23) 
#Tarr = np.array([100,200,300,400,500,600])*1.38*10**(-23)
L = 1e-5
dt = 1e-6
rho = 1000
mu = 0.05
Np = 20
psize = 1e-7
interac = 1e-10
maxt = 1e-2
xh = code1.main(N,T,L,dt,rho,mu,Np,psize,interac,maxt)
    
#dxes = []
#for i in range(len(Tarr)):
#    xh,tk,th = code1.main(N,Tarr[i],L,dt,rho,mu,Np,psize,interac,maxt)
#    xh = xh[:,:,100:]
#    dx = (xh-0.5).flatten()
#    dxes.append(dx)


np.save('testing/xh.npy',xh)
#np.save('testing/tk.npy',tk)
#np.save('testing/th.npy',th)
#np.save('testing/dx.npy',dxes)
#np.save('testing/temps.npy',Tarr)
