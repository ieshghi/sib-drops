import immersed_fluid as code1
import matplotlib.pyplot as plt
import numpy as np
N = 20
T = 3000*1.38*10**(-23) 
L = 1e-5
dt = 1e-8
rho = 1000
mu = 0.01
Np = 200
psize = 1e-7
interac = 2e-14
maxt = 5*1e-3
    
xh,th,tk = code1.main(N,T,L,dt,rho,mu,Np,psize,interac,maxt)

np.save('outfiles/xh.npy',xh)
