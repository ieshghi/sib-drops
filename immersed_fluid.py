import numpy as np
import matplotlib.pyplot as plt
from interpfuncs import *
import temptests

def main():
    N = 50
    T = 310*(1.38*10**(-23))
    L = 100
    dt = 10 
    h = L/N
    rho = 1
    mu = 0.001
    Np = 0 
    psize = 0.1
    interac = 1e-7 
    fric = (6*np.pi*mu*psize) #stokes friction, why not
    Dexp = T/fric

    maxt = 10000
    nsteps = int(np.ceil(maxt*1.0/dt))
    xhist = np.empty((Np,3,nsteps))

    u = init_fluid(N,h)
    k,dk,tau_k,plong,ptrans,ak,xik,sigk = init_arrs(N,h,mu,rho,dt,T)
    X = init_particles(Np,L)

    temphist = np.empty(nsteps)
    ukmaghist = np.empty((nsteps,N,N,N))

    for i in range(nsteps):
        print('Time = ',i*dt)
        
        xhist[:,:,i] = X.T;
        bigF = interparticle_force(X,psize,interac,fric,dt)
        #XX = X + dt*bigF/fric
        XX = X + particle_mot(u,X,T,bigF,fric,psize,dt,N,h)        
        force = vec_spread(bigF,X,h,N) 
        uu,uk = fluid_step(u,force,ak,tau_k,sigk,ptrans,rho) 
        temphist[i] = temptests.instanttemp(rho,u,h,N) 
        ukmaghist[i,:,:,:] = np.sum(uk*np.conj(uk),axis=0)
        #if i%10==0:
        #    plt.plot(uu[0,int(np.floor(N/2)),int(np.floor(N/2)),:])
        u = uu.copy()
        X = XX.copy()
        
    tktemp = temptests.modetemp(ukmaghist,rho,L,N)  
    return xhist,temphist,tktemp

def particle_mot(u,X,T,bigF,fric,psize,dt,N,h):
    Np = X.shape[1]
    adv = dt*vec_interp(u,X,N,h)
    therm = np.sqrt(2*fric*T/dt)*np.random.randn(3,Np)

    return adv + fric**(-1)*bigF + therm

def interparticle_force(X,psize,interac,fric,dt):
    Np = X.shape[1]
    nji = np.empty((3,Np,Np))
    rji = np.empty((Np,Np))
    cutoff_range = 3000*psize
    maxforce = psize*fric/dt
    for i in range(Np):
        for j in range(Np):
            d = X[:,j]-X[:,i]
            rji[i,j] = np.sqrt(np.sum(d**2))
            if abs(rji[i,j])>0:
                nji[:,i,j] = d/rji[i,j]
            else:
                nji[:,i,j] = 0

    fji = nji.copy()
    #fji_scal = np.zeros(rji.shape)
    fji_scal = lennardjones(rji,psize,interac)
    
    fji *= np.nan_to_num(fji_scal)
    fji_tot = np.sum(fji,axis=2)
    return np.maximum(fji_tot,-maxforce) 

def lennardjones(r,sigma,epsilon):
    return 4*epsilon*(6*sigma**6/r**7-12*sigma**12/r**13)

def init_particles(Np,L):
    loc1 = np.array([0,0.3,0])
    loc2 = np.array([0,0.8,0])
    cloudsize = L/5
    Xi = np.empty((3,Np))
    i1 = int(np.floor(Np/2))
    for i in range(i1):
        Xi[:,i] = cloudsize*np.random.rand(3)+loc1
    for i in range(i1,Np):
        Xi[:,i] = cloudsize*np.random.rand(3)+loc2

    return Xi
    #return L*np.random.rand(3,Np) #uniformly distribute particles

def imposeforce(N):
    force = np.zeros((3,N,N,N))
    force[0,:,:,0] = 1
    force[0,:,:,int(np.floor(N/2))] = -1

    return force

def init_fluid(N,h):
    return np.zeros((3,N,N,N))

def fluid_step(u,force,ak,tau_k,sigk,ptrans,rho):
    N = np.size(u[0,:,0,0])
    uk = np.fft.fft(u,axis=1,norm='forward')
    uk = np.fft.fft(uk,axis=2,norm='forward')
    uk = np.fft.fft(uk,axis=3,norm='forward')
    fk = np.fft.fft(force,axis=1,norm='forward')
    fk = np.fft.fft(fk,axis=2,norm='forward')
    fk = np.fft.fft(fk,axis=3,norm='forward')
   
    akarr = np.repeat(ak,3,axis=0)
    tkarr = np.repeat(tau_k,3,axis=0)
    sigarr = np.repeat(sigk,3,axis=0)
    
    fterm = (np.einsum('mrnkl,rnkl->mnkl',ptrans,fk))*tkarr*(1-akarr)/rho

    noise = np.random.randn(3,N,N,N)+1j*np.random.randn(3,N,N,N)
    noiseterm = (np.einsum('mrnkl,rnkl->mnkl',ptrans,sigarr*noise))

    uk = akarr*uk + fterm + noiseterm

    uu = np.fft.ifft(uk,axis=1,norm='forward')
    uu = np.fft.ifft(uu,axis=2,norm='forward')
    uu = np.fft.ifft(uu,axis=3,norm='forward')

    return uu,uk

def init_arrs(N,h,mu,rho,dt,T):
    L = N*h

    k = np.empty((3,N,N,N))
    tau_k = np.empty((1,N,N,N))
    for i in range(N):
        for j in range(N):
            for l in range(N):
                k[0,i,j,l] = i
                k[1,i,j,l] = j
                k[2,i,j,l] = l
                tau_k[0,i,j,l] = (4*mu/(rho*h**2)*np.sum(np.sin(np.pi*k[:,i,j,l]*1.0/N)**2))**(-1)

    dk = (0+1j)*np.sin(2*np.pi*k*1.0/N)/h
    plong = np.empty((3,3,N,N,N))
    ptrans = np.empty((3,3,N,N,N))
    xik = np.empty((1,N,N,N))

    for i in range(N):
        for j in range(N):
            for l in range(N):
                dkijl = dk[:,i,j,l]
                if ((i==0)or(i==N/2))and((j==0)or(j==N/2))and((l==0)or(l==N/2)):
                    plong[:,:,i,j,l] = 0
                    xik[0,i,j,l] = T/(rho*L**3*tau_k[0,i,j,l])
                else:
                    plong[:,:,i,j,l] = np.outer(dkijl,dkijl)/(np.sum(abs(dkijl)**2))
                    xik[0,i,j,l] = T/(2*rho*L**3*tau_k[0,i,j,l])
                    
                ptrans[:,:,i,j,l] = np.identity(3)-plong[:,:,i,j,l]

    tau_k[:,0,0,0] = 0;
    ak = np.exp(-dt/tau_k)
    ak[:,0,0,0] = 0;
    sigk = np.sqrt(xik*tau_k*(1-ak**2))
    sigk[:,0,0,0] = 0;

    return k,dk,tau_k,plong,ptrans,ak,xik,sigk
