import numpy as np
def instanttemp(rho,u,h,N):
    ua = np.sum(u*np.conj(u),axis=0)
    return rho*h**3/(2*(N**3+4))*np.sum(ua)*(1/(1.38*10**(-23)))

def modetemp(ukmaghist,rho,L,N):
    tktemp = np.empty((N,N,N)) 
    for i in range(N):
        for j in range(N):
            for l in range(N):
                if ((i==0)or(i==N/2))and((j==0)or(j==N/2))and((l==0)or(l==N/2)):
                    tktemp[i,j,l] = np.mean(ukmaghist[:,i,j,l])*rho*L**3/3*(1/(1.38*10**(-23)))
                else:
                    tktemp[i,j,l] = np.mean(ukmaghist[:,i,j,l])*rho*L**3/2*(1/(1.38*10**(-23)))
    return tktemp

