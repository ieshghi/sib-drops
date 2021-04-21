import numpy as np

def vec_interp(u,X,h,N): #throughout this code we break the standards for array shapes that we've been following so far, because we need to just copy what's in the ib3d code for this. To get the right vectorization the original authors do some array magic which I do not understand, and so I must copy it blindly. Someday I will come back and optimize this properly and write it cleanly
    Nb = X.shape[1]
    U = np.empty((Nb,3))
    s = X.T/h
    i = np.floor(s)
    r = s-i #position of particle relative to gridpoint "below" it
    w = vec_phi1(r[:,0])*vec_phi2(r[:,1])*vec_phi3(r[:,2]) 
    w = w.transpose([1,2,3,0])
    for k in range(Nb):
        i1 = np.mod(np.arange((i[k,0]-1),i[k,0]+3),N).astype(int)
        i2 = np.mod(np.arange((i[k,1]-1),i[k,1]+3),N).astype(int)
        i3 = np.mod(np.arange((i[k,2]-1),i[k,2]+3),N).astype(int)

        ww = w[:,:,:,k]
        ukeep = u[:,i1,:,:][:,:,i2,:][:,:,:,i3]
        UU = np.array([np.sum(ww*ukeep[0,:,:,:]),np.sum(ww*ukeep[1,:,:,:]),np.sum(ww*ukeep[2,:,:,:])])
        U[k,:] = UU

    return U.T

def vec_spread(force,X,h,N):
    Nb = X.shape[1]
    fluidf = np.zeros((3,N,N,N))

    c = 1/(h*h*h)
    s = X.T/h
    i = np.floor(s)
    r = s-i

    w = vec_phi1(r[:,0])*vec_phi2(r[:,1])*vec_phi3(r[:,2]) 
    w = w.transpose([1,2,3,0])

    for k in range(Nb):
        i1 = np.mod(np.arange((i[k,0]-1),i[k,0]+3),N).astype(int)
        i2 = np.mod(np.arange((i[k,1]-1),i[k,1]+3),N).astype(int)
        i3 = np.mod(np.arange((i[k,2]-1),i[k,2]+3),N).astype(int)

        ww = w[:,:,:,k]

        for l in range(3):
            fluidf[l,:,:,:][i1,:,:][:,i2,:][:,:,i3] += c*force[l,k]*ww

    return fluidf
    

#the three functions below will be easily combine-able into one function... I will eventually do this
def vec_phi1(r):
    w = np.zeros((r.shape[0],4,4,4))
    q = np.sqrt(1+4*r*(1-r))
    nq = len(q)
    w[:,3,:,:] = np.tile(((1+2*r-q)/8).reshape(1,nq),[4,4,1]).T
    w[:,2,:,:] = np.tile(((1+2*r+q)/8).reshape(1,nq),[4,4,1]).T
    w[:,1,:,:] = np.tile(((3-2*r+q)/8).reshape(1,nq),[4,4,1]).T
    w[:,0,:,:] = np.tile(((3-2*r-q)/8).reshape(1,nq),[4,4,1]).T
    return w

def vec_phi2(r):
    w = np.zeros((r.shape[0],4,4,4))
    q = np.sqrt(1+4*r*(1-r))
    nq = len(q)
    w[:,:,3,:] = np.tile(((1+2*r-q)/8).reshape(1,nq),[4,4,1]).T
    w[:,:,2,:] = np.tile(((1+2*r+q)/8).reshape(1,nq),[4,4,1]).T
    w[:,:,1,:] = np.tile(((3-2*r+q)/8).reshape(1,nq),[4,4,1]).T
    w[:,:,0,:] = np.tile(((3-2*r-q)/8).reshape(1,nq),[4,4,1]).T
    return w

def vec_phi3(r):
    w = np.zeros((r.shape[0],4,4,4))
    q = np.sqrt(1+4*r*(1-r))
    nq = len(q)
    w[:,:,:,3] = np.tile(((1+2*r-q)/8).reshape(1,nq),[4,4,1]).T
    w[:,:,:,2] = np.tile(((1+2*r+q)/8).reshape(1,nq),[4,4,1]).T
    w[:,:,:,1] = np.tile(((3-2*r+q)/8).reshape(1,nq),[4,4,1]).T
    w[:,:,:,0] = np.tile(((3-2*r-q)/8).reshape(1,nq),[4,4,1]).T
    return w

