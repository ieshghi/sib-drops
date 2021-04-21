import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def msd(xh):
    npart = xh.shape[0]
    nstep = xh.shape[2]
    msd = np.empty((npart,nstep-1))
    for i in range(nstep-1):
        msd[:,i] = np.mean(np.sum(np.diff(xh[:,:,::(i+1)],axis=2)**2,axis=1),axis=1)

    return msd

def partpic(xh,frame,L=1):
    xh = np.mod(xh,L)
    npart = xh.shape[0]
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    for i in range(npart):
        ax.scatter(xh[i,0,frame],xh[i,1,frame],xh[i,2,frame],'r.')

    # Setthe axes properties
    ax.set_xlim3d([0.0, 1])
    ax.set_xlabel('X')
    ax.set_ylim3d([0, 1.0])
    ax.set_ylabel('Y')    
    ax.set_zlim3d([0, 1.0])
    ax.set_zlabel('Z')    
#    ani = animation.FuncAnimation(fig, update_lines, n, fargs=(data, lines),
                        #      interval=50, blit=False)
    plt.show()
