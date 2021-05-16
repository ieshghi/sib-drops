import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


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

def midmov(xh,u=0,ifsave=0):
    def update_lines(num,data,lines):
        conf = data[num,:,:] # current particle configuration
        for line,pos in zip(lines,conf):
            x,y = pos
            line.set_data(x,y)
    
    ndim = 2
    xh = xh[:,:2,:]
    data = np.swapaxes(xh,0,1)
    data = np.swapaxes(data,0,2)
    natom = data.shape[1]
    ndim = data.shape[2]
    nframe = data.shape[0]
    
    # Attach 3D axis to the figure
    fig = plt.figure()

    
    lines = [plt.plot(data[0,:,0], data[0,:,1],'bo')[0] for dat in data[0]]
    plt.xlim([0,1])
    plt.ylim([0,1])
    #plt.show()
    
    # Creating the Animation object
    ani = animation.FuncAnimation(fig, update_lines, nframe, fargs=(data, lines),interval=50, blit=False)
    if ifsave:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('movies/above.mp4', writer=writer)

    plt.show()
 
def midmov_heatmap(xh,u=0,ifsave=0):
    ndim = 2
    xh = xh[:,:2,:]
    data = np.swapaxes(xh,0,1)
    data = np.swapaxes(data,0,2)
    natom = data.shape[1]
    nframe = data.shape[0]
    
    # Attach 3D axis to the figure
    fig = plt.figure()

    d,x,y = np.histogram2d(data[0,:,0],data[0,:,1],50, range=[[0.3,1],[0,1]])
    im = plt.imshow(d.T,origin='lower',cmap='plasma')
    def animate(i):
        d,x,y = np.histogram2d(data[i,:,0],data[i,:,1],50, range=[[0.3,1],[0,1]])
        im.set_data(d)

    #plt.show()
    
    # Creating the Animation object
    ani = animation.FuncAnimation(fig, animate, nframe,interval=50, blit=False)
    if ifsave:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('movies/above.mp4', writer=writer)

    plt.show()
    

def makemov(xh,L=1,ifsave = 0):
    def update_lines(num,data,lines):
        conf = data[num,:,:] # current particle configuration
        for line,pos in zip(lines,conf):
            x,y,z = pos
            line.set_data(x,y)
            line.set_3d_properties(z)
    
    ndim = 3
    data = np.swapaxes(xh,0,1)
    data = np.swapaxes(data,0,2)
    natom = data.shape[1]
    ndim = data.shape[2]
    nframe = data.shape[0]
    
    # Attach 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.axes.set_xlim3d(left=0, right=1)
    ax.axes.set_ylim3d(bottom=0, top=1)
    ax.axes.set_zlim3d(bottom=0, top=1)
    ax.view_init(elev=30., azim=0)
    
    lines = [ax.plot(data[0,:,1], data[0,:,0], data[0,:,2],'o')[0] for dat in data[0]]
    #plt.show()
    
    # Creating the Animation object
    ani = animation.FuncAnimation(fig, update_lines, nframe, fargs=(data, lines),interval=50, blit=False)
    if ifsave:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('movies/3d.mp4', writer=writer)
    plt.show()
    
