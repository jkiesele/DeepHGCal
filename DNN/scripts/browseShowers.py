#!/usr/bin/env python


from argparse import ArgumentParser
parser = ArgumentParser('Browse entries in a DeepHGCal TrainData file. Assumes standard ordering (3D image as second entry)')
parser.add_argument('inputFile')
args = parser.parse_args()

import matplotlib.pyplot as plt
from plotting import plot4d

from DeepJetCore.TrainData import TrainData
import mpl_toolkits.mplot3d.art3d as a3d

td=TrainData()
td.readIn(args.inputFile)
x_chmap=td.x[1]
del td
nentries=x_chmap.shape[0]
ncolors=x_chmap[0].shape[3]

xcenter=x_chmap[0].shape[0]/2
xmax=x_chmap[0].shape[0]
ycenter=x_chmap[0].shape[1]/2
ymax=x_chmap[0].shape[1]
zcenter=x_chmap[0].shape[2]/2

print(ncolors)

for i in range(nentries):
    print(x_chmap[i].shape)
    for j in [0]: #range(ncolors):
        ax,plot,x,y,z,c=plot4d(x_chmap[i][:,:,:,j:j+1],'','etabin','layer','phibin')
        
        l = a3d.Line3D((xcenter,xcenter),(0,0),(0,ymax),c = 'k', ls = '--')
        ax.add_line(l)
        l2 = a3d.Line3D((0,xmax),(0,0),(ycenter,ycenter),c = 'k', ls = '--')
        ax.add_line(l2)
        #ax.view_init(0,90)
        
        #ax.plot(x, y, 'r+', zdir='y')#, zs=1.5)
        #ax.plot(z, y, 'g+', zdir='x')#, zs=-0.5)
        #ax.plot(x, z, 'b+', zdir='z')#, zs=-1.5)

        plot.show()
        #input("Press [enter] to continue.")
        plot.close()
    
