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

nentries=td.x[0].shape[0]

print('data read. making plots')

for i in range(nentries):
    
    
    
    for k in range(2): #ecal hcal
        recmap=td.x[k]
        nentries=recmap.shape[0]
        ncolors=recmap[0].shape[3]
        
        xcenter=recmap[0].shape[0]/2
        xmax=recmap[0].shape[0]
        ycenter=recmap[0].shape[1]/2
        ymax=recmap[0].shape[1]
        zcenter=recmap[0].shape[2]/2
        
        xyrange=34
        zrange=8
        if k:
            print('hcal')
            xyrange=17
            zrange=10
        else:
            print('ecal')
            
        print('true energy: ', td.y[1][i])
        print('true ID [isGamma, isElectron, isPionCharged, isNeutralPion, isMuon]', td.y[0][i])

        for j in [1]: #range(ncolors): only the energy entry
            
            
            ax,plot,x,y,z,c=plot4d(recmap[i][:,:,:,j:j+1],'','phi-bin','layer','eta-bin')
            
            ax.set_ylim3d(0,zrange)
            ax.set_zlim3d(0,xyrange)
            ax.set_xlim3d(0,xyrange)
            
            l = a3d.Line3D((xcenter,xcenter),(0,0),(0,ymax),c = 'k', ls = '--')
            #ax.add_line(l)
            l2 = a3d.Line3D((0,xmax),(0,0),(ycenter,ycenter),c = 'k', ls = '--')
            #ax.add_line(l2)
            
            
            #ax.view_init(0,90)
            
            #ax.plot(x, y, 'r+', zdir='y')#, zs=1.5)
            #ax.plot(z, y, 'g+', zdir='x')#, zs=-0.5)
            #ax.plot(x, z, 'b+', zdir='z')#, zs=-1.5)
        
            plot.show()
            #input("Press [enter] to continue.")
            #plot.close()
    
