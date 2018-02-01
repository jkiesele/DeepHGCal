
def createRecHitMap(Filename_in,nevents,
                    nbins,width,
                    maxlayers,
                    maxhitsperpixel):
    
    import numpy as np
    import c_createRecHitMap
    
    array = np.zeros((nevents,nbins,nbins,maxlayers,2+4*maxhitsperpixel) , dtype='float32')
    
    
    c_createRecHitMap.fillRecHitMap(array,Filename_in,maxhitsperpixel,nbins,width,maxlayers)
   
    
   
    return array