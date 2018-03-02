
def createRecHitMap(Filename_in,nevents,
                    nbins,width,
                    maxlayers,
                    maxhitsperpixel):
    
    import numpy as np
    import c_createRecHitMap
    
    array = np.zeros((nevents,nbins,nbins,maxlayers,2+4*maxhitsperpixel) , dtype='float32')
    
    
    c_createRecHitMap.fillRecHitMap(array,Filename_in,maxhitsperpixel,nbins,width,maxlayers)
   
    
   
    return array

def createRecHitMapNoTime(Filename_in,nevents,
                    nbins,width,
                    maxlayers,
                    maxhitsperpixel):
    
    import numpy as np
    import c_createRecHitMap
    
    array = np.zeros((nevents,nbins,nbins,maxlayers,2+3*maxhitsperpixel) , dtype='float32')
    
    
    c_createRecHitMap.fillRecHitMapNoTime(array,Filename_in,maxhitsperpixel,nbins,width,maxlayers)
   
    
   
    return array


def setTreeName(name):
    import c_createRecHitMap
    
    c_createRecHitMap.setTreeName(name)