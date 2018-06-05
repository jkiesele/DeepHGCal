
def createRecHitMap(Filename_in,nevents,
                    nbins,width,
                    maxlayer, minlayer,
                    maxhitsperpixel):
    
    import numpy as np
    import c_createRecHitMap
    
    array = np.zeros((nevents,nbins,nbins,maxlayers,2+4*maxhitsperpixel) , dtype='float32')
    
    
    c_createRecHitMap.fillRecHitMap(array,Filename_in,maxhitsperpixel,nbins,width,maxlayer,minlayer)
   
    
   
    return array

def createRecHitMapNoTime(Filename_in,nevents,
                    xbins,xwidth,
                    ybins,ywidth,
                    maxlayer,minlayer,
                    maxhitsperpixel):
    
    import numpy as np
    import c_createRecHitMap
    
    array = np.zeros((nevents,nbins,nbins,maxlayers,2+3*maxhitsperpixel) , dtype='float32')
    
    
    c_createRecHitMap.fillRecHitMapNoTime(array,Filename_in,maxhitsperpixel,
                                          xbins,xwidth,
                                          ybins,ywidth,
                                          maxlayer,minlayer)
   
    
   
    return array


def simple3Dstructure(Filename_in,nevents,
                    xbins,xwidth,
                    ybins,ywidth,
                    maxlayer,minlayer,sumenergy=False):
    
    import numpy as np
    import c_createRecHitMap
    
    array = np.zeros((nevents,xbins,ybins,maxlayer-minlayer,2) , dtype='float32')
    
    
    c_createRecHitMap.simple3Dstructure(array,Filename_in,
                                          xbins,xwidth,
                                          ybins,ywidth,
                                          maxlayer,minlayer,sumenergy)
   
    
   
    return array

def setTreeName(name):
    import c_createRecHitMap
    
    c_createRecHitMap.setTreeName(name)