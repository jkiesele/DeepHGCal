
from TrainData_FCC import TrainData_FCC, fileTimeOut
from numpy import shape

class TrainData_FCC_rec(TrainData_FCC):
    
    def __init__(self):
        import numpy 
        TrainData_FCC.__init__(self)
        
        
        
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        import numpy
        import ROOT
        import c_createRecHitMap
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        
        
        
        
        Tuple = self.readTreeFromRootToTuple(filename)  
        self.nsamples=len(Tuple)
        
        hitsperseed = 20000 # around 4000 sufficient for EM - but zero masked anyway
        # 4 per rechit: deta, dphi, energy layer
        rechitlist=numpy.zeros((self.nsamples,hitsperseed,4),dtype='float32')
        c_createRecHitMap.setTreeName("tree")
        c_createRecHitMap.fillRecHitListNoTime(rechitlist,filename,hitsperseed,0.5,20) 
        
        
        
        idtruthtuple =  self.reduceTruth(Tuple[self.truthclasses])
        energytruth  =  numpy.array(Tuple[self.regtruth])
        #simple by-hand scaling to around 0 with a width of max about 1
        energytruth = energytruth/100.
        
        weights=numpy.zeros(len(idtruthtuple))
        
        notremoves=numpy.zeros(energytruth.shape[0])
        notremoves+=1
    
       
        before=len(rechitlist)
        
        if self.remove:
            weights=weights[notremoves>0]
            rechitlist=rechitlist[notremoves>0]
            idtruthtuple=idtruthtuple[notremoves>0]
            energytruth=energytruth[notremoves>0]
        
        print('reduced to '+str(len(rechitlist))+' of '+ str(before))
        self.nsamples=len(rechitlist)
        
        self.w=[weights,weights]
        self.x=[rechitlist]
        self.y=[idtruthtuple,energytruth]
        
        