
from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as djfto

def fileTimeOut(a,b):
    return djfto(a,b)

class TrainData_FCC(TrainData):
    
    def __init__(self):
        import numpy 
        TrainData.__init__(self)
        
        self.treename="tree"
        
        self.undefTruth=['']
    
        self.truthclasses=['isGamma',
                           'isElectron',
                           'isPionCharged',
                           'isNeutralPion',
                           ]
        
        self.weightbranchX='true_energy'
        self.weightbranchY='seed_eta'
        
        #is already flat
        self.referenceclass='flatten'
        self.weight_binX = numpy.array([0,0.1,40000],dtype=float) 
        self.weight_binY = numpy.array([-40000,40000],dtype=float) 
        
        
        
        self.registerBranches(['rechit_energy',
                               'rechit_eta',
                               'rechit_phi',
                               'rechit_layer',
                               'nrechits',
                               'seed_eta',
                               'seed_phi',
                               'true_energy'])
        
        
        self.regtruth='true_energy'

        self.regressiontargetclasses=['E']
        
        self.registerBranches(self.truthclasses)
        
        self.reduceTruth(None)
        
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        #the first part is standard, no changes needed
        from converters import createRecHitMapNoTime, setTreeName
        setTreeName("tree")
        import numpy
        import ROOT
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("tree")
        self.nsamples=tree.GetEntries()
        
        x_chmapbase=createRecHitMapNoTime(filename,self.nsamples,
                                    nbins=29,
                                    width=0.15,
                                    maxlayers=20,
                                    maxhitsperpixel=6)
        
        
        
        Tuple = self.readTreeFromRootToTuple(filename)  
        
        idtruthtuple =  self.reduceTruth(Tuple[self.truthclasses])
        energytruth  =  numpy.array(Tuple[self.regtruth])
        #simple by-hand scaling to around 0 with a width of max about 1
        energytruth = energytruth/100.
        
        weights=numpy.zeros(len(idtruthtuple))
        
        notremoves=numpy.zeros(energytruth.shape[0])
        notremoves+=1
        
        if self.remove:
            from augmentation import mirrorInPhi,duplicateImage,evaluateTwice
            
            x_chmapbase= mirrorInPhi(x_chmapbase)
            
            notremoves=evaluateTwice(weighter.createNotRemoveIndices,Tuple)
            
            weights=duplicateImage(weighter.getJetWeights(Tuple))
            energytruth   =duplicateImage(energytruth)
            idtruthtuple  =duplicateImage(idtruthtuple)
            
            #notremoves -= energytruth<50
            
        
        before=len(x_chmapbase)
        
        if self.remove:
            weights=weights[notremoves>0]
            x_chmapbase=x_chmapbase[notremoves>0]
            idtruthtuple=idtruthtuple[notremoves>0]
            energytruth=energytruth[notremoves>0]
        
        print('reduced to '+str(len(x_chmapbase))+' of '+ str(before))
        self.nsamples=len(x_chmapbase)
        
        self.w=[weights,weights]
        self.x=[x_chmapbase]
        self.y=[idtruthtuple,energytruth]
        
        