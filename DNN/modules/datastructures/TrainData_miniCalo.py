
from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as djfto

def fileTimeOut(a,b):
    return djfto(a,b)

class TrainData_miniCalo(TrainData):
    
    def __init__(self):
        import numpy 
        TrainData.__init__(self)
        
        self.treename="B4"
        
        from converters import simple3Dstructure, setTreeName, simpleRandom3Dstructure
        setTreeName(self.treename)
        
        self.pu_multi=1
        
        self.undefTruth=['']
    
        self.truthclasses=['isPionCharged']
        
        self.weightbranchX='true_energy'
        self.weightbranchY='true_x'
        
        #is already flat
        self.referenceclass='flatten'
        self.weight_binX = numpy.array([0,0.1,40000],dtype=float) 
        self.weight_binY = numpy.array([-40000,40000],dtype=float) 
        
        
        
        self.registerBranches(['rechit_energy',
                               'rechit_x',
                               'rechit_y',
                               'rechit_layer'])
        
        
        self.regtruth='true_energy'

        self.regressiontargetclasses=['E']
        
        self.registerBranches(self.truthclasses)
        self.registerBranches([self.regtruth])
        
        self.reduceTruth(None)
        
        self.nbinsx = 0
        self.nbinsy = 0
        self.nbinsl = 0
        
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        #the first part is standard, no changes needed
        
        from converters import simple3Dstructure, setTreeName, simpleRandom3Dstructure
        setTreeName(self.treename)
        import numpy
        import ROOT
        
        
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()
        
        from DeepJetCore.stopwatch import stopwatch
        checktime= stopwatch()
        
        #just for testing
        #self.remove=False
        #self.nsamples=10
        #end just for testing
        
        removemuons=False
        makeplots=False
        onlypions=False
        
        import math
        
        x = simple3Dstructure(filename,self.nsamples,
                                    xbins=int(self.nbinsx),
                                    xwidth=100. / float(self.nbinsx),
                                    ybins=int(self.nbinsy),
                                    ywidth=100. / float(self.nbinsy),
                                    maxlayer=int(self.nbinsl), minlayer=0)
        
        x = x / 1e6

        Tuple = self.readTreeFromRootToTuple(filename)  
        
        idtruthtuple =  self.reduceTruth(Tuple[self.truthclasses])
        
        energytruth  =  numpy.array(Tuple[self.regtruth])
        
        
        print(x[0],energytruth[0])
        print(x[1],energytruth[1])
        print(x[2],energytruth[2])
        print(x[3],energytruth[3])
        
        print(x.shape)
        
        
        weights=numpy.zeros(len(idtruthtuple))+1
        
        
        
        self.w=[weights,weights]
        self.x=[x]
        self.y=[idtruthtuple,energytruth]
        
        