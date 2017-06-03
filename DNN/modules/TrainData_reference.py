
from TrainData import TrainData,fileTimeOut

class TrainData_reference(TrainData):
    
    def __init__(self):
        '''
        
        '''
        TrainData.__init__(self)
        
        #define truth:
        self.undefTruth=['']
        self.truthclasses=['isGamma','isHadron','isFake']
        self.referenceclass=''
        
        #this is only needed because the truth definitions are different from deepFlavour
        self.allbranchestoberead=[]
        self.registerBranches(self.truthclasses)
        
        #switch off all flavour rewightings from DeepJet
        self.remove=False    
        self.weight=False
        
        self.registerBranches(['rechit_energy',
                               'rechit_eta',
                               'rechit_phi',
                               'nrechits',
                               'seed_eta',
                               'seed_phi',
                               'true_energy'])
        ## standard part
        
        self.regtruth='true_energy'
        
        self.addBranches(['seed_eta'])

        
    
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['isReal','isFake']
        if tuple_in is not None:
            q = tuple_in['isGamma'].view(numpy.ndarray)
            w = tuple_in['isHadron'].view(numpy.ndarray)
            t = tuple_in['isFake'].view(numpy.ndarray)
            real=q+w
            
            return numpy.vstack((real,t)).transpose()  
        
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        #the first part is standard, no changes needed
        from preprocessing import MeanNormApply, createDensityMap, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        import ROOT
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
        
        
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        
        #flatten everything out for now
        x_chmap = createDensityMap(filename,TupleMeanStd,
                                   'rechit_energy', #use the energy to create the image
                                   self.nsamples,
                                   ['rechit_eta','seed_eta',3,0.2], 
                                   ['rechit_phi','seed_phi',3,0.2],
                                   'nrechits')    
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        idtruthtuple =  self.reduceTruth(Tuple[self.truthclasses])
        energytruth  =  numpy.array(Tuple[self.regtruth])
        
        weights=numpy.zeros(len(idtruthtuple))
        
        self.w=[weights,weights]
        self.x=[x_global,x_chmap]
        self.y=[idtruthtuple,energytruth]
        
        
        
        
        
        
        
