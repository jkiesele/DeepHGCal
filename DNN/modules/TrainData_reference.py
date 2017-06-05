
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
                               'rechit_layer',
                               'nrechits',
                               'seed_eta',
                               'seed_phi',
                               'true_energy'])
        ## standard part
        
        self.regtruth='true_energy'
        
        self.addBranches(['seed_eta'])

        
    
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['isGamma','isHadron','isFake']
        if tuple_in is not None:
            q = tuple_in['isGamma'].view(numpy.ndarray)
            w = tuple_in['isHadron'].view(numpy.ndarray)
            t = tuple_in['isFake'].view(numpy.ndarray)
            #real=q+w
            
            return numpy.vstack((q,w,t)).transpose()  
        
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        #the first part is standard, no changes needed
        from preprocessing import MeanNormApply,createDensityLayers, createDensityMap, MeanNormZeroPad, MeanNormZeroPadParticles
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
        x_chmap = createDensityLayers(filename,
                                      TupleMeanStd,
                                      inbranch='rechit_energy', 
                                      layerbranch='rechit_layer',
                                      maxlayers=55,
                                      layeroffset=1,
                                      nevents=self.nsamples,
                                      dimension1=['rechit_eta','seed_eta',23,0.3], 
                                      dimension2=['rechit_phi','seed_phi',23,0.3],
                                      counterbranch='nrechits')
        
        from plotting import plot4d, rotanimate
        giffile=filename.replace('/','_')
        for i in range(0,3):
            ax,_=plot4d(x_chmap[i],giffile+"_"+str(i)+".pdf",'etabin','layer','phibin')
            rotanimate(ax,giffile+'_'+str(i)+'.gif',delay=5,prefix=giffile)
        #
        #
        #
        #exit()
        
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        idtruthtuple =  self.reduceTruth(Tuple[self.truthclasses])
        energytruth  =  numpy.array(Tuple[self.regtruth])
        
        weights=numpy.zeros(len(idtruthtuple))
        
        self.w=[weights,weights]
        self.x=[x_global,x_chmap]
        self.y=[idtruthtuple,energytruth]
        
        
        
        
        
        
        
