
from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as djfto

def fileTimeOut(a,b):
    return djfto(a,b)

class TrainData_FCC(TrainData):
    
    def __init__(self):
        import numpy 
        TrainData.__init__(self)
        
        self.treename="events"
        
        self.pu_multi=1
        
        self.undefTruth=['']
    
        self.truthclasses=['isGamma',
                           'isElectron',
                           'isPionCharged',
                           'isNeutralPion',
                           'isMuon',
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
        from converters import simple3Dstructure, setTreeName, simpleRandom3Dstructure
        setTreeName("events")
        import numpy
        import ROOT
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("events")
        self.nsamples=tree.GetEntries()
        
        from DeepJetCore.stopwatch import stopwatch
        checktime= stopwatch()
        
        #just for testing
        #self.remove=False
        #self.nsamples=10
        #end just for testing
        
        removemuons=False
        makeplots=False
        
        import math
        
        x_chmapecal=simple3Dstructure(filename,self.nsamples,
                                    xbins=34,
                                    xwidth=2*math.pi/704*34/2,
                                    ybins=34,
                                    ywidth=0.17,
                                    maxlayer=8, minlayer=0)
        
        
        #granularity needs to be checked  deltaEta=0.025 und deltaPhi=2Pi/256
        x_chmaphcal=simple3Dstructure(filename,self.nsamples,
                                    xbins=17,
                                    xwidth=2*math.pi/256*17/2,
                                    ybins=17,
                                    ywidth=0.2125,
                                    maxlayer=18, minlayer=8,sumenergy=True)
        
        
        ## add pileup from
        
        # pass to random seed eta = 0.36 or -0.36 and random phi binned in 2*math.pi/704
        # with random starting point in event number with loop from last back to first event
        # make chain out of input samples and return ECal and HCal structure
        # Add switch to add PU event N times to get N*200 PU
        import glob
        pufiles = glob.glob("/eos/experiment/fcc/hh/simulation/samples/v03/physics/MinBias/bFieldOn/etaFull/ntupPU200/positions/*.root")
        #pufiles=pufiles[1:2]
        #pufiles=['/eos/experiment/fcc/hh/simulation/samples/v03/physics/MinBias/bFieldOn/etaFull/ntupPU200/positions/*.root']
        
        import random
        pu_x_chmapecal=None
        pu_x_chmaphcal=None
        for i in range(self.pu_multi):
            print('adding PU '+str((i+1)*200))
            seed = random.randint(0, 1e7)
            #create PU only once - VERY SLOW
            if not i:
                pu_x_chmapecal=simpleRandom3Dstructure(pufiles,self.nsamples,
                                            xbins=34,
                                            xwidth=2*math.pi/704*34/2,
                                            ybins=34,
                                            ywidth=0.17,
                                            maxlayer=8, minlayer=0, seed=seed)
                
                pu_x_chmaphcal=simpleRandom3Dstructure(pufiles,self.nsamples,
                                            xbins=17,
                                            xwidth=2*math.pi/256*17/2,
                                            ybins=17,
                                            ywidth=0.2125,
                                            maxlayer=18, minlayer=8,seed=seed)
            else:
                #just re-use PU and random shuffle to other events
                from sklearn.utils import shuffle
                pu_x_chmapecal=shuffle(pu_x_chmapecal, random_state=seed)
                pu_x_chmaphcal=shuffle(pu_x_chmaphcal, random_state=seed)
            
            x_chmapecal = numpy.add(x_chmapecal,pu_x_chmapecal)
            x_chmaphcal = numpy.add(x_chmaphcal,pu_x_chmaphcal)
        
        
        if makeplots:
            from plotting import plot4d
            
            
            for i in range(10):
                plot4d(x_chmapecal[i],"ecal"+str(i)+".pdf")
                
                plot4d(x_chmaphcal[i],"hcal"+str(i)+".pdf")
                print('printed '+str(i))
            
            exit()
        
        
        Tuple = self.readTreeFromRootToTuple(filename)  
        
        idtruthtuple =  self.reduceTruth(Tuple[self.truthclasses])
        
        energytruth  =  numpy.array(Tuple[self.regtruth])
        
        
        
        
        #simple by-hand scaling to around 0 with a width of max about 1
        
        
        weights=numpy.zeros(len(idtruthtuple))
        
        notremoves=numpy.zeros(energytruth.shape[0])
        notremoves+=1
        
        #remove muons
        if removemuons:
            ismuon=Tuple['isMuon']
            notremoves[ismuon>0] = 0
            
        
        #no augmentation so far
        #if False and self.remove:
        #    from augmentation import mirrorInPhi,duplicateImage,evaluateTwice
        #    
        #    x_chmapecal= mirrorInPhi(x_chmapecal)
        #    x_chmaphcal= mirrorInPhi(x_chmaphcal)
        #    
        #    notremoves=evaluateTwice(weighter.createNotRemoveIndices,Tuple)
        #    
        #    weights=duplicateImage(weighter.getJetWeights(Tuple))
        #    energytruth   =duplicateImage(energytruth)
        #    idtruthtuple  =duplicateImage(idtruthtuple)
        #    
            #notremoves -= energytruth<50
            
        
        before=len(x_chmapecal)
        
        if self.remove:
            weights=weights[notremoves>0]
            x_chmapecal=x_chmapecal[notremoves>0]
            x_chmaphcal=x_chmaphcal[notremoves>0]
            idtruthtuple=idtruthtuple[notremoves>0]
            energytruth=energytruth[notremoves>0]
        
        print('reduced to '+str(len(x_chmapecal))+' of '+ str(before))
        print('took '+ str(checktime.getAndContinue())+ ' for '+ str(before)+ ' events')
        self.nsamples=len(x_chmapecal)
        
        self.w=[weights,weights]
        self.x=[x_chmapecal,x_chmaphcal]
        self.y=[idtruthtuple,energytruth]
        
        