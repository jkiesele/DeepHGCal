
from TrainData import TrainData,fileTimeOut

class TrainData_small_corr(TrainData):
    
    def __init__(self):
        '''
        
        '''
        import numpy
        TrainData.__init__(self)
        
        #define truth:
        self.undefTruth=['']
    
        self.truthclasses=['isGamma',
                            'isElectron',
                            'isMuon',
                            'isTau',
                            'isPionZero',
                            'isPionCharged',
                            'isProton',
                            'isKaonCharged',
                            'isEta',
                            'isOther',
                            'isFake']
       
        
        self.weightbranchX='true_energy'
        self.weightbranchY='true_eta'
        
        self.referenceclass='flatten'
        self.weight_binX = numpy.array([0,1,2,3,4,5,7.5,10,20,30,40,
                                        50,60,80,100,120,140,160,200,240,300,400],dtype=float) 
        #allow for falling spectrum after 400
        
        self.weight_binY = numpy.array([-10.,10.], dtype=float )
        
        #this is only needed because the truth definitions are different from deepFlavour
        self.allbranchestoberead=[]
        self.registerBranches(self.truthclasses)
        
        #switch off all flavour rewightings from DeepJet
        self.remove=True    
        self.weight=False
        
        self.registerBranches(['rechit_energy',
                               'rechit_eta',
                               'rechit_phi',
                               'rechit_time',
                               'rechit_layer',
                               'nrechits',
                               'seed_eta',
                               'seed_phi',
                               'true_energy','true_eta','totalrechit_energy'])
        ## standard part
        
        self.regtruth='true_energy'
        
        self.addBranches(['seed_eta'])

        self.regressiontargetclasses=['E']
    
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['isGamma',
                                  'isElectron',
                                  'isMuon',
                                  #'isTau',
                                  #'isPionZero',
                                  'isPionCharged',
                                  'isProton',
                                  'isKaonCharged',
                                  'isOther']
        if tuple_in is not None:
            g = tuple_in['isGamma'].view(numpy.ndarray)
            e = tuple_in['isElectron'].view(numpy.ndarray)
            mu = tuple_in['isMuon'].view(numpy.ndarray)
            tau= tuple_in['isTau'].view(numpy.ndarray)
            p0= tuple_in['isPionZero'].view(numpy.ndarray)
            pc= tuple_in['isPionCharged'].view(numpy.ndarray)
            pt= tuple_in['isProton'].view(numpy.ndarray)
            kc= tuple_in['isKaonCharged'].view(numpy.ndarray)
            oth= tuple_in['isOther'].view(numpy.ndarray)
            alloth=oth#+tau+p0
            #real=q+w
            
            return numpy.vstack((g,e,mu,pc,pt,kc,alloth)).transpose()  
        
        #remove isMuon  isTau  isFake
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        #the first part is standard, no changes needed
        from preprocessing import MeanNormApply,createDensityLayers, createDensityMap, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        import ROOT
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
        
        
        
        x_globalbase = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        
        
        #flatten everything out for now
        x_chmapbase = createDensityLayers(filename,
                                      TupleMeanStd,
                                      inbranches=['rechit_energy','rechit_layer','rechit_seeddr'], 
                                      modes=['sum','single','average'],
                                      layerbranch='rechit_layer',
                                      maxlayers=55,
                                      layeroffset=1,
                                      nevents=self.nsamples,
                                      dimension1=['rechit_eta','seed_eta',17,0.25], 
                                      dimension2=['rechit_phi','seed_phi',17,0.25],
                                      counterbranch='nrechits',
                                      scales=[1,50,0.25])
        
        
        #training data
        
        Tuple = self.readTreeFromRootToTuple(filename)  
        
        idtruthtuple =  self.reduceTruth(Tuple[self.truthclasses])
        energytruth  =  numpy.array(Tuple[self.regtruth])
        #simple by-hand scaling to around 0 with a width of max about 1
        energytruth = energytruth/100.
        
        totalrecenergy=numpy.array(Tuple['totalrechit_energy'])/100.
        
        weights=numpy.zeros(len(idtruthtuple))
        
        notremoves=numpy.zeros(totalrecenergy.shape[0])
        notremoves+=1
        if self.remove:
            from augmentation import augmentRotationalSymmetry6,duplicate6,evaluate6
            
            x_global=duplicate6(x_globalbase)
            x_chmap= augmentRotationalSymmetry6(x_chmapbase)
            
            notremoves=evaluate6(weighter.createNotRemoveIndices,Tuple)
            
            weights=duplicate6(weighter.getJetWeights(Tuple))
            totalrecenergy=duplicate6(totalrecenergy)
            energytruth   =duplicate6(energytruth)
            idtruthtuple  =duplicate6(idtruthtuple)
            notremoves   -=duplicate6(Tuple['isFake'])
            notremoves   -=duplicate6(Tuple['isEta'])
        else:
            notremoves-=Tuple['isFake']
            notremoves-=Tuple['isEta']
            x_global=x_globalbase
            x_chmap=x_chmapbase 
        
        
        before=len(x_global)
        
        if self.remove:
            weights=weights[notremoves>0]
            x_global=x_global[notremoves>0]
            x_chmap=x_chmap[notremoves>0]
            idtruthtuple=idtruthtuple[notremoves>0]
            energytruth=energytruth[notremoves>0]
            totalrecenergy=totalrecenergy[notremoves>0]
        
        print('reduced to '+str(len(x_global))+' of '+ str(before))
        
        #make control plot for energy
        #import matplotlib.pyplot as plt
        #plt.hist(energytruth.flatten(), normed=False, bins=30)
        #plt.savefig(giffile+"_eshape.pdf")
        #from plotting import plot4d, rotanimate
        #giffile=filename.replace('/','_')
        #giffile='gifs/'+giffile
        #for i in range(0,len(select)):
        #    if not select[i]: continue
        #    
        #    ax,_=plot4d(x_chmap[i][:,:,:,:1],giffile+"_"+str(i)+"energy_.pdf",'etabin','layer','phibin')
        #    rotanimate(ax,giffile+'_'+str(i)+'_energy.gif',delay=5,prefix=giffile)
        #    print('energy')
        #    timeentries=x_chmap[i][:,:,:,3:4]
        #    timeentries[timeentries<0]=0.00000000001
        #    ax,_=plot4d(timeentries,giffile+"_"+str(i)+"time_.pdf",'etabin','layer','phibin')
        #    rotanimate(ax,giffile+'_'+str(i)+'_time.gif',delay=5,prefix=giffile)
        #    print('time')
        
        self.w=[weights,weights]
        self.x=[x_global,x_chmap,totalrecenergy]
        self.y=[idtruthtuple,energytruth]
        
        
        
        
        
        
        
