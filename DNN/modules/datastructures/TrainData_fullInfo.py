
from TrainDataDeepHGCal import TrainDataDeepHGCal,fileTimeOut

class TrainData_fullInfo(TrainDataDeepHGCal):
    
    def __init__(self):
        '''
        
        '''
        import numpy
        TrainDataDeepHGCal.__init__(self)
        
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
        self.weight_binX = numpy.array([0,40,80,120,160,200,240,300,400],dtype=float) 
                                        #4000],dtype=float) 
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
                                  #'isProton',
                                  #'isKaonCharged',
                                  'isOther'
                                  ]
        if tuple_in is not None:
            g = tuple_in['isGamma'].view(numpy.ndarray)
            e = tuple_in['isElectron'].view(numpy.ndarray)
            mu = tuple_in['isMuon'].view(numpy.ndarray)
            pc= tuple_in['isPionCharged'].view(numpy.ndarray)
            
            tau= tuple_in['isTau'].view(numpy.ndarray)
            p0= tuple_in['isPionZero'].view(numpy.ndarray)
            pt= tuple_in['isProton'].view(numpy.ndarray)
            kc= tuple_in['isKaonCharged'].view(numpy.ndarray)
            
            oth= tuple_in['isOther'].view(numpy.ndarray)
            alloth=oth+tau+p0+pt+kc
            #real=q+w
            
            return numpy.vstack((g,e,mu,pc,alloth)).transpose()  
        
        #remove isMuon  isTau  isFake
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        #the first part is standard, no changes needed
        from preprocessing import MeanNormZeroPad
        from converters import createRecHitMap
        import numpy
        import ROOT
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
        
        
        
        x_globalbase = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        
        x_chmapbase=createRecHitMap(filename,self.nsamples,
                                    nbins=13,
                                    width=0.10,
                                    maxlayers=52,
                                    maxhitsperpixel=6)
        
        
        #print(x_chmapbase[0][6][6][15])
        #print(x_chmapbase[0][6][6][14])
        #print(x_chmapbase[0][6][6][13])
        #print(x_chmapbase[0][7][7][13])
        #exit()
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
            from augmentation import mirrorInPhi,duplicateImage,evaluateTwice
            
            x_global=duplicateImage(x_globalbase)
            x_chmap= mirrorInPhi(x_chmapbase)
            
            notremoves=evaluateTwice(weighter.createNotRemoveIndices,Tuple)
            
            weights=duplicateImage(weighter.getJetWeights(Tuple))
            totalrecenergy=duplicateImage(totalrecenergy)
            energytruth   =duplicateImage(energytruth)
            idtruthtuple  =duplicateImage(idtruthtuple)
            notremoves   -=duplicateImage(Tuple['isFake'])
            notremoves   -=duplicateImage(Tuple['isEta'])
            
            #notremoves -= energytruth<50
            
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
        self.nsamples=len(x_global)
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
        
        
        
class TrainData_fullInfo_noremove(TrainData_fullInfo):
    
    def __init__(self):   
        TrainData_fullInfo.__init__(self)
        self.weight_binX = numpy.array([0,4000],dtype=float) 
             


class TrainData_fullInfo_noremove_large(TrainData_fullInfo_noremove):
    
    def __init__(self):
        '''
        
        '''
        
        TrainData_fullInfo_noremove.__init__(self)
        
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        #the first part is standard, no changes needed
        from preprocessing import MeanNormZeroPad
        from converters import createRecHitMap
        import numpy
        import ROOT
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
        
        
        
        x_globalbase = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        
        x_chmapbase=createRecHitMap(filename,self.nsamples,
                                    nbins=21,
                                    width=0.16,
                                    maxlayers=52,
                                    maxhitsperpixel=6)
        
        
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
            from augmentation import mirrorInPhi,duplicateImage,evaluateTwice
            
            x_global=duplicateImage(x_globalbase)
            x_chmap= mirrorInPhi(x_chmapbase)
            
            notremoves=evaluateTwice(weighter.createNotRemoveIndices,Tuple)
            
            weights=duplicateImage(weighter.getJetWeights(Tuple))
            totalrecenergy=duplicateImage(totalrecenergy)
            energytruth   =duplicateImage(energytruth)
            idtruthtuple  =duplicateImage(idtruthtuple)
            notremoves   -=duplicateImage(Tuple['isFake'])
            notremoves   -=duplicateImage(Tuple['isEta'])
            
            #notremoves -= energytruth<50
            
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
        self.nsamples=len(x_global)
        
        self.w=[weights,weights]
        self.x=[x_global,x_chmap,totalrecenergy]
        self.y=[idtruthtuple,energytruth]
