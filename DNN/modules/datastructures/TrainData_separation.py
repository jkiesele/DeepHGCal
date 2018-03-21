from TrainDataDeepHGCal import TrainDataDeepHGCal,fileTimeOut
import numpy as np
import ROOT
import DeepJetCore.preprocessing as prep
from converters import createRecHitMap

class TrainData_separation(TrainDataDeepHGCal):
    def __init__(self):
        '''

        '''
        TrainDataDeepHGCal.__init__(self)

        # define truth:
        self.undefTruth = ['']

        self.truthclasses = ['isGamma',
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

        self.weightbranchX = 'true_energy'
        self.weightbranchY = 'true_eta'

        self.referenceclass = 'flatten'

        # this is only needed because the truth definitions are different from deepFlavour
        self.allbranchestoberead = []
        self.registerBranches(self.truthclasses)

        # switch off all flavour rewightings from DeepJet
        self.remove = True
        self.weight = False

        self.registerBranches(['rechit_energy',
                               'rechit_eta',
                               'rechit_phi',
                               'rechit_time',
                               'rechit_layer',
                               'rechit_total_fraction',
                               'nrechits',
                               'seed_eta',
                               'seed_phi',
                               'true_energy', 'true_eta', 'totalrechit_energy'])
        self.addBranches(['seed_eta'])



    def readFromRootFile(self, filename, TupleMeanStd, weighter):

        # the first part is standard, no changes needed
        from DeepJetCore.preprocessing import MeanNormApply, createDensityLayers, createDensityMap, MeanNormZeroPad, \
            MeanNormZeroPadParticles

        fileTimeOut(filename, 120)  # give eos 2 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()

        print(TupleMeanStd[0])
        print(len(TupleMeanStd[0]))

        x_globalbase = MeanNormZeroPad(filename, TupleMeanStd,
                                       [self.branches[0]],
                                       [self.branchcutoffs[0]], self.nsamples)

        # flatten everything out for now
        X = createRecHitMap(filename, self.nsamples,
                                      nbins=13,
                                      width=0.2,
                                      maxlayers=55,
                                      maxhitsperpixel=6)

        Y = createDensityLayers(filename,
                                          TupleMeanStd,
                                          inbranches=['rechit_total_fraction'],
                                          modes=['sum'],
                                          layerbranch='rechit_layer',
                                          maxlayers=55,
                                          layeroffset=1,
                                          nevents=self.nsamples,
                                          dimension1=['rechit_eta', 'seed_eta', 13, 0.2],
                                          dimension2=['rechit_phi', 'seed_phi', 13, 0.2],
                                          counterbranch='nrechits',
                                          scales=[1])

        print("Hey", np.shape(Y), np.shape(X))

        Tuple = self.readTreeFromRootToTuple(filename)

        self.nsamples = len(x_globalbase)
        self.w = [np.ones_like(Y)]
        self.x = [X]
        self.y = [Y]

