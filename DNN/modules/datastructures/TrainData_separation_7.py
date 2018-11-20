from TrainDataDeepHGCal import TrainDataDeepHGCal,fileTimeOut
import numpy as np
import ROOT
import DeepJetCore.preprocessing as prep
from converters import createRecHitMap
from libs.binner import convert_file

class TrainData_separation_7(TrainDataDeepHGCal):
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

        X, Y = convert_file(filename)

        self.nsamples = len(X)
        print("Read", self.nsamples)
        self.w = [np.ones_like(Y)]
        self.x = [X]
        self.y = [Y]

