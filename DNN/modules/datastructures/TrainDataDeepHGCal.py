
from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as djfto

def fileTimeOut(a,b):
    return djfto(a,b)

class TrainDataDeepHGCal(TrainData):
    
    def __init__(self):
        import numpy 
        TrainData.__init__(self)
        
        self.treename="deepntuplizer/tree"
        
        self.weightbranchX='true_energy'
        self.weightbranchY='true_eta'
        
        self.referenceclass='flatten'
        self.weight_binX = numpy.array([0,1,2,3,4,5,7.5,10,20,30,40,
                                        50,60,80,100,120,140,160,200,240,300,400],dtype=float) 
        
        
        
        self.registerBranches(['rechit_energy',
                               'rechit_eta',
                               'rechit_phi',
                               'rechit_time',
                               'rechit_layer',
                               'nrechits',
                               'seed_eta',
                               'seed_phi',
                               'true_energy','true_eta','true_energyfraction'])