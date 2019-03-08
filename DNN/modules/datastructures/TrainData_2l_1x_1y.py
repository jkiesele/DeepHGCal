from TrainData_miniCalo import TrainData_miniCalo


class TrainData_2l_1x_1y(TrainData_miniCalo):
    
    def __init__(self):
        import numpy 
        TrainData_miniCalo.__init__(self)
        
        
        self.nbinsx = 1
        self.nbinsy = 1
        
        self.nbinsl = 2