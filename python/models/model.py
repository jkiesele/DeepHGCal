

class Model:
    error = "Trying to instantiate an abstract class"

    def __init__(self):
        raise NotImplementedError(Model.error)

    def initialize( self ):
        raise NotImplementedError(Model.error)

    def get_summary(self):
        raise NotImplementedError(Model.error)

    def get_placeholders(self):
        raise NotImplementedError(Model.error)

    def get_compute_graphs(self):
        raise NotImplementedError(Model.error)

    def get_losses(self):
        raise NotImplementedError(Model.error)

    def get_optimizer(self):
        raise NotImplementedError(Model.error)