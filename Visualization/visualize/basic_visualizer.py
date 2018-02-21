from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cmx
from interface import Interface


class GrabInterestPointsInterface(Interface):

    """
    Grab most interesting values. intensity is a scale that can vary from 0-100
    """
    def find_indices(self, D, intensity):
        pass


class BasicVisualizer:
    def __init__(self, interesting_points_grabber):
        self.D=[]
        self.current_layer = 0
        self.current_channel = 0
        self.interesting_points_grabber = interesting_points_grabber
        self.intensity=10

    """
    Add a new layer of data.
    
    data should be numpy array of shape (4, channels)

    First three coordinates are x,y,z and fourth is the value to be plotted
    
    """
    def add_layer(self, data):
        shape = np.shape(data)
        if len(shape) != 4:
            raise ValueError("Invalid shape of input array. Should be 4-D tensor with last dimension as channels.")
        self.D.append(data)

    def on_key(self, event):
        print(event.key)

        # Up the channel (+1)
        if event.key == 'right':
            self.current_layer = min(self.current_layer + 1, len(self.D) - 1)
        # Down the channel (-1)
        elif event.key == 'left':
            self.current_layer = max(self.current_layer - 1, 0)

        _, _, _, current_layer_channels = np.shape(self.D[self.current_layer])

        self.current_channel = min(self.current_channel, current_layer_channels - 1)

        # Up the channel (+1)
        if event.key == 'up':
            self.current_channel = min(self.current_channel + 1, current_layer_channels - 1)
        # Down the channel (-1)
        elif event.key == 'down':
            self.current_channel = max(self.current_channel - 1, 0)

        # Update the view
        self.update_view()

    def show(self):
        assert len(self.D) != 0
        self.current_layer = 0
        self.update_view(True)


    """
    Displays the plots
    """
    def update_view(self, do_plot=False):
        # Get the required layer
        D_current = self.D[self.current_layer]
        # Get the required channel
        D_current = D_current[:,:,:,self.current_channel]

        I = self.interesting_points_grabber.find_indices(D_current, self.intensity)
        l,s = np.shape(I)

        energy = np.array([float(D_current[tuple(I[x].tolist())]) for x in range(l)])

        x = I[:, 0]
        y = I[:, 1]
        z = I[:, 2]

        cm = plt.get_cmap('Reds')
        cNorm = matplotlib.colors.Normalize(vmin=min(energy), vmax=max(energy))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        fig = plt.figure(1)
        fig.canvas.mpl_connect('key_release_event', self.on_key)
        ax = Axes3D(fig)
        ax.scatter(x, y, z, c=scalarMap.to_rgba(np.exp(energy)))
        scalarMap.set_array(energy)
        fig.colorbar(scalarMap)
        plt.title("Channel %d - Layer - %d" % (self.current_channel + 1, self.current_layer + 1))
        if do_plot:
            plt.show()
        else:
            plt.draw()
