import numpy as np
import numpy as np
import ROOT
import root_numpy
from root_numpy import tree2array, root2array
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cmx
import random
import sys
import timeit

branches = ['particle_eta', 'particle_phi', 'particle_r_origin', 'particle_r_decay',
            'particle_x_origin', 'particle_y_origin', 'particle_z_origin',
            'particle_x_decay', 'particle_y_decay', 'particle_z_decay',
            'isGamma', 'isElectron', 'isMuon', 'isPionCharged', 'true_energy', 'rechit_total_fraction',
            'rechit_energy', 'rechit_phi', 'rechit_eta', 'rechit_x', 'rechit_y', 'rechit_z']




def find_text(isGama, isElectron, isMuon, isPionCharged, trueEnergy):
    s = ""
    if isGama:
        s = "Gamma"
    elif isElectron:
        s = "Electron"
    elif isMuon:
        s = "Muon"
    elif isPionCharged:
        s = "Pion Charged"
    else:
        s = "Unknown"
    return s + " " + trueEnergy + " GeV"

def plot_rechits(A, i, text, figure, all):
    if not all:
        indices = np.where((A['rechit_total_fraction'][i] > 0.5))
        print("Sum >= 0.5 ",np.sum(A['rechit_total_fraction'][i] >= 0.5))
    else:
        indices = np.arange(np.size(A['rechit_total_fraction'][i]))

    print(np.size(indices))


    fig = plt.figure(figure)
    ax = Axes3D(fig)

    particles_x_origin = A['particle_x_origin'][i]
    particles_y_origin = A['particle_y_origin'][i]
    particles_z_origin = A['particle_z_origin'][i]
    particles_x_decay = A['particle_x_decay'][i]
    particles_y_decay = A['particle_y_decay'][i]
    particles_z_decay = A['particle_z_decay'][i]
    particles_r_origin = np.sqrt(particles_x_origin ** 2 + particles_y_origin ** 2 + particles_z_origin ** 2)
    particles_r_decay = np.sqrt(particles_x_decay ** 2 + particles_y_decay ** 2 + particles_z_decay ** 2)
    particles_eta = A['particle_eta'][i]
    particles_phi = A['particle_phi'][i]

    num_particles = np.size(particles_x_origin)

    for j in range(num_particles):
        xs = np.linspace(particles_x_origin[j], particles_x_decay[j])
        ys = np.linspace(particles_y_origin[j], particles_y_decay[j])
        zs = np.linspace(particles_z_origin[j], particles_z_decay[j])
        rs = np.linspace(particles_r_origin[j], particles_r_decay[j])

        etas = np.ones(np.shape(rs)) * particles_eta[j]
        phis = np.ones(np.shape(rs)) * particles_phi[j]
        if plot_in_xyz:
            ax.plot(xs, ys, zs)
        else:
            ax.plot(zs, etas, phis)



    if (np.size(indices)) == 0:
        return

    phi = A['rechit_phi'][i]
    eta = A['rechit_eta'][i]
    x = A['rechit_x'][i]
    y = A['rechit_y'][i]
    z = A['rechit_z'][i]
    energy = np.log(A['rechit_energy'][i] + 1)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    energy = energy[indices]
    r = r[indices]
    eta = eta[indices]
    phi = phi[indices]
    x=x[indices]
    y=y[indices]
    z=z[indices]

    if plot_in_xyz:
        x_min = min(0,np.min(x))
        y_min = min(0,np.min(y))
        z_min = min(0,np.min(z))

        x_max = max(0,np.max(x))
        y_max = max(0,np.max(y))
        z_max = max(0,np.max(z))

        xx, yy = np.meshgrid(np.linspace(x_min,x_max), np.linspace(y_min,y_max))
        zz = 320 * np.ones(np.shape(xx))
        ax.plot_surface(xx, yy, zz, alpha=0.3,cmap=plt.cm.RdYlBu_r)
        ax.plot_surface(xx, yy, -zz, alpha=0.3,cmap=plt.cm.RdYlBu_r)
        ax.scatter(x, y, z, s=np.log(energy+1)*100,  cmap=cmx.hsv)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    else:
        x_min = min(0,np.min(z))
        y_min = np.min(eta)
        z_min = np.min(phi)

        x_max = max(0,np.max(z))
        y_max = np.max(eta)
        z_max = np.max(phi)
        ax.scatter(r, eta, phi, s=np.log(energy+1)*100,  cmap=cmx.hsv)
        ax.set_xlabel('r')
        ax.set_ylabel('eta')
        ax.set_zlabel('phi')



    ax.set_title("Everything "+text)

    ax.set_xbound(x_min, x_max)
    ax.set_ybound(y_min, y_max)
    ax.set_zbound(z_min, z_max)

    ax.set_title(text)

plot_in_xyz = True

if __name__ =="__main__":
    if len(sys.argv) < 2:
        print("Usage: python bin/plot_events path/to/root_file.root")
        exit()

    file = sys.argv[1]
    A = root_numpy.root2array(file, branches=branches)

    for i in range(192):
        text = find_text(A['isGamma'][i], A['isElectron'][i], A['isMuon'][i], A['isPionCharged'][i],
                         str(float(A['true_energy'][i])))

        plot_rechits(A, i, text, 1, False)
        plot_rechits(A, i, text, 2, True)
        plt.show()