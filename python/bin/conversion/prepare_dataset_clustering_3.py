import os
import sys
from queue import Queue  # TODO: Write check for python3 (module name is queue in python3)
import argparse
from threading import Thread
import time
import numpy as np
import sys
import tensorflow as tf
import time
from sklearn.utils import shuffle

max_gpu_events = 500

parser = argparse.ArgumentParser(description='Run multi-process conversion from root to tfrecords')
parser.add_argument('input',
                    help="Path to file which should contain full paths of all the root files on separate lines")
parser.add_argument('output', help="Path where to produce output files")
parser.add_argument("--jobs", default=4, help="Number of processes")
args = parser.parse_args()

with open(args.input) as f:
    content = f.readlines()
file_paths = [x.strip() for x in content]


def plot4d(x,y,z,c, outname, xlabel='', ylabel='', zlabel='',areas=np.array([])):

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    from pylab import cm

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    if len(areas):
        c=c/areas
    # c=c/c.max()
    # c*=100
    s = c /c.max()*100
    from math import log10
    s = s 
    for i in range(len(s)):
        c[i]=log10(c[i]+1)
        s[i]=log10(s[i]+1)
        
    
    #sarr=c>0.06914700
    #sarr2=c<0.06914702
    #sarr=sarr&sarr2
    #s[sarr]=100
    #print(c)
        
    #colmap = cm.ScalarMappable(cmap=cm.hsv)
    #colmap.set_array(c)
    
    #s=s+1
    ax.scatter(x, z, y, c=cm.hot(c),s=s)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    
    
    #fig.axes.get_xaxis().set_visible(False)
    #fig.axes.get_yaxis().set_visible(False)
    #fig.axes.get_zaxis().set_visible(False)
    if len(outname):
        fig.savefig(outname)
    #plt.show()
    return ax,plt,x,y,z,c
    
    #plt.close()


def _write_entry(data, writer):
    feature = dict()
    feature['data'] = tf.train.Feature( # TODO: Fix this
        float_list=tf.train.FloatList(value=data.astype(np.float32).flatten()))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def write_to_tf_records(data, i, output_file_prefix):
    writer = tf.python_io.TFRecordWriter(output_file_prefix + '.tfrecords',
                                         options=tf.python_io.TFRecordOptions(
                                             tf.python_io.TFRecordCompressionType.GZIP))
    for i in range(len(data)):
        _write_entry(data[i], writer)
        print("Written", i)

    writer.close()

def make_fixed_array(a,expand=True):
    if expand:
        return np.expand_dims(np.array(a.tolist(), dtype='float32'), axis=2)
    else:
        return np.array(a.tolist(), dtype='float32')

def concat_all_branches(nparray,branches):
    allarrays=[]
    for b in branches:
        allarrays.append(make_fixed_array(nparray[b]))
    return np.concatenate(allarrays,axis=-1)
    

def run_conversion_simple(input_file, firstrun=False):
    
    
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    
    np.set_printoptions(threshold=np.nan)
    
    
    usebranches = ['rechit_x', 'rechit_y', 'rechit_z', 'rechit_vxy', 'rechit_vz', 'rechit_energy', 'rechit_layer',
                   'true_x','true_y']
    from root_numpy import  root2array
    nparray = root2array(input_file, 
                treename = "B4", 
                branches = usebranches,
                #stop=5
                ).view(np.ndarray)
    
    print(len(nparray))
    #common part:
    common = concat_all_branches(nparray,
                                 ['rechit_x', 'rechit_y', 'rechit_z', 'rechit_vxy', 'rechit_vz','rechit_layer'])
    
    positions = concat_all_branches(nparray,['rechit_x', 'rechit_y', 'rechit_z'])
    
    energy1 = make_fixed_array(nparray['rechit_energy'],expand=False)
    
    shuffleindices=np.array(range(1,energy1.shape[0]))
    shuffleindices = np.concatenate([shuffleindices, np.array([0])])
    energy2 = energy1[shuffleindices] 
    print(energy1.shape)
    maxenergyids1 = energy1.argmax(axis=1)
    maxenergyids2 = energy2.argmax(axis=1)
    
    positions1 = positions[range(energy1.shape[0]),maxenergyids1]
    positions2 = positions[range(energy2.shape[0]),maxenergyids2]
    
    
    #np.logical_or(a, b)
    diff = positions1-positions2
    diff[:,0] *= 1e6
    diff[:,1] *= 1e3
    totdiff = diff[:,0]+ diff[:,1] + diff[:,2]
    
    #print(totdiff)
    #common = common[totdiff!=0]
    
    
    #for i in range(4):
    #    plot4d(positions[i][:,0],positions[i][:,1],positions[i][:,2],energy1[i],"out"+ str(i) +".pdf",
    #       areas=np.square(common[0][:,3])*common[0][:,4])
    

    
    esum = energy2+energy1
    print(esum.shape)
    
    
    fraction1 = energy1/esum #np.ma.masked_array(esum, mask=esum==0)
    fraction1[esum==0]=0
    fraction2 = energy2/esum
    fraction2[esum==0]=0
    
    fraction_temp=np.array(fraction1)
    fraction1[totdiff>0] = fraction2[totdiff>0]
    fraction2[totdiff>0] = fraction_temp[totdiff>0]
    
    
    #prepare additional information about the seeds
    #BX1 maxenergy1, make same ordering
    maxenergyids1_temp=np.array(maxenergyids1)
    maxenergyids1[totdiff>0] = maxenergyids2[totdiff>0]
    maxenergyids2[totdiff>0] = maxenergyids1_temp[totdiff>0]
    
    maxenergyids1=np.expand_dims(maxenergyids1, axis=1)
    maxenergyids2=np.expand_dims(maxenergyids2, axis=1)
    
    moreinfo = np.concatenate([maxenergyids1,maxenergyids2],axis=-1)

    esum = np.expand_dims(esum,axis=2)
    fraction1 = np.expand_dims(fraction1,axis=2)
    fraction2 = np.expand_dims(fraction2,axis=2)
    
    
    allout = np.concatenate([esum,common,fraction1,fraction2],axis=-1)

    zeropad = np.zeros(shape=(moreinfo.shape[0],allout.shape[2]-moreinfo.shape[1]))
    moreinfo = np.concatenate([moreinfo,zeropad],axis=-1)
    moreinfo = np.expand_dims(moreinfo,axis=1)
    
    allout = np.concatenate([allout, moreinfo],axis=1)
    
    allout = allout[totdiff!=0] #remove same seeded showers
    
    if firstrun:
        print('output shape ',allout.shape)
        print('last entry in axis 1 is: [idx seed0, idx seed1, 0, ...]')
        print('other are: [esum,rechit_x, rechit_y, rechit_z, rechit_vxy, rechit_vz,rechit_layer,fraction1,fraction2]')
        print('ordering of seed0 and seed1 is done in order by: x,y,z. Events with same positioned seeds are removed')
    
    just_file_name = os.path.splitext(os.path.split(input_file)[1])[0] + '_'
    output_file_prefix = os.path.join(args.output, just_file_name)
    write_to_tf_records([allout],0,output_file_prefix)
    


start = time.time()
firstrun=True
for item in file_paths:
    run_conversion_simple(item,firstrun)
    firstrun=False
end = time.time()

print("It took", end - start, "seconds")