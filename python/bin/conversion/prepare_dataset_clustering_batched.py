import os
import argparse
import numpy as np
import tensorflow as tf

max_gpu_events = 500

parser = argparse.ArgumentParser(description='Run multi-process conversion from root to tfrecords')
parser.add_argument('input',
                    help="Path to file which should contain full paths of all the root files on separate lines")
parser.add_argument('output', help="Path where to produce output files")
parser.add_argument("--isGamma", default=False, help="Whether only gamma", type=bool)
args = parser.parse_args()

with open(args.input) as f:
    content = f.readlines()
file_paths = [x.strip() for x in content]


def _write_entry(data, writer):
    feature = dict()

    main_data = data[0]
    true_values_1 = data[1]
    true_values_2 = data[2]
    feature['data'] = tf.train.Feature( # TODO: Fix this
        float_list=tf.train.FloatList(value=main_data.astype(np.float32).flatten()))
    feature['truth_values_1'] = tf.train.Feature( # TODO: Fix this
        float_list=tf.train.FloatList(value=true_values_1.astype(np.float32).flatten()))
    feature['truth_values_2'] = tf.train.Feature( # TODO: Fix this
        float_list=tf.train.FloatList(value=true_values_2.astype(np.float32).flatten()))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def write_to_tf_records(data, i, output_file):
    writer = tf.python_io.TFRecordWriter(output_file,
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
    np.set_printoptions(threshold=np.nan)

    location = 'B4'
    branches = ['rechit_x', 'rechit_y', 'rechit_z', 'rechit_vxy', 'rechit_vz', 'rechit_energy', 'rechit_layer', 'true_x', 'true_y', 'true_r', 'true_energy', 'isGamma']

    types = ['float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int32']

    # max_size = [2679 for _ in range(7)] + [1, 1, 1, 1, 1]

    from root_numpy import root2array
    nparray = root2array(input_file,
                         treename="B4",
                         branches=branches,
                         # stop=5
                         ).view(np.ndarray)

    print(len(nparray))

    if args.isGamma:
        isGamma = np.where(nparray[11]==1)
        nparray = [x[isGamma] for x in nparray]

    print(np.shape(nparray), nparray.shape, nparray[0].shape)

    true_values_1 = concat_all_branches(nparray, ['true_x', 'true_y', 'true_r', 'true_energy'])

    #common part:
    common = concat_all_branches(nparray,
                                 ['rechit_x', 'rechit_y', 'rechit_z', 'rechit_vxy', 'rechit_vz',
                                  'rechit_layer'])
    
    positions = concat_all_branches(nparray,['rechit_x', 'rechit_y', 'rechit_z'])
    
    energy1 = make_fixed_array(nparray['rechit_energy'],expand=False)
    
    shuffleindices=np.array(range(1,energy1.shape[0]))
    shuffleindices = np.concatenate([shuffleindices, np.array([0])])
    energy2 = energy1[shuffleindices]
    true_values_2 = true_values_1[shuffleindices]
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

    print("Required fraction", np.sum(totdiff==0)/float(np.alen(totdiff)))
    
    #print(totdiff)
    #common = common[totdiff!=0]
    
    esum = energy2+energy1
    print(esum.shape)
    
    
    fraction1 = energy1/esum #np.ma.masked_array(esum, mask=esum==0)
    fraction1[esum==0]=0
    fraction2 = energy2/esum
    fraction2[esum==0]=0
    
    fraction_temp=np.array(fraction1)
    fraction1[totdiff>0] = fraction2[totdiff>0]
    fraction2[totdiff>0] = fraction_temp[totdiff>0]

    true_values_temp = np.array(true_values_1)
    true_values_1[totdiff>0] = true_values_2[totdiff>0]
    true_values_2[totdiff>0] = true_values_temp[totdiff>0]
    
    
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

    # allout = allout[totdiff!=0] #remove same seeded showers

    output_data = []

    for i in range(len(allout)):
        if totdiff[i] != 0:
            output_data.append((allout[i], true_values_1[i], true_values_2[i]))

    
    if firstrun:
        print('output shape ',allout.shape)
        print('last entry in axis 1 is: [idx seed0, idx seed1, 0, ...]')
        print('other are: [esum,rechit_x, rechit_y, rechit_z, rechit_vxy, rechit_vz,rechit_layer,fraction1,fraction2]')
        print('ordering of seed0 and seed1 is done in order by: x,y,z. Events with same positioned seeds are removed')
    
    write_to_tf_records(output_data,0,args.output)


run_conversion_simple(args.input, True)
