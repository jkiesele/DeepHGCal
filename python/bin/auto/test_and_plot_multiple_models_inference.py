import argparse
import os
import subprocess
import configparser as cp
import shutil

parser = argparse.ArgumentParser(description='Plot clustering model output')
parser.add_argument('input', help="Path to the config file which was used to test/train")
parser.add_argument('configs', help="Path to text file containing different configurations to test and plot")
parser.add_argument('combined', help="Path to a directory where to output combined results of everything")
parser.add_argument('--inference', help="Whether to run inference again", default=True)
parser.add_argument('--plot', help="whether to run plots again", default=True)

args = parser.parse_args()

config_file = cp.ConfigParser()
config_file.read(args.input)

def str2bool(v):
    if type(v) == bool:
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

index=0
with open(args.configs) as f:
    content = f.readlines()
    for current_config in content:
        current_config = current_config.strip()
        config = config_file[current_config]

        if str2bool(args.inference):
            command = "python bin/train/sparse_conv_clustering.py %s %s --test True" % (args.input, current_config)
            subprocess.call(command, shell=True)
        if str2bool(args.plot):
            command = "python bin/plot/plot_inference_clustering.py %s %s" % (args.input, current_config)
            subprocess.call(command, shell=True)
            print(command)
        print()
        print()

        for file in os.listdir(config['test_out_path']):
            if file.endswith('.jpg') or file.endswith('.txt') or file.endswith('.png') or file.endswith('.pdf'):
                ext = os.path.splitext(file)[-1]
                directory_path = os.path.join(args.combined, file.replace('.', '_'))
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                shutil.copy(os.path.join(config['test_out_path'], file), os.path.join(directory_path, current_config+ext))


