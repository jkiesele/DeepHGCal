import os
import sys
from Queue import Queue # TODO: Write check for python3 (module name is queue in python3)
import argparse
from threading import Thread
import time
from collect_few import convert

parser = argparse.ArgumentParser(description='Run multi-process conversion from root to tfrecords')
parser.add_argument('input',
                    help="Input file")
parser.add_argument('output', help="Path where to produce output files")
parser.add_argument('particle', help="Type")
parser.add_argument("--pileup", default=1, help="Pileup")

args = parser.parse_args()


def run_conversion(input_file):
    just_file_name = os.path.splitext(os.path.split(input_file)[1])[0] + '_'

    convert(input_file, os.path.join(args.output, just_file_name), int(args.particle), int(args.pileup))

run_conversion(args.input)