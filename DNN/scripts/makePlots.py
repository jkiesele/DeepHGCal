#!/usr/bin/env python

from DeepJetCore.evaluation import makePlots_async
from argparse import ArgumentParser
parser = ArgumentParser('Bla')
parser.add_argument('inputFile')
parser.add_argument('outdir')
args = parser.parse_args()

legends = ['10 GeV',
           '20 GeV',
           '30 GeV',
           '40 GeV',
           '50 GeV',
           '60 GeV',
           '70 GeV',
           '80 GeV',
           '90 GeV',
           '100 GeV',]

truthselection= ['abs(true_energy-10 )<1',
                 'abs(true_energy-20 )<1',
                 'abs(true_energy-30 )<1',
                 'abs(true_energy-40 )<1',
                 'abs(true_energy-50 )<1',
                 'abs(true_energy-60 )<1',
                 'abs(true_energy-70 )<1',
                 'abs(true_energy-80 )<1',
                 'abs(true_energy-90 )<1',
                 'abs(true_energy-100 )<1']

                     
makePlots_async(args.inputFile, #input file or file list
                    legends, #legend names [as list]
                    'reg_E/true_energy', #variable to plot --> yaxis:xaxis
                    truthselection, #list of cuts to apply
                    colours='auto,dashed', #list of color and style, e.g. ['red,dashed', ...]
                    outpdffile=args.outdir+'/response.pdf', #output file (pdf)
                    xaxis='E_{pred}/E_{true}', #xaxisname
                    yaxis='A.U.', #yaxisname
                    normalized=False, #normalize
                    profiles=False, #make a profile plot
                    treename='B4',
                    nbins=50,
                    xmin=0.7,
                    xmax=1.3
                    )
               