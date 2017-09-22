
from testing import makePlots_async,makeROCs_async
from argparse import ArgumentParser



parser = ArgumentParser('Apply a model to a (test) sample and create friend trees to inject it inthe original ntuple')
parser.add_argument('inputDir')
args = parser.parse_args()


infile=args.inputDir+'/tree_association.txt'
outdir=args.inputDir+'/'


#some ROCs
truthclasses=['isGamma',
              'isElectron',
              'isMuon',
              #'isTau',
              #'isPionZero',
              'isPionCharged',
              #'isProton',
              #'isKaonCharged',
              #'isOther'
              ]


energyrange="true_energy >0&&true_energy <500"
cuts=[]#[energyrange]
for i in range(len(truthclasses)):
    cuts.append(energyrange+'&&'+truthclasses[i])
# 1
makePlots_async(infile, #input file or file list
                ['INVISIBLE'], #legend names [as list]
                'reg_E*100/true_energy-1', #variable to plot --> yaxis:xaxis
                'true_energy>100', #list of cuts to apply
                'auto,dashed', #list of color and style, e.g. ['red,dashed', ...]
                outdir+'/pionres_E100.pdf', #output file (pdf)
                'resolution', #xaxisname
                '', #yaxisname
                False, #normalize
                False, #make a profile plot
                0.1, #override min value of y-axis range
                2.) #override max value of y-axis range

# 2
makePlots_async(infile, #input file or file list
                truthclasses, #legend names [as list]
                'reg_E*100/true_energy:true_energy', #variable to plot --> yaxis:xaxis
                cuts, #list of cuts to apply
                'auto,dashed', #list of color and style, e.g. ['red,dashed', ...]
                outdir+'/resolution_ClassProfile.pdf', #output file (pdf)
                'true energy [GeV]', #xaxisname
                'response', #yaxisname
                False, #normalize
                True, #make a profile plot
                0.1, #override min value of y-axis range
                2.) #override max value of y-axis range
# 3
makePlots_async(infile, #input file or file list
                truthclasses, #legend names [as list]
                '(reg_E*100-true_energy):true_energy', #variable to plot --> yaxis:xaxis
                cuts, #list of cuts to apply
                'auto,dashed', #list of color and style, e.g. ['red,dashed', ...]
                outdir+'/abserror_ClassProfile.pdf', #output file (pdf)
                'true energy [GeV]', #xaxisname
                'response', #yaxisname
                False, #normalize
                True, #make a profile plot
                -100, #override min value of y-axis range
                100) #override max value of y-axis range

# 4
makePlots_async(infile, #input file or file list
                truthclasses, #legend names [as list]
                'reg_E*100', #variable to plot --> yaxis:xaxis
                cuts, #list of cuts to apply
                'auto,dashed', #list of color and style, e.g. ['red,dashed', ...]
                outdir+'/pred_ClassProfile.pdf', #output file (pdf)
                'pred energy [GeV]', #xaxisname
                'response', #yaxisname
                True, #normalize
                False, #make a profile plot
                ) #override max value of y-axis range


#makePlots_async(intextfile, name_list, variables, cuts, colours,
#                     outpdffile, xaxis='',yaxis='',
#                     normalized=False,profiles=False,
#                     minimum=-1e100,maximum=1e100,widthprofile=False,
#                     treename="deepntuplizer/tree"):

for t in truthclasses:
    makePlots_async(infile, #input file or file list
                ['E < 150 GeV',
                 'E = [150,300] GeV',
                 'E > 300 GeV',], #legend names [as list]
                'reg_E*100/true_energy', #variable to plot --> yaxis:xaxis
                [t+'&& true_energy<150',
                 t+'&& true_energy>150 && true_energy<300',
                 t+'&& true_energy>300'], #list of cuts to apply
                'auto', #list of color and style, e.g. ['red,dashed', ...]
                outdir+'/resolution_'+ t+'.pdf', #output file (pdf)
                'response', #xaxisname
                'A.U.', #yaxisname
                normalized=True) #override max value of y-axis range
# 'reg_E*100/true_energy-1'
legs=[]
all=''
probs=[]
vetos=[]
for t in truthclasses:
    veto=''
    legs.append(t[2:])
    probs.append('prob_'+t)
    all+=t+'+'
    for t2 in truthclasses:
        if t==t2: continue
        veto+= t2 + '+'
    veto=veto[0:-1]
    vetos.append(veto)

all=all[0:-1]


# 5
##exit()
#for c in truthclasses:
makeROCs_async(intextfile=infile,
                   name_list=legs,
                   probabilities_list=probs,
                   truths_list=truthclasses,
                   vetos_list=vetos,
                   colors_list='auto,dashed',
                   outpdffile=outdir+'ROC_against_all.pdf',
                   cuts='true_energy>5')
                   #, cmsstyle,
                   #firstcomment,
                   #secondcomment,
                   #invalidlist,
                   #extralegend,
                   #logY)
# 6
makeROCs_async(intextfile=infile,
                   name_list=['Gamma','Electrons'],
                   probabilities_list=['prob_isGamma','prob_isElectron'],
                   truths_list=['isGamma','isElectron'],
                   vetos_list=2*['isPionCharged+isProton+isKaonCharged+isOther'],
                   colors_list='auto,dashed',
                   outpdffile=outdir+'ROC_egamma_hadr.pdf',
                   cuts='true_energy>0')
# 7
makeROCs_async(intextfile=infile,
                   name_list=          ['0-5 GeV','5-20 GeV','20-100 GeV','>100 GeV'],
                   probabilities_list= 'prob_isMuon',
                   truths_list=        'isMuon',
                   vetos_list=         'isGamma+isElectron+isPionCharged+isProton+isKaonCharged+isOther',
                   colors_list='auto,dashed',
                   outpdffile=outdir+'ROC_muon_all.pdf',
                   cuts=['true_energy>0&&true_energy<5',
                         'true_energy>5&&true_energy<20',
                         'true_energy>20&&true_energy<100',
                         'true_energy>100'
                         ])

# 8
makeROCs_async(intextfile=infile,
                   name_list=['Gamma','Electrons'],
                   probabilities_list=['prob_isGamma','prob_isElectron'],
                   truths_list=['isGamma','isElectron'],
                   vetos_list=2*['isPionCharged+isProton+isKaonCharged+isOther'],
                   colors_list='auto,dashed',
                   outpdffile=outdir+'ROC_egamma_hadr.pdf',
                   cuts='true_energy>5')

exit()
# 9
makePlots_async(infile,      #input file or file list
                ['DNN: gamma','multicluster: hadron',
                 'DNN: hadron','multicluster: gamma'],    #legend names (needs to be list)
                ['(reg_sigmaE-true_energy)/true_energy',
                 '(multicluster_energy-true_energy)/true_energy',
                 '(reg_sigmaE-true_energy)/true_energy',
                 '(multicluster_energy-true_energy)/true_energy'],    #variable to plot
                ['isGamma &&    (reg_sigmaE-true_energy)/true_energy <10 &&   true_energy > 0 ' ,
                 'isPionCharged&& (multicluster_energy-true_energy)/true_energy <10 &&true_energy > 0 ',
                 'isPionCharged&& (reg_sigmaE-true_energy)/true_energy <10 &&  true_energy > 0 ',
                 'isGamma &&  (multicluster_energy-true_energy)/true_energy <10 &&   true_energy > 0 '
                  ],#cut to apply
                ['green','red','green,dashed','purple,dashed'],     #line color and style (e.g. 'red,dashed')
                args.inputDir+'/resolution.pdf',  #output file (pdf)
                '(E_{reco}-E_{true})/E_{true}',     #xaxisname
                'A.U.',     #yaxisname
                True)       #normalise


# 10
makePlots_async(infile,      #input file or file list
                ['DNN: gamma','multicluster: hadron',
                 'DNN: hadron','multicluster: gamma'],    #legend names (needs to be list)
                ['(reg_sigmaE-true_energy)/true_energy',
                 '(multicluster_energy-true_energy)/true_energy',
                 '(reg_sigmaE-true_energy)/true_energy',
                 '(multicluster_energy-true_energy)/true_energy'],    #variable to plot
                ['true_drclosestparticle > 0.01 && isGamma &&    (reg_sigmaE-true_energy)/true_energy <10 &&   true_energy > 0 ' ,
                 'true_drclosestparticle > 0.01 && isPionCharged&& (multicluster_energy-true_energy)/true_energy <10 &&true_energy > 0 ',
                 'true_drclosestparticle > 0.01 && isPionCharged&& (reg_sigmaE-true_energy)/true_energy <10 &&  true_energy > 0 ',
                 'true_drclosestparticle > 0.01 && isGamma &&  (multicluster_energy-true_energy)/true_energy <10 &&   true_energy > 0 '
                  ],#cut to apply
                ['green','red','green,dashed','purple,dashed'],     #line color and style (e.g. 'red,dashed')
                outdir+'/resolution_dr0.01.pdf',  #output file (pdf)
                '(E_{reco}-E_{true})/E_{true}',     #xaxisname
                'A.U.',     #yaxisname
                True)       #normalise
# 11
makePlots_async(infile,      #input file or file list
                ['DNN: gamma','multicluster: hadron',
                 'DNN: hadron','multicluster: gamma'],    #legend names (needs to be list)
                ['(reg_sigmaE-true_energy)/true_energy',
                 '(multicluster_energy-true_energy)/true_energy',
                 '(reg_sigmaE-true_energy)/true_energy',
                 '(multicluster_energy-true_energy)/true_energy'],    #variable to plot
                ['true_drclosestparticle < 0.01 && isGamma &&    (reg_sigmaE-true_energy)/true_energy <10 &&   true_energy > 0 ' ,
                 'true_drclosestparticle < 0.01 && isPionCharged&& (multicluster_energy-true_energy)/true_energy <10 &&true_energy > 0 ',
                 'true_drclosestparticle < 0.01 && isPionCharged&& (reg_sigmaE-true_energy)/true_energy <10 &&  true_energy > 0 ',
                 'true_drclosestparticle < 0.01 && isGamma &&  (multicluster_energy-true_energy)/true_energy <10 &&   true_energy > 0 '
                  ],#cut to apply
                ['green','red','green,dashed','purple,dashed'],     #line color and style (e.g. 'red,dashed')
                outdir+'/resolution_dr0_0.01.pdf',  #output file (pdf)
                '(E_{reco}-E_{true})/E_{true}',     #xaxisname
                'A.U.',     #yaxisname
                True)       #normalise
# 12
makePlots_async(infile,      #input file or file list
                ['DNN: gamma','multicluster: hadron',
                 'DNN: hadron','multicluster: gamma'],    #legend names (needs to be list)
                ['(reg_sigmaE-true_energy)/true_energy',
                 '(multicluster_energy-true_energy)/true_energy',
                 '(reg_sigmaE-true_energy)/true_energy',
                 '(multicluster_energy-true_energy)/true_energy'],    #variable to plot
                ['true_ncloseparticles > 5 && isGamma &&    (reg_sigmaE-true_energy)/true_energy <10 &&   true_energy > 0 ' ,
                 'true_ncloseparticles > 5 && !isGamma && !isOther && !isFake&& (multicluster_energy-true_energy)/true_energy <10 &&true_energy > 0 ',
                 'true_ncloseparticles > 5 && !isGamma && !isOther && !isFake&& (reg_sigmaE-true_energy)/true_energy <10 &&  true_energy > 0 ',
                 'true_ncloseparticles > 5 && isGamma &&  (multicluster_energy-true_energy)/true_energy <10 &&   true_energy > 0 '
                  ],#cut to apply
                ['green','red','green,dashed','purple,dashed'],     #line color and style (e.g. 'red,dashed')
                outdir+'/resolution_np6.pdf',  #output file (pdf)
                '(E_{reco}-E_{true})/E_{true}',     #xaxisname
                'A.U.',     #yaxisname
                True)       #normalise

# 13
makePlots_async(infile,      #input file or file list
                ['DNN: gamma','multicluster: hadron',
                 'DNN: hadron','multicluster: gamma'],    #legend names (needs to be list)
                ['(reg_sigmaE-true_energy)/true_energy',
                 '(multicluster_energy-true_energy)/true_energy',
                 '(reg_sigmaE-true_energy)/true_energy',
                 '(multicluster_energy-true_energy)/true_energy'],    #variable to plot
                ['true_ncloseparticles > 5 && isGamma &&    (reg_sigmaE-true_energy)/true_energy <10 &&                                 true_energy > 5 ' ,
                 'true_ncloseparticles > 5 && !isGamma && !isOther && !isFake&& (multicluster_energy-true_energy)/true_energy <10 &&true_energy > 5 ',
                 'true_ncloseparticles > 5 && !isGamma && !isOther && !isFake&& (reg_sigmaE-true_energy)/true_energy <10 &&             true_energy > 5 ',
                 'true_ncloseparticles > 5 && isGamma &&  (multicluster_energy-true_energy)/true_energy <10 &&                      true_energy > 5 '
                  ],#cut to apply
                ['green','red','green,dashed','purple,dashed'],     #line color and style (e.g. 'red,dashed')
                outdir+'/resolution_np6_e5.pdf',  #output file (pdf)
                '(E_{reco}-E_{true})/E_{true}',     #xaxisname
                'A.U.',     #yaxisname
                True)       #normalise

# 14
makePlots_async(infile,      #input file or file list
                ['DNN: gamma','multicluster: hadron',
                 'DNN: hadron','multicluster: gamma'],    #legend names (needs to be list)
                ['(reg_sigmaE-true_energy)/true_energy',
                 '(multicluster_energy-true_energy)/true_energy',
                 '(reg_sigmaE-true_energy)/true_energy',
                 '(multicluster_energy-true_energy)/true_energy'],    #variable to plot
                ['true_ncloseparticles > 5 && isGamma &&    (reg_sigmaE-true_energy)/true_energy <10 &&                                 true_energy > 20 ' ,
                 'true_ncloseparticles > 5 && !isGamma && !isOther && !isFake&& (multicluster_energy-true_energy)/true_energy <10 &&true_energy > 20 ',
                 'true_ncloseparticles > 5 && !isGamma && !isOther && !isFake&& (reg_sigmaE-true_energy)/true_energy <10 &&             true_energy > 20 ',
                 'true_ncloseparticles > 5 && isGamma &&  (multicluster_energy-true_energy)/true_energy <10 &&                      true_energy > 20 '
                  ],#cut to apply
                ['green','red','green,dashed','purple,dashed'],     #line color and style (e.g. 'red,dashed')
                outdir+'/resolution_np6_e20.pdf',  #output file (pdf)
                '(E_{reco}-E_{true})/E_{true}',     #xaxisname
                'A.U.',     #yaxisname
                True)       #normalise
# 15
makePlots_async(infile,      #input file or file list
                ['DNN: #pi^{#pm}','multicluster: #pi^{#pm}',
                 ],    #legend names (needs to be list)
                ['(reg_sigmaE-true_energy)/true_energy',
                 '(multicluster_energy-true_energy)/true_energy'],    #variable to plot
                [
                 'isPionCharged&& true_energy > 100 ',
                 'isPionCharged&&   true_energy > 100 ',

                  ],#cut to apply
                ['green','red'],     #line color and style (e.g. 'red,dashed')
                outdir+'/resolution_true_E100.pdf',  #output file (pdf)
                '(E_{reco}-E_{true})/E_{true}',     #xaxisname
                'A.U.',     #yaxisname
                True)       #normalise


# 16
makePlots_async(infile,      #input file or file list
                ['DNN: #pi^{#pm}','simcluster: #pi^{#pm}',
                 ],    #legend names (needs to be list)
                ['(reg_sigmaE-true_energy)/true_energy',
                 '(simcluster_energy-true_energy)/true_energy'],    #variable to plot
                [
                 'isPionCharged&& true_energy > 100 ',
                 'isPionCharged&&   true_energy > 100 ',

                  ],#cut to apply
                ['green','red'],     #line color and style (e.g. 'red,dashed')
                outdir+'/resolutionsimcl_true_E100.pdf',  #output file (pdf)
                '(E_{reco}-E_{true})/E_{true}',     #xaxisname
                'A.U.',     #yaxisname
                True)       #normalise
# 17
makePlots_async(infile,      #input file or file list
                ['DNN: gamma','DNN: electron',
                 'DNN: hadron'],    #legend names (needs to be list)
                ['(reg_sigmaE-true_energy)/reg_uncPt',
                 '(reg_sigmaE-true_energy)/reg_uncPt',
                 '(reg_sigmaE-true_energy)/reg_uncPt'],    #variable to plot
                ['isGamma &&  true_energy > 0 ' ,
                  'isElectron &&  true_energy > 0 ',
                 ' !isGamma && !isOther && true_energy > 0 ',

                  ],#cut to apply
                ['green','green,dashed','red'],     #line color and style (e.g. 'red,dashed')
                outdir+'/pull.pdf',  #output file (pdf)
                '(E_{reco}-E_{true})/#sigma_{rec}',     #xaxisname
                'A.U.',     #yaxisname
                True)
# 18
makePlots_async(infile, #input file or file list
                ['inclusive','Photon','PionCharged'], #legend names [as list]
                'reg_uncPt/true_energy:true_energy', #variable to plot --> yaxis:xaxis
                ['true_energy>0',
                 "true_energy>0&& isGamma",
                 "true_energy>0 && isPionCharged"], #list of cuts to apply
                ['black','green','red'], #list of color and style, e.g. ['red,dashed', ...]
                outdir+'/uncertainty_Profile.pdf', #output file (pdf)
                'true energy [GeV]', #xaxisname
                '#sigma_{rec}/E_{true}', #yaxisname
                False, #normalize
                True, #make a profile plot
                0., #override min value of y-axis range
                0.5) #override max value of y-axis range

# 19
makePlots_async(infile, #input file or file list
                ['inclusive','Photon','PionCharged'], #legend names [as list]
                'reg_sigmaE/true_energy:true_energy', #variable to plot --> yaxis:xaxis
                [energyrange,
                 energyrange+ "&& isGamma",
                 energyrange+ "&& isPionCharged"], #list of cuts to apply
                ['black','green','red'], #list of color and style, e.g. ['red,dashed', ...]
                outdir+'/resolution_Profile.pdf', #output file (pdf)
                'true energy [GeV]', #xaxisname
                'response', #yaxisname
                False, #normalize
                True, #make a profile plot
                0.2, #override min value of y-axis range
                1.8) #override max value of y-axis range
# 20
makePlots_async(infile,      #input file or file list
                ['DNN: gamma',
                 'DNN: hadron'],    #legend names (needs to be list)
                [
                 'reg_uncPt',

                 'reg_uncPt'],    #variable to plot
                ['isGamma && prob_isGamma>0.9' ,

                 'isPionCharged && prob_isPionCharged>0.9',

                  ],#cut to apply
                ['green','red'],     #line color and style (e.g. 'red,dashed')
                outdir+'/sigma_reco_highpur.pdf',  #output file (pdf)
                '(E_{reco}-E_{true})/E_{true}',     #xaxisname
                'A.U.',     #yaxisname
                True)
# 21
makePlots_async(infile,      #input file or file list
                ['DNN: gamma',
                 'DNN: hadron'],    #legend names (needs to be list)
                [
                 'reg_sigmaE',

                 'reg_sigmaE'],    #variable to plot
                ['isGamma ' ,

                 'isPionCharged',

                  ],#cut to apply
                ['green','red'],     #line color and style (e.g. 'red,dashed')
                outdir+'/regEnergy.pdf',  #output file (pdf)
                'E_{reco}',     #xaxisname
                'A.U.',     #yaxisname
                True)
# 22
makePlots_async(infile,      #input file or file list
                ['#gamma',
                 '#pi^{#pm}','#pi^{0}'],    #legend names (needs to be list)
                [
                 'true_energy',
                 'true_energy',
                 'true_energy'],    #variable to plot
                ['isGamma ' ,

                 'isPionCharged','isPionZero'

                  ],#cut to apply
                ['green','red','blue'],     #line color and style (e.g. 'red,dashed')
                outdir+'/trueEnergy.pdf',  #output file (pdf)
                'E_{true}',     #xaxisname
                'events',     #yaxisname
                False)
# 23
makePlots_async(infile,      #input file or file list
                ['DNN: pion charged'],    #legend names (needs to be list)
                ['(reg_sigmaE-true_energy)/true_energy'
                 ],    #variable to plot
                ['isPionCharged',

                  ],#cut to apply
                ['green'],     #line color and style (e.g. 'red,dashed')
                outdir+'/test4.pdf',  #output file (pdf)
                'E:(E_{reco}-E_{true})/E_{true}',     #xaxisname
                'A.U.',     #yaxisname
                True)

