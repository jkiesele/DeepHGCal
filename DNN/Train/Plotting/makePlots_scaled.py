
from DeepJetCore.evaluation import makePlots_async,makeROCs_async
from argparse import ArgumentParser



parser = ArgumentParser('Apply a model to a (test) sample and create friend trees to inject it inthe original ntuple')
parser.add_argument('inputDir')
parser.add_argument('select',default='')
args = parser.parse_args()


infile=args.inputDir+'/tree_association.txt'
outdir=args.inputDir+'/'


#some ROCs
truthclasses=['isGamma',
              'isElectron',
              'isMuon',
              'isNeutralPion',
              'isPionCharged',
              ]


energyrange="true_energy >0&&true_energy <2000"
cuts=[]#[energyrange]
for i in range(len(truthclasses)):
    cuts.append(energyrange+'&&'+truthclasses[i])


if args.select != 'idonly':
    makePlots_async(infile, #input file or file list
                    truthclasses, #legend names [as list]
                    'reg_E/true_energy:true_energy', #variable to plot --> yaxis:xaxis
                    cuts, #list of cuts to apply
                    'auto,dashed', #list of color and style, e.g. ['red,dashed', ...]
                    outdir+'/resolution_ClassProfile.pdf', #output file (pdf)
                    'E_{true} [GeV]', #xaxisname
                    'E_{pred}/E_{true}', #yaxisname
                    False, #normalize
                    True, #make a profile plot
                    0.8, #override min value of y-axis range
                    1.2,
                    treename='events') #override max value of y-axis range
    
    makePlots_async(infile, #input file or file list
                    truthclasses, #legend names [as list]
                    'reg_E/true_energy:true_energy', #variable to plot --> yaxis:xaxis
                    cuts, #list of cuts to apply
                    'auto,dashed', #list of color and style, e.g. ['red,dashed', ...]
                    outdir+'/resolutionwidth_ClassProfile.pdf', #output file (pdf)
                    'E_{true} [GeV]', #xaxisname
                    'E_{pred}/E_{true}', #yaxisname
                    False, #normalize
                    True, #make a profile plot
                    0.1, #override min value of y-axis range
                    2.,
                    treename='events',
                    widthprofile=True) #override max value of y-axis range
    
    
    #widthprofile
    
    
    makePlots_async(infile, #input file or file list
                    truthclasses, #legend names [as list]
                    '(reg_E-true_energy):true_energy', #variable to plot --> yaxis:xaxis
                    cuts, #list of cuts to apply
                    'auto,dashed', #list of color and style, e.g. ['red,dashed', ...]
                    outdir+'/abserror_ClassProfile.pdf', #output file (pdf)
                    'E_{true} [GeV]', #xaxisname
                    'E_{pred} - E_{true}', #yaxisname
                    False, #normalize
                    True, #make a profile plot
                    -100, #override min value of y-axis range
                    100,
                    treename='events') #override max value of y-axis range
    
    
    makePlots_async(infile, #input file or file list
                    truthclasses, #legend names [as list]
                    'reg_E', #variable to plot --> yaxis:xaxis
                    cuts, #list of cuts to apply
                    'auto,dashed', #list of color and style, e.g. ['red,dashed', ...]
                    outdir+'/pred_ClassProfile.pdf', #output file (pdf)
                    'pred energy [GeV]', #xaxisname
                    'E_{pred}', #yaxisname
                    True, #normalize
                    False, #make a profile plot
                    treename='events') #override max value of y-axis range



for part in truthclasses:
    
    cpmclasses=[]
    for compare in truthclasses:
        if part==compare:continue
        cpmclasses.append(compare)
    
    if args.select != 'enonly':
        makeROCs_async(intextfile=infile, 
                       name_list=          cpmclasses, 
                       probabilities_list= 'prob_'+part, 
                       truths_list=        part, 
                       vetos_list=         cpmclasses, 
                       colors_list='auto,dashed', 
                       outpdffile=outdir+'ROC_'+part+ '_vs_all.pdf', 
                       cuts='',
                       treename='events')
    
    if args.select != 'idonly':
    ##energy resolution plot
        xrange=0.5
        if part == 'isMuon':
            xrange=0.8
        elif part == 'isElectron' or part == 'isGamma':
            xrange = 0.06
        makePlots_async(infile, #input file or file list
                ['E < 75 GeV',
                 'E = [75,150] GeV',
                 'E = [150,400] GeV',
                 'E >400 GeV'], #legend names [as list]
                
                4*['reg_E/true_energy - 1'], #variable to plot --> yaxis:xaxis
                [part+'&& true_energy<75',
                 part+'&& true_energy>75 && true_energy<150',
                 part+'&& true_energy>150 && true_energy<400',
                 part+'&& true_energy>400'], #list of cuts to apply
                'auto', #list of color and style, e.g. ['red,dashed', ...]
                outdir+'/resolution_'+ part+'.pdf', #output file (pdf)
                'E_{pred}/E_{true}-1', #xaxisname
                'A.U.', #yaxisname
                normalized=True,
                nbins=37,
                xmin=-xrange,xmax=xrange,
                treename='events') #override max value of y-axis range
        makePlots_async(infile,
                ['E = 50 GeV',
                 'E = 100 GeV',
                 'E = 200 GeV',
                 'E = 500 GeV',
                 'E = 1000 GeV'], #legend names [as list]
                
                5*['reg_E/true_energy - 1'], #variable to plot --> yaxis:xaxis
                [part+'&& abs(true_energy - 50)<5',
                 part+'&& abs(true_energy - 100)<10',
                 part+'&& abs(true_energy - 200)<20',
                 part+'&& abs(true_energy - 500)<50',
                 part+'&& abs(true_energy - 1000)<100'], #list of cuts to apply
                'auto', #list of color and style, e.g. ['red,dashed', ...]
                outdir+'/resolution_fixeden_'+ part+'.pdf', #output file (pdf)
                'E_{pred}/E_{true}-1', #xaxisname
                'A.U.', #yaxisname
                normalized=True,
                nbins=37,
                xmin=-xrange,xmax=xrange,
                treename='events') 
        
    
    if args.select != 'enonly':
        for compare in truthclasses:
            if part==compare:continue
        
        
        
            makeROCs_async(intextfile=infile, 
                       name_list=          ['<200 GeV','200-500 GeV','>500 GeV'], 
                       probabilities_list= 'prob_'+part, 
                       truths_list=        part, 
                       vetos_list=         compare, 
                       colors_list='auto,dashed', 
                       outpdffile=outdir+'ROC_'+part+ '_vs_'+compare+'.pdf', 
                       cuts=['true_energy<200',
                             'true_energy>=200 && true_energy< 500',
                             'true_energy>=500'
                             ],
                       treename='events')
    


exit()

#makePlots_async(intextfile, name_list, variables, cuts, colours,
#                     outpdffile, xaxis='',yaxis='',
#                     normalized=False,profiles=False,
#                     minimum=-1e100,maximum=1e100,widthprofile=False,
#                     treename="deepntuplizer/tree"): 

for t in truthclasses:
    makePlots_async(infile, #input file or file list
                ['MCl: E < 150 GeV',
                 'MCl: E = [150,300] GeV',
                 'MCl: E > 300 GeV',
                 'DNN: E < 150 GeV',
                 'DNN: E = [150,300] GeV',
                 'DNN: E > 300 GeV',], #legend names [as list]
                
                3*['multicluster_energy/true_energy-1']+
                3*['reg_E*100/true_energy-1'], #variable to plot --> yaxis:xaxis
                2*[t+'&& true_energy<150',
                 t+'&& true_energy>150 && true_energy<300',
                 t+'&& true_energy>300'], #list of cuts to apply
                ['red',
                 'darkblue',
                 'darkgreen',
                 'dashed,red',
                 'dashed,darkblue',
                 'dashed,darkgreen',], #list of color and style, e.g. ['red,dashed', ...]
                outdir+'/resolution_'+ t+'.pdf', #output file (pdf)
                'E_{pred}/E_{true}-1', #xaxisname
                'A.U.', #yaxisname
                normalized=True,
                nbins=37,
                xmin=-0.7,xmax=0.7) #override max value of y-axis range
    
    makePlots_async(infile, #input file or file list
                ['MCl: E < 150 GeV',
                 'MCl: E = [150,300] GeV',
                 'MCl: E > 300 GeV',
                 'DNN: E < 150 GeV',
                 'DNN: E = [150,300] GeV',
                 'DNN: E > 300 GeV',], #legend names [as list]
                
                3*['multicluster_energy/true_energy-1']+
                3*['reg_E*100/true_energy-1'], #variable to plot --> yaxis:xaxis
                2*[t+'&& npu>100 && true_energy<150',
                 t+  '&& npu>100&& true_energy>150 && true_energy<300',
                 t+  '&& npu>100&& true_energy>300'], #list of cuts to apply
                ['red',
                 'darkblue',
                 'darkgreen',
                 'dashed,red',
                 'dashed,darkblue',
                 'dashed,darkgreen',], #list of color and style, e.g. ['red,dashed', ...]
                outdir+'/resolution_'+ t+'_highPU.pdf', #output file (pdf)
                'E_{pred}/E_{true}-1', #xaxisname
                'A.U.', #yaxisname
                normalized=True,
                nbins=37,
                xmin=-0.7,xmax=0.7) #override max value of y-axis range
    
    makePlots_async(infile, #input file or file list
                ['MCl: E < 150 GeV',
                 'MCl: E = [150,300] GeV',
                 'MCl: E > 300 GeV',
                 'DNN: E < 150 GeV',
                 'DNN: E = [150,300] GeV',
                 'DNN: E > 300 GeV',], #legend names [as list]
                
                3*['multicluster_energy/true_energy-1']+
                3*['reg_E*100/true_energy-1'], #variable to plot --> yaxis:xaxis
                2*[t+'&& npu<100 && true_energy<150',
                 t+  '&& npu<100&& true_energy>150 && true_energy<300',
                 t+  '&& npu<100&& true_energy>300'], #list of cuts to apply
                ['red',
                 'darkblue',
                 'darkgreen',
                 'dashed,red',
                 'dashed,darkblue',
                 'dashed,darkgreen',], #list of color and style, e.g. ['red,dashed', ...]
                outdir+'/resolution_'+ t+'_lowPU.pdf', #output file (pdf)
                'E_{pred}/E_{true}-1', #xaxisname
                'A.U.', #yaxisname
                normalized=True,
                nbins=37,
                xmin=-0.7,xmax=0.7) #override max value of y-axis range
    
    
    
    #true_drclosestparticle
    
    
    makePlots_async(infile, #input file or file list
                ['DNN: DR > 0.25',
                 'DNN: DR = [0.06,0.25]',
                 'DNN: DR < 0.06',], #legend names [as list]
                
                3*['reg_E*100/true_energy-1'], #variable to plot --> yaxis:xaxis
                 [t+'&&npu>100 &&                            abs(reg_E*100/true_energy-1)<1&& true_drclosestparticle<0',
                 t+'  &&npu>100 &&true_drclosestparticle>0&& abs(reg_E*100/true_energy-1)<1&& true_drclosestparticle>0.06 && true_drclosestparticle<0.25',
                 t+'  &&npu>100 &&true_drclosestparticle>0&& abs(reg_E*100/true_energy-1)<1&& true_drclosestparticle<0.06'], #list of cuts to apply
                ['red',
                 'darkblue',
                 'darkgreen'], #list of color and style, e.g. ['red,dashed', ...]
                outdir+'/overlap_singleDNN_'+ t+'.pdf', #output file (pdf)
                'E_{pred}/E_{true}-1', #xaxisname
                'A.U.', #yaxisname
                normalized=True,
                nbins=37,
                xmin=-0.5,xmax=2) #override max value of y-axis range
    
    makePlots_async(infile, #input file or file list
                ['MCl: DR = [0.10,0.25]',
                 'MCl: DR = [0.06,0.25]',
                 'MCl: DR < 0.06',], #legend names [as list]
                
                3*['multicluster_energy/true_energy-1'], #variable to plot --> yaxis:xaxis
                 [t+'&&npu>100 &&                            abs(multicluster_energy/true_energy-1)<1&& true_drclosestparticle<0',
                 t+'  &&npu>100 &&true_drclosestparticle>0&& abs(multicluster_energy/true_energy-1)<1&& true_drclosestparticle>0.06 && true_drclosestparticle<0.25',
                 t+'  &&npu>100 &&true_drclosestparticle>0&& abs(multicluster_energy/true_energy-1)<1&& true_drclosestparticle<0.06'], #list of cuts to apply
                ['red',
                 'darkblue',
                 'darkgreen'], #list of color and style, e.g. ['red,dashed', ...]
                outdir+'/overlap_singleMCl_'+ t+'.pdf', #output file (pdf)
                'E_{pred}/E_{true}-1', #xaxisname
                'A.U.', #yaxisname
                normalized=True,
                nbins=37,
                xmin=-0.5,xmax=2) #override max value of y-axis range
    

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




makeROCs_async(intextfile=infile, 
                   name_list=['EG vs #pi+','EG vs #mu'], 
                   probabilities_list=['prob_isGamma+prob_isElectron','prob_isGamma+prob_isElectron'], 
                   truths_list=['isGamma+isElectron','isElectron+isElectron'], 
                   vetos_list=['isPionCharged','isMuon'], 
                   colors_list='auto,dashed', 
                   outpdffile=outdir+'ROC_egamma_hadr.pdf', 
                   cuts='true_energy>5')
                  
exit()

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

