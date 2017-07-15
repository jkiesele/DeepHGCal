#!/bin/bash

COUNTER=0
OTHERCOUNTER=0

echo ${1}


#exit

for f in ${1}/*.root
do
./exe/convert $f ~/eos_DeepJet/Jan_test/t_$COUNTER.root&
echo t_$COUNTER.root
COUNTER=$((COUNTER + 1))
OTHERCOUNTER=$((OTHERCOUNTER + 1))
if [ "${OTHERCOUNTER}" -gt "40" ]
then
wait
OTHERCOUNTER=0
fi
done


exit


for f in /eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/FlatRandomEGunProducer_SinglePion_35GeV_20170505/NTUP/*.root
do
./exe/convert $f converted/pion_$COUNTER.root&
COUNTER=$((COUNTER + 1))
done
wait

for f in /eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/FlatRandomEGunProducer_SinglePhoton_35GeV_20170505/NTUP/*.root
do
./exe/convert $f converted/gamma_$COUNTER.root&
COUNTER=$((COUNTER + 1))
done
wait

