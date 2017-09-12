#!/bin/bash

COUNTER=0
OTHERCOUNTER=0

echo ${1}


#exit

for f in ${1}/*.root
do
./exe/convert $f /eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/FlatRandomPtGunProducer_jkiesele_PDGid11_id13_id211_id22_x8_Pt2.0To100_eta_2.3to2.5_timing_20170907/DeepData/t_$COUNTER.root &
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

