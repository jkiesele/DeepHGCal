

COUNTER=1



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

