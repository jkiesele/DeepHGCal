thisdir=`pwd`
cd ../../../DeepJet/environment
source lxplus_env.sh
cd $thisdir
export DEEPHGCAL=`pwd`/../
export PATH=$DEEPHGCAL/scripts:$PATH
export PYTHONPATH=`pwd`:$PYTHONPATH
export PATH=$DEEPHGCAL/../Converter/exe:$DEEPHGCAL/../Converter/scripts:$PATH
