
export DEEPHGCAL=`pwd`
export DEEPJETCORE=`pwd`/../DeepJetCore

cd $DEEPJETCORE
source lxplus_env.sh
cd $DEEPHGCAL

export DEEPJETCORE_SUBPACKAGE=$DEEPHGCAL
export PATH=$DEEPHGCAL/DNN/scripts:$PATH
export PYTHONPATH=$DEEPHGCAL/DNN/modules:$PYTHONPATH
export PATH=$DEEPHGCAL/Converter/exe:$DEEPHGCAL/Converter/scripts:$PATH
export LD_LIBRARY_PATH=~/.lib:$LD_LIBRARY_PATH