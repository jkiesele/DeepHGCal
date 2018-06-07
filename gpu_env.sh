

export DEEPHGCAL=`pwd`
export DEEPJETCORE=`pwd`/../DeepJetCore

cd $DEEPJETCORE
source gpu_env.sh
cd $DEEPHGCAL

export PATH=$DEEPHGCAL/scripts:$PATH
export PYTHONPATH=$DEEPHGCAL/DNN/modules:$PYTHONPATH
export PATH=$DEEPHGCAL/Converter/exe:$DEEPHGCAL/Converter/scripts:$PATH
export LD_LIBRARY_PATH=~/.lib:$LD_LIBRARY_PATH
