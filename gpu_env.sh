
export DEEPHGCAL=`pwd`
export DEEPJETCORE=`pwd`/../DeepJetCore #/afs/cern.ch/user/j/jkiesele/work/DeepLearning/DJCore/DeepJetCore

cd $DEEPJETCORE
source gpu_env.sh
cd $DEEPHGCAL

export DEEPJETCORE_SUBPACKAGE=$DEEPHGCAL
export PATH=$DEEPHGCAL/DNN/scripts:$PATH
export PYTHONPATH=$DEEPHGCAL/DNN/modules:$DEEPHGCAL/DNN/modules/datastructures:$PYTHONPATH
export PATH=$DEEPHGCAL/Converter/exe:$DEEPHGCAL/Converter/scripts:$PATH
export LD_LIBRARY_PATH=~/.lib:$LD_LIBRARY_PATH

alias plot="python $DEEPHGCAL/DNN/Train/Plotting/makePlots_scaled.py $@"
