export DEEPHGCAL=`pwd`
export DEEPJETCORE=../DeepJetCore

cd $DEEPJETCORE
source lxplus_env.sh
cd $DEEPHGCAL

export PATH=$DEEPHGCAL/scripts:$PATH
export PYTHONPATH=$DEEPHGCAL/DNN/modules:$PYTHONPATH
export PATH=$DEEPHGCAL/Converter/exe:$DEEPHGCAL/Converter/scripts:$PATH
