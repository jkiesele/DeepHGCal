thisdir=`pwd`
cd ../../../DeepJet/environment
source gpu_env.sh
cd $thisdir
export PYTHONPATH=`pwd`:$PYTHONPATH
