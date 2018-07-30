
#!/bin/zsh

predict.py $1 $2 $3
python $DEEPHGCAL/DNN/Train/Plotting/makePlots_scaled.py $3 $4