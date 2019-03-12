from trainers.sparse_conv_clustering import SparseConvClusteringTrainer
import argparse
from trainers.sparse_conv_multiple_optimizers import SparseConvTrainerMulti


parser = argparse.ArgumentParser(description='Run training for graph based clustering')
parser.add_argument('input', help="Path to config file")
parser.add_argument('config', help="Config section within the config file")
parser.add_argument('--test', default=False, help="Whether to run evaluation on test set")
parser.add_argument('--profile', default=False, help="Whether to run evaluation on test set")
parser.add_argument('--visualize', default=False, help="Whether to run layer wise visualization (x-mode only)")
args = parser.parse_args()


trainer = SparseConvClusteringTrainer(args.input, args.config)

if args.test:
    trainer.test()
elif args.profile:
    trainer.profile()
elif args.visualize:
    trainer.visualize()
else:
    trainer.train()
