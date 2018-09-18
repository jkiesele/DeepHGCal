from trainers.sparse_conv_2 import SparseConvTrainer2
import argparse
from trainers.sparse_conv_multiple_optimizers import SparseConvTrainerMulti


parser = argparse.ArgumentParser(description='Run training for recurrent cal')
parser.add_argument('input', help="Path to config file")
parser.add_argument('config', help="Config section within the config file")
parser.add_argument('--test', default=False, help="Whether to run evaluation on test set")
parser.add_argument('--profile', default=False, help="Whether to run evaluation on test set")
parser.add_argument('--multiple', default=False, help="Whether to run the multi-optimizer trainer")
args = parser.parse_args()


trainer = SparseConvTrainer2(args.input, args.config) if not args.multiple else SparseConvTrainerMulti(args.input, args.config)

if args.test:
    trainer.test()
elif args.profile:
    trainer.profile()
else:
    trainer.train()
