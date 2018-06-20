from trainers.sparse_conv import SparseConvTrainer
import argparse


parser = argparse.ArgumentParser(description='Run training for recurrent cal')
parser.add_argument('input', help="Path to config file")
parser.add_argument('config', help="Config section within the config file")
parser.add_argument('--test', default=False, help="Whether to run evaluation on test set")
parser.add_argument('--profile', default=False, help="Whether to run evaluation on test set")
args = parser.parse_args()


trainer = SparseConvTrainer(args.input, args.config)

if args.test:
    trainer.test()
elif args.profile:
    trainer.profile()
else:
    trainer.train()
