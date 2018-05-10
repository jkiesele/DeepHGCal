from trainers.sparse_conv import SparseConvTrainer
import argparse



parser = argparse.ArgumentParser(description='Run training for recurrent cal')
parser.add_argument('input',
                    help="Path to config file")
parser.add_argument('config',
                    help="Config section within the config file")
args = parser.parse_args()

trainer = SparseConvTrainer(args.input, args.config)
trainer.train()