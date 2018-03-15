from python.trainers.recurrent_cal import RecurrentCalTrainer # TODO: Might want to fix python thing
import argparse


parser = argparse.ArgumentParser(description='Run training for recurrent cal')
parser.add_argument('input',
                    help="Path to config file")
args = parser.parse_args()

trainer = RecurrentCalTrainer(args.input, 'recurrent_cal')
trainer.train()