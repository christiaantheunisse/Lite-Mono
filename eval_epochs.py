import os
import pickle
import argparse

from evaluate_depth_to_npy import evaluate, ParserImitate

# =============== HOW TO RUN ===============
# python eval_epochs.py --models_dir tmp/full_model_30_march_1605/models/  \
#                       --performance_dir tmp/full_model_30_march_1605/test \
#                       --start_epoch 40
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument("--models_dir",
                    help="the directory where the models for every epoch are stored")
parser.add_argument("--performance_dir",
                    help="the folder where the pkl files with the results should be stored")
parser.add_argument("--start_epoch",
                    help="the number of the epoch where the evaluation should start",
                    type=int,
                    default=0)

args = parser.parse_args()

models_dir = args.models_dir
performance_dir = args.performance_dir

print(f"The models are taken from file {models_dir}; starting at epoch{args.start_epoch}")

# Check if the performance_dir exists
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)
    print(f"The directory {performance_dir} is created, because is did not exist")

# Obtain the weight directories in a correctly sorted list
weight_dirs = os.listdir(models_dir)
weight_dirs.sort(key=lambda item: (len(item), item))
weight_dirs.remove("opt.json")

for weight_dir in weight_dirs[args.start_epoch:]:
    # Load the weights
    epoch = weight_dir[8:]
    load_weights_folder = os.path.join(models_dir, weight_dir)
    print(f"Evaluating the model in {load_weights_folder} for epoch {epoch}:")

    # Evaluate the model performance
    options = ParserImitate(load_weights_folder)
    eval_stats = evaluate(options)
    print(f"\t The results are: a1 = {eval_stats['a1']}, a2 = {eval_stats['a2']}, a3 = {eval_stats['a3']}")

    # Save the performance in a dict to a file
    with open(os.path.join(performance_dir, f'epoch_{epoch}.pkl'), 'wb') as f:
        pickle.dump(eval_stats, f)
