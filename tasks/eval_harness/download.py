# Downloads the specified taks in the evaluation harness
# This is particularly useful when running in environments where the GPU nodes 
# do not have internet access. This way we can pre-download them and use the cached data-set during evaluation.

from lm_eval.base import LM
from lm_eval import evaluator, tasks
from lm_eval.tasks import ALL_TASKS
import argparse
import os


parser = argparse.ArgumentParser(description='Download evaluation harness', allow_abbrev=False)
parser.add_argument('--task_list', type=str, default = "all", help='Either "all" or comma separated list of tasks to download.')
parser.add_argument('--save_path', type=str, default = "./task_cache.pickle", help='Path to where the downloaded data tasks will be stored.')
args = parser.parse_args()

import pickle
    
def main():
    task_list = ALL_TASKS if args.task_list == 'all' else args.task_list.split(',')
    task_dict = tasks.get_task_dict(task_list)
    with open(args.save_path, 'wb') as file:
        pickle.dump(task_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tasks have been saved to {args.save_path}!")

if __name__ == '__main__':
    main()


    