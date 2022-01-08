# How to run lm-eval on Megatron-DeepSpeed checkpoint using the original setup

This particular setup uses the normal deepspeed checkpoint and requires no conversion to Megatron-LM.

This doc assumes usage on JZ, so some peculiar requirements in places. Ignore these if you're not running this on JZ.

## Prerequisites

On login console with external network

Get lm-eval harness (https://github.com/EleutherAI/lm-evaluation-harness)
```
start-prod
pip install lm-eval==0.0.1
```
Note: currently @master doesn't work with this script, later may have to edit the hardcoded version above

some symlinks due to lm-harness' issues with relative position of data
```
mkdir data
ln -s data tasks/eval_harness/data
```

Also make sure `data` is not on one of the limited paritions like WORKSF.

Then install datasets for the tasks:
```
python ./tasks/eval_harness/download.py --task_list
arc_challenge,arc_easy,boolq,copa,hellaswag,lambada,logiqa,mathqa,mc_taco,mrpc,multirc,openbookqa,piqa,prost,pubmedqa,qnli,qqp,race,rte,sciq,sst,webqs,wic,winogrande,wnli,wsc
```

Prepare the run script:

```
cp examples/run_evalharness_deepspeed.slurm run_evalharness.slurm
```

now edit `run_evalharness.slurm`

you have to replicate the same config as in the original slurm script but you want:

```
ZERO_STAGE=0
```
and add:
```
export HF_DATASETS_OFFLINE=1
```
if you didn't have one already

Adjust this to fit the GPU, probably ~12 for 32GB and 4-6 for 16GB for 1.3B model
```
EVAL_MICRO_BATCH_SIZE=12
```
Do not modify `MICRO_BATCH_SIZE` which is from the original slurm training script (should remain the same).


## Eval

Currently it takes 8.5h to run on 32GB for 1.3B model, so should probably still fit into 16GB over 20h, but will need a smaller --micro-batch-size

```
srun --account=six@gpu --constraint=v100-32g --nodes=1 --ntasks=1 --cpus-per-task=40 --gres=gpu:1 --hint=nomultithread --time=20:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

```
cd /gpfsssd/worksf/projects/rech/six/commun/code/eval/Megatron-DeepSpeed
PYTHONPATH=. sh ./run_evalharness.sh
```

## Short eval

if you just want to quickly test that everything can run to the end, edit `tasks/eval_harness/evaluate.py`,  e.g. to run only 10 batches:
```
- results = evaluator.evaluate(adaptor, task_dict, False, 0, None)
+ results = evaluator.evaluate(adaptor, task_dict, False, 0, 10)
```

(XXX: could be a cmd line option so that code won't need to be modified)
