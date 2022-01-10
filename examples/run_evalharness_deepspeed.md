# How to run lm-eval on Megatron-DeepSpeed checkpoint using the original setup

This particular setup uses the normal deepspeed checkpoint and requires no conversion to Megatron-LM.

This doc assumes usage on JZ, so some peculiar requirements in places. Ignore these if you're not running this on JZ.

## Prerequisites

On login console with external network

Get lm-eval harness (https://github.com/EleutherAI/lm-evaluation-harness) and `best-download==0.0.7` needed to download some tasks.
```
start-prod
pip install lm-eval==0.0.1 best-download==0.0.7
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
arc_challenge,arc_easy,boolq,copa,hellaswag,lambada,logiqa,mathqa,mc_taco,mrpc,multirc,openbookqa,piqa,prost,pubmedqa,qnli,qqp,race,rte,sciq,sst,triviaqa,webqs,wic,winogrande,wnli,wsc
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

Currently it takes 2-3 hours to run on 32GB for 1.3B model, so it should easily fit into 16GB over 20h, but will need a smaller `--micro-batch-size`.

When ready, launch:
```
sbatch ./run_evalharness.slurm
```

Note that the original ETA at the start of the run can be 10x too longer than the actual outcome. For example it may suggest 18 hours but will complete in 2 hours.


## Short eval

if you just want to quickly test that everything can run to the end, edit `tasks/eval_harness/evaluate.py`,  e.g. to run only 10 batches:
```
- results = evaluator.evaluate(adaptor, task_dict, False, 0, None)
+ results = evaluator.evaluate(adaptor, task_dict, False, 0, 10)
```

(XXX: could be a cmd line option so that code won't need to be modified)


## Import into spreadsheet

https://docs.google.com/spreadsheets/d/1CI8Q9RCblLRzUOPJ6ViqBmo284-8ojluQ-CmaEuhuv0/edit?usp=sharing

Note that the spreadsheet format is quite different, so use this script:
```
./tasks/eval_harness/report-to-csv.py results.json
```
to reformat the json results into csv while changing its shape to match the spreadsheet format

Since some records might be missing or extraneous here is the best way to do it:

1. copy the data from first 2 columns to some place under the main spreadsheet

2. put the pointer to the 3rd column next to where the 2 first columns were copied.

3. import `results.csv` using file-> import -> file ->

Import location: Replace data at selected cell

4. Now it should be easy to align the new records with the old ones - delete irrelevant records and Insert->Cells where data is missing until the first 2 columns match

5. now create 2 cols in the main table on top and now it should be safe to Copy-n-Paste the 2-col data range, without the task/metrics columns into the newly created space.
