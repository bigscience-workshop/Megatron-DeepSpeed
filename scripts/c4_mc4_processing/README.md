# C4 & mC4 Data Processing for Megatron

We export text data in `*.jsonl` for Megatron-LM script `tools/preprocess_data.py`.

## Processing data with HF-Datasets

### Caching Data
HF-`datasets` host full `c4` (around 1TB) and `mC4` (around 27TB) dataset. To prepare that dataset for Megatron, at first, we cache them using, 

```
bash scripts/c4_mc4_processing/cache_c4_mc4.sh
```
Running this script may require bandwidth `~30MiB/s` per language. You may run this script multiple times. Sometimes there is cap for a single download request. So in that case you can just run the script multiple time to do parallel downloading and processing. Also once the data is downloaded, it starts caching. Caching may take a lot of time. In that time other process can keep downloading their stuffs. The script `tools/c4_mc4/c4_mc4_cache.py` will perform caching if a caching folder for a language doesn't exist. So running the script multiple times with the same cache folder will be ok. Make sure you add your desired language in the script.


> :warning: **Caching failure**: There are lots of ways the caching mechanism may fail. Some of the common reason is hard disk space and per-day bandwidth limitation to your broadband provider. Make sure that you have a reasonable amount of HD space and your system supports downloading a large chunk of data from the internet. If caching fails sometimes it's difficult to recover. Please follow this [issue](https://github.com/huggingface/datasets/issues/1706).

> :warning: **Caching Time**: Please note that at this moment, caching is a sequential process in HF-datasets. After downloading the data, it takes a lot of time to do the caching. It may require 2-5 days to complete the caching. But if you want to perform an additional operation (tokenization, additional pre-processing stuffs etc) on the data, caching is very preferable.

### Calculate Sampling Probability

Most of the time we are not going to train the language model with the full data (27TB data in this case). Following [Raffel et al.](https://arxiv.org/abs/1910.10683), [Lample et al.](https://arxiv.org/abs/1901.07291) we sample our selected data using a multinomial distribution with some additional conditions.

Notes, 

1. At first we define a low resource language. We sum up the total size of the dataset and take the mean size for each of the languages. The language which has a low amount of data compared to the mean size is considered low resource language.
2. For low resource language, we take the full datasets (sampling probability 1.0).
3. For high resource language, we set two parameters, `min_high_resource_size` and `max_high_resource_size`. 
4. We set a parameter `new-expected-size` which is the estimated size of data that will be sampled from the full datasets (dataset combining all the languages).
5. Finally we calculate the sampling probability for each of the datasets.

> :warning: **Dataset Size**: It is to be noted that datasets are cached in `gzip (level 6)` format and after writing them in a preferable encoding method, the size may vary a lot. But regardless of this, it gives a comparable estimation of the dataset to be sampled. This step is done without the `english` language. For english, we sample the full data (sampling prob `1.0`) from `c4`. Please note that we didn't use English mC4 which is around `~10TB` of data.

The processing script for this is, 

```
bash scripts/c4_mc4_processing/data_resize.sh
```
The script will export a `*.json` file to `--output-dir` which will contain the sampling probability for each of the languages.

### Extract data from the cache.

In this step, we extract data from Hf-`datasets` and write them in a `*.jsonl`. 

```
bash scripts/c4_mc4_processing/extract_from_hf_dataset.sh
```
Please note that this process is also sequential and it may take some time (1-3 days) to complete. If you cache your data in SSD, this process might be a lot faster.

### Preprocess data

Preprocess data using,
```
bash scripts/c4_mc4_processing/single_shard_data_process.sh
```

## Processing data from git lfs

For some unknown reason (???) I was stuck (in the `Extract data from cache.` stage) processing `russian` language (~3TB of data) in my system. So in the later stage, I focus on the source of the data. In the original [git lfs repo](https://huggingface.co/datasets/allenai/c4), data is stored in smaller shards.  

Download the data by, 

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include "en/*"   # for C4 data
for lang in "ar" "sw" "zh" "zh-Latn" "ca" "fr" "hi" "ur" "bn" "id" "pt" "es" "ru" "ru-Latn" "ja" "am"; do
    git lfs pull --include "multilingual/c4-nl.*.json.gz" # for mC4 data
done
``` 

What I did is to process smaller shards with multi-process and sample data based on the sampling probability from each of the shards by the following script. 

```
bash scripts/c4_mc4_processing/extract_from_git_lfs.sh
```

### Preprocess data

Preprocess data using,
```
bash scripts/c4_mc4_processing/multi_shard_data_process.sh
```

Note that `multi_shard_data_process.sh` also merge the shards into single `*.bin` and `*.idx` files.

## Calculate iterator probability for each of the datasets.

```
bash scripts/c4_mc4_processing/calc_iterator_prob.sh
```

The scripts expect all the language processed (`*.bin`, and `*.idx`) to be in a same folder in the following format, 

```
data_folder_path/
├── lang_1 
│   ├── train_text_document.bin
│   ├── train_text_document.idx
│   ├── validation_text_document.bin
│   └── validation_text_document.idx
├── lang_2
│   ├── train_text_document.bin
│   ├── train_text_document.idx
│   ├── validation_text_document.bin
│   └── validation_text_document.idx
...
...
├── lang_n
│   ├── train_text_document_1.bin
│   ├── train_text_document_1.idx
│   ├── train_text_document_2.bin
│   ├── train_text_document_2.idx
│   ├── train_text_document_3.bin
│   ├── train_text_document_3.idx
│   ├── validation.bin
│   └── validation.idx
```


# Data Binarization Stat 

If you tokenize English with `t5` tokenizer, `784GB` raw data becomes `344GB` (`*.bin` size, including validation). But if you tokenize the same `784GB` raw data with `mt5` it becomes `756GB` This is not what we were expecting at first. Earlier we were expecting 50\% English and 50\% remaining language, but now after binarization (a.k.a. tokenization) that calculation doesn't hold. For most of the languages, after binarization, the size reduced drastically. The stats are below, 

|Language|Raw Size|Binary Size|
|--|--|--|
|am|2.4GB|1.3GB|
|ar|86GB| 28GB |
|bn|23GB|7.6GB|
|ca|14GB|15GB|
|en|784GB|763GB|
|es|110GB|111GB|
|fr|120GB|112GB|
|hi|23GB|8.7GB|
|id|33GB|34GB|
|ja|148GB|64GB|
|pt|59GB|60GB|
|ru|339GB|69GB|
|ru-Latn|3.5GB|3.2GB|
|sw|3.3GB|3.8GB|
|ur|28GB|9.2GB|
|zh|43GB|22GB|
|zh-Latn|738MB|779MB|

All the sizes are `*.bin` + `*.idx` file size.