LANG_SAMPLE_DICT_PATH="dumped/c4_mc4_raw_data_resized-576/lang_dict.json"
NUM_PROC=64

# In original repo mc4 and c4 data is in different folder. 
# We took english data from c4 dataset which is much clean.
# We took remaining multi-lingual data from mc4.


# mc4 data extract
for lang in "ru" ; do 
    CACHE_DIR="dumped/en_ru/c4/multilingual/c4-$lang.\*.json.gz"
    OUTPUT_DIR="dumped/mc4_json_data/"
    mkdir -p $OUTPUT_DIR
    OUTPUT_LANG_DIR=$OUTPUT_DIR$lang
    mkdir -p $OUTPUT_LANG_DIR
    python3 -u tools/c4_mc4/extract_from_git_lfs.py \
    --language $lang \
    --dataset-name "train.jsonl" \
    --lang-sampling-dict-path $LANG_SAMPLE_DICT_PATH \
    --cache-dir $CACHE_DIR \
    --output-dir $OUTPUT_LANG_DIR \
    --num-proc $NUM_PROC
    
    CACHE_DIR="dumped/en_ru/c4/multilingual/c4-$lang-validation.\*.json.gz"
    OUTPUT_DIR="dumped/mc4_json_data/"
    mkdir -p $OUTPUT_DIR
    OUTPUT_LANG_DIR=$OUTPUT_DIR$lang
    mkdir -p $OUTPUT_LANG_DIR
    python3 -u tools/c4_mc4/extract_from_git_lfs.py \
    --language $lang \
    --dataset-name "validation.jsonl" \
    --sampling-ratio .1 \
    --cache-dir $CACHE_DIR \
    --output-dir $OUTPUT_LANG_DIR \
    --num-proc $NUM_PROC
done

# C4 extract

lang="en"

CACHE_DIR="dumped/en_ru/c4/$lang/c4-train.\*.json.gz"
OUTPUT_DIR="dumped/mc4_json_data/"
mkdir -p $OUTPUT_DIR
OUTPUT_LANG_DIR=$OUTPUT_DIR$lang
mkdir -p $OUTPUT_LANG_DIR
python3 -u tools/c4_mc4/extract_from_git_lfs.py \
--language $lang \
--dataset-name "train.jsonl" \
--sampling-ratio 1 \
--cache-dir $CACHE_DIR \
--output-dir $OUTPUT_LANG_DIR \
--num-proc $NUM_PROC


CACHE_DIR="dumped/en_ru/c4/$lang/c4-validation.\*"
OUTPUT_DIR="dumped/mc4_json_data/"
mkdir -p $OUTPUT_DIR
OUTPUT_LANG_DIR=$OUTPUT_DIR$lang
mkdir -p $OUTPUT_LANG_DIR
python3 -u tools/c4_mc4/extract_from_git_lfs.py \
--language $lang \
--dataset-name "validation.jsonl" \
--sampling-ratio 1 \
--cache-dir $CACHE_DIR \
--output-dir $OUTPUT_LANG_DIR \
--num-proc $NUM_PROC

