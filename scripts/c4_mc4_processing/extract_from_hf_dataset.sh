DATASET_NAME="mc4"
LANG_SAMPLE_DICT_PATH="dumped/c4_mc4_raw_data_resized-576/lang_dict.json"
CACHE_DIR="c4/mc4_splits/"
OUTPUT_DIR="dumped/mc4_json_data/"

mkdir -p $OUTPUT_DIR

# completed am, hi, ur, bn, id, ca, zh-Latn, sw, ru-Latn, zh

# for LANG in "am" "hi" "ur" "bn" "id" "ca" "zh-Latn" "sw" "ru-Latn" "zh"; do # 1
# for LANG in "ar" ; do # 0
# for LANG in "fr" ; do # 2
# for LANG in "pt" ; do # 3
# for LANG in "es" ; do # 4
for LANG in "ru" ; do # 5
# for LANG in "ja" ; do # 6
    OUTPUT_LANG_DIR=$OUTPUT_DIR$LANG
    if [ ! -d "$OUTPUT_LANG_DIR" ] 
    then
        mkdir -p $OUTPUT_LANG_DIR
        python3 -u tools/c4_mc4/extract_data_from_hf_dataset.py \
        --dataset-name $DATASET_NAME \
        --language $LANG \
        --lang-sampling-dict-path $LANG_SAMPLE_DICT_PATH \
        --cache-dir $CACHE_DIR \
        --output-dir $OUTPUT_LANG_DIR
    else
        echo "[*] Directory $OUTPUT_LANG_DIR already exists."
        echo "[*] Aborting operation"
    fi
done

