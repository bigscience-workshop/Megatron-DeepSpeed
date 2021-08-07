# Note: Running this script may require bandwidth ~30MB/s per language.
#       You may run this script mutiple times to make the caching faster.
#       The script `tools/c4_mc4/c4_mc4_cache.py` will perform caching, 
#           if a caching folder for a language doesn't exists. So running 
#           the script multiple times with same cache folder will be ok.

CACHE_DIR="dumped/c4_mc4_raw_data"
mkdir -p $CACHE_DIR

# excluding en since it's already been processed. Please add your language here.
# for LANG in "ar" "sw" "zh" "zh-Latn" "ca" "fr" "hi" "ur" "bn" "id" "pt" "es" "ru" "ru-Latn" "ja" "am"; do
for LANG in "am" ; do
    DATASET_NAME="mc4"
    if [[ $LANG == "en" ]] 
    then
        DATASET_NAME="c4"
    fi
    echo "Caching "$LANG
    sleep $((1 + RANDOM % 2))
    python3 -u tools/c4_mc4/c4_mc4_cache.py \
      --dataset-name $DATASET_NAME \
      --lang $LANG \
      --cache-dir $CACHE_DIR
done