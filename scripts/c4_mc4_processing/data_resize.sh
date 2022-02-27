CACHE_DIR="../c4/mc4_splits/"
DATASET_NAME="mc4"
NEW_EXPECTED_SIZE=576
OUTPUT_DIR="dumped/c4_mc4_raw_data_resized-"$NEW_EXPECTED_SIZE

mkdir -p $OUTPUT_DIR
ALPHA=.01
MIN_HIGH_RESOURCE_SIZE=12
MAX_HIGH_RESOURCE_SIZE=100

python3 -u tools/c4_mc4/data_resize.py \
  --dataset-name $DATASET_NAME \
  --size-format "GB" \
  --languages "ar" "sw" "zh" "zh-Latn" "ca" "fr" "hi" "ur" "bn" "id" "pt" "es" "ru" "ru-Latn" "ja" "am"  \
  --cache-dir $CACHE_DIR \
  --new-expected-size $NEW_EXPECTED_SIZE \
  --output-dir $OUTPUT_DIR \
  --min_high_resource_size $MIN_HIGH_RESOURCE_SIZE \
  --max_high_resource_size $MAX_HIGH_RESOURCE_SIZE \
  --alpha $ALPHA
