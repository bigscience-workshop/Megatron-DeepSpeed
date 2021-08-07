
# Process single shard data
# for lang in "am" "hi" "ur" "bn" "id" "ca" "zh-Latn" "sw" "ru-Latn" "zh" "ar" "ja" "pt" "es" "fr"; do

for lang in "am"; do
    python tools/preprocess_data.py \
        --input dumped/mc4_json_data/$lang/train.jsonl \
        --output-prefix dumped/mc4_json_data/$lang/train \
        --dataset-impl mmap \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path "google/mt5-base" \
        --workers 92
    
    python tools/preprocess_data.py \
        --input dumped/mc4_json_data/$lang/validation.jsonl \
        --output-prefix dumped/mc4_json_data/$lang/validation \
        --dataset-impl mmap \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path "google/mt5-base" \
        --workers 92
done



# Process multiple shard data
lang='en'
cnt=0
for _file in dumped/mc4_json_data/$lang/validation* ; do
    PREFIX=${_file%.*}
    python tools/preprocess_data.py \
        --input $_file \
        --output-prefix $PREFIX \
        --dataset-impl mmap \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path "google/mt5-base" \
        --workers 92
done


python3 tools/merge_preprocessed_data.py \
  --datasets dumped/mc4_json_data/en/validation.*.bin \
  --output-prefix dumped/mc4_json_data/en/validation


