for LANG in "am" "hi" "ur" "bn" "id" "ca" "zh-Latn" "sw" "ru-Latn" "zh"; do
# for LANG in "ar" "ja"; do
# for LANG in "pt"; do
    python tools/preprocess_data.py \
        --input dumped/mc4_json_data/$LANG/train.jsonl \
        --output-prefix dumped/mc4_json_data/$LANG/train \
        --dataset-impl mmap \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path "google/mt5-large" \
        --workers 92
    
    python tools/preprocess_data.py \
        --input dumped/mc4_json_data/$LANG/validation.jsonl \
        --output-prefix dumped/mc4_json_data/$LANG/validation \
        --dataset-impl mmap \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path "google/mt5-large" \
        --workers 92
done