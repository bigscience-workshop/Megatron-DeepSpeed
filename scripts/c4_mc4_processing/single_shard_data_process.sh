
# Process single shard data
# for lang in "am" "hi" "ur" "bn" "id" "ca" "zh-Latn" "sw" "ru-Latn" "zh" "ar" "ja" "pt" "es" "fr"; do

for lang in "fr"; do
    python tools/preprocess_data.py \
        --input dumped/raw_sampled/$lang/train.jsonl \
        --output-prefix dumped/raw_sampled/$lang/train \
        --dataset-impl mmap \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path "google/mt5-base" \
        --workers 92
    
    # python tools/preprocess_data.py \
    #     --input dumped/raw_sampled/$lang/validation.jsonl \
    #     --output-prefix dumped/raw_sampled/$lang/validation \
    #     --dataset-impl mmap \
    #     --tokenizer-type PretrainedFromHF \
    #     --tokenizer-name-or-path "google/mt5-base" \
    #     --workers 92
done
