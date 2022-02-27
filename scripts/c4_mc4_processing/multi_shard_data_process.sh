#
# Process multiple shard data
#

lang='en'
for _file in dumped/raw_sampled/$lang/train.* ; do
    PREFIX=${_file%.*}
    python tools/preprocess_data.py \
        --input $_file \
        --output-prefix $PREFIX \
        --dataset-impl mmap \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path "t5-base" \
        --workers 92
done

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


# merge processed binaries

FILES=''
for _file in dumped/raw_sampled/en/train_shards/train_shard_1/train.*.bin; do
    _file_base_name=${_file%.*}
    FILES="$FILES $_file_base_name"
done
echo $FILES

python3 tools/merge_preprocessed_data.py \
  --datasets $FILES \
  --output-prefix dumped/raw_sampled/en/train_shards/train_shard_1/train



FILES=''
for _file in dumped/mc4_json_data/$lang/validation.*.bin; do
    _file_base_name=${_file%.*}
    FILES="$FILES $_file_base_name"
done
echo $FILES

python3 tools/merge_preprocessed_data.py \
  --datasets $FILES \
  --output-prefix dumped/mc4_json_data/$lang/validation

  


