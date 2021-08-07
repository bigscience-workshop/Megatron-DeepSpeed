PATH=dumped/mc4_json_data
NEW_PATH=dumped/raw_sampled
# for LANG in "hi" "ur" "bn" "id" "ca" "zh-Latn" "sw" "ru-Latn" "zh"; do
# for LANG in "ja" "pt"; do
# for LANG in "ar"; do
for LANG in "es" "fr"; do
    NEW_FILE_PATH=$NEW_PATH/$LANG/
    /bin/mkdir -p $NEW_FILE_PATH

    FILE_PATH=$PATH/$LANG/train.jsonl
    echo "running /bin/gzip $FILE_PATH"
    /bin/gzip $FILE_PATH
    /bin/mv $FILE_PATH".gz" $NEW_FILE_PATH

    FILE_PATH=$PATH/$LANG/validation.jsonl
    echo "running /bin/gzip $FILE_PATH"
    /bin/gzip $FILE_PATH
    /bin/mv $FILE_PATH".gz" $NEW_FILE_PATH
done
