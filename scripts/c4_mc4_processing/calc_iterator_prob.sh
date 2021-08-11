IT_SELECTION_PRB_FILE_PATH=dumped/mc4_processed_data/sample_iterator_probs
mkdir -p $IT_SELECTION_PRB_FILE_PATH
for alpha in .1 .2 .3 .4 .5 .6 .7 .8 .9; do
    python tools/c4_mc4/calc_iterator_prob.py \
    --data-folder-path dumped/mc4_processed_data/ \
    --size-format G \
    --alpha $alpha \
    --output-dir $IT_SELECTION_PRB_FILE_PATH \
    --name-prefix 'train' \
    --extension-name 'bin'
done
