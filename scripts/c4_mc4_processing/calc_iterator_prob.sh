# for alpha in .1 .2 .3 .4 .5 .6 .7 .8 .9; do
for alpha in .1; do
    python tools/c4_mc4/calc_iterator_prob.py \
    --data-folder-path dumped/mc4_json_data/ \
    --size-format GB \
    --alpha $alpha \
    --output-dir dumped/mc4_json_data/ \
    --name-prefix 'train' \
    --extension-name 'bin'
done
