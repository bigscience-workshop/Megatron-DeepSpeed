
for FILEPATH in dumped/raw_sampled/*; do
    for _file in $FILEPATH/*; do
        echo "running /bin/gzip $_file"    
        /bin/gzip $_file
    done
done
