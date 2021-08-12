#!/bin/bash

datadir=`pwd`/oscar/download
mkdir -p $datadir
pushd $datadir
  for part in {1..670} ; do
    filename="en_part_${part}.txt.gz"
    url="https://s3.amazonaws.com/datasets.huggingface.co/oscar/1.0/unshuffled/deduplicated/en/${filename}"
    echo $url
    wget $url
    if [ $? -ne 0 ] ; then
      echo "Failed to download: $filename"
      rm -f $filename
      break
    fi
  done
popd
