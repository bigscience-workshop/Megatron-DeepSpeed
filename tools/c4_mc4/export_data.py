import os
import time
import math
import copy
import json
import tqdm
import datasets
import argparse
import subprocess
from collections import OrderedDict

def export_jsonl_data(dataset, path, dataset_name, chunk_size):
    filePtr = open(os.path.join(path, dataset_name),"w")
    wrt_lst = []
    for dt in tqdm.tqdm(dataset, desc='{}'.format(dataset_name)):
        filePtr.write(json.dumps(dt)+"\n")
    filePtr.close()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True,
                        help="Name of the dataset.",
                        choices=['c4', 'mc4'])
    parser.add_argument('--language', type=str, required=True,
                        help="Name of the langugae.")
    parser.add_argument('--lang-sampling-dict-path', type=str, default=None,
                        help="Path of the json where the sampling ratio is mentioned. ('key' -> language-name, 'value'-> sampling ratio)")
    parser.add_argument('--sampling-ratio', type=float, default=None,
                        help="Sampling ratio for --language")
    parser.add_argument('--cache-dir', type=str, required=True,
                        help='Path to the cache dir. (The directory may require very large space if it\'s not cached earlier.)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path where the jsonl data will be exported.')
    parser.add_argument('--write-chunk', type=int, default=100000,
                        help='Number of samples written at a time.')
    args = parser.parse_args()
    
    if args.sampling_ratio is not None and args.lang_sampling_dict_path is not None:
        raise ValueError("Both --lang-sampling-dict-path and --sampling-ratio is used. Please use either one of them.")
    if args.sampling_ratio is not None:
        sampling_ratio = args.sampling_ratio
    elif args.lang_sampling_dict_path is not None:
        sampling_dict = json.load(open(args.lang_sampling_dict_path, "r"))
        sampling_ratio = sampling_dict[args.language]
    else:
        raise ValueError("Both --lang-sampling-dict-path and --sampling-ratio is None. Please use any one of them.")
    
    start_time = time.time()
    dataset = datasets.load_dataset(args.dataset_name, args.language, cache_dir=args.cache_dir)
    load_time = time.time() - start_time
    print("{} dataset load time : {}".format(args.language, time.strftime('%H:%M:%S', time.gmtime(load_time))))
    print("{} dataset size : {}".format(args.language, dataset))
    start_time = time.time()


    sampling_ratio = 0.99999 if sampling_ratio == 1.0 else sampling_ratio
    train_dataset = dataset['train']
    sampled_train_dataset = train_dataset.train_test_split(train_size=float(sampling_ratio), shuffle=True)
    shuffle_time = time.time() - start_time
    print("{} dataset shuffle and selection time : {}".format(args.language, time.strftime('%H:%M:%S', time.gmtime(shuffle_time))))
    print("{} dataset size after sampling: {}".format(args.language, len(dataset)))
    print("{} sampled dataset size : {}".format(args.language, sampled_train_dataset))

    export_jsonl_data(sampled_train_dataset['train'], args.output_dir, 'train.jsonl', args.write_chunk)
    export_jsonl_data(dataset['validation'], args.output_dir, 'validation.jsonl', args.write_chunk)


if __name__ == '__main__':
    main()
