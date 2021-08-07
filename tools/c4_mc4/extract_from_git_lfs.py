import os
import time
import glob
import json
import gzip
import tqdm
import argparse
import subprocess
import numpy as np
import concurrent.futures

def export_jsonl_data(source_file_path, path, dataset_name, sampling_ratio):
    print(source_file_path, path, dataset_name, sampling_ratio)
    serial = source_file_path.split("-of-")[0].split(".")[-1] + "-of-" + source_file_path.split("-of-")[-1].split(".")[0] 
    tot_sample = 0
    with gzip.open(source_file_path, 'rt', encoding='utf-8') as gzFilePtr:
        for _ in gzFilePtr:
            tot_sample += 1
    indices = list(range(tot_sample))
    np.random.shuffle(indices)
    num_selected_samples = int(sampling_ratio*tot_sample)
    indices = set(indices[0:num_selected_samples])
    with gzip.open(source_file_path, 'rt', encoding='utf-8') as gzFilePtr:
        new_dataset_name = dataset_name.replace(".jsonl", ".{}.jsonl".format(serial))
        filePtr = open(os.path.join(path, new_dataset_name),"w")
        for idx, dt in tqdm.tqdm(enumerate(gzFilePtr), desc='{}'.format(source_file_path)):
            if idx in indices:
                filePtr.write(str(dt).strip()+"\n")
        filePtr.close()
    return 0
    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, required=True,
                        help="Name of the langugae.")
    parser.add_argument('--dataset-name', type=str, default="train.jsonl",
                        help="Name of the langugae.")
    parser.add_argument('--lang-sampling-dict-path', type=str, default=None,
                        help="Path of the json where the sampling ratio is mentioned. ('key' -> language-name, 'value'-> sampling ratio)")
    parser.add_argument('--sampling-ratio', type=float, default=None,
                        help="Sampling ratio for --language")
    parser.add_argument('--cache-dir', type=str, required=True,
                        help='Path to the cache dir. (The directory may require very large space if it\'s not cached earlier.)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path where the jsonl data will be exported.')
    parser.add_argument('--num-proc', type=int, default=64,
                        help='Number of files to be processed in parallel.')
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
    
    np.random.seed(1234)
    files = sorted(glob.glob(args.cache_dir.replace("\\","")))
    # for _file in files:
    #     export_jsonl_data(_file, args.output_dir, args.dataset_name, sampling_ratio)
    
    # TODO: log any fail
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_proc) as executor:
        results = executor.map(
                    export_jsonl_data, 
                    files, 
                    [args.output_dir for _ in files], 
                    [args.dataset_name for _ in files], 
                    [sampling_ratio for _ in files]
                )


if __name__ == '__main__':
    main()

