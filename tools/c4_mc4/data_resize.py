import os
import math
import copy
import json
import datasets
import argparse
import subprocess
from collections import OrderedDict

def get_size_stats(args):
    lang_size_dict, tot_size = {}, 0
    for lang in args.languages:
        lang_folder_path = os.path.join(
            os.path.join(args.cache_dir, args.dataset_name),
            lang
        )
        lang_size = subprocess.check_output("du -s {}".format(lang_folder_path), shell=True)
        lang_size = int(lang_size.decode("utf-8").split("\t")[0])
        if args.size_format == 'B':
            _conv = 1
        elif args.size_format == 'MB':
            _conv = 1024
        elif args.size_format == 'GB':
            _conv = 1024*1024
        elif args.size_format == 'TB':
            _conv = 1024*1024*1024
        lang_size_gb = round(lang_size/float(_conv), 2)
        tot_size += lang_size_gb
        lang_size_dict[lang] = lang_size_gb
    return lang_size_dict

def print_stat(args, lang_size_dict):
    lang_list = sorted([(k,v) for k, v in lang_size_dict.items()], key=lambda tup: tup[1])
    total_size = 0
    print("Language : Size ")
    print("-"*20)
    for lang, size in lang_list:
        print("{} :   {}".format(lang, size))
        total_size += size
    print("-"*20)
    print("Total size : {}".format(total_size))
    print("Expected size afted resizing : {}".format(args.new_expected_size))
    print("Per language allocated size : {}".format(args.new_expected_size/len(args.languages)))

def find_and_distribute_low_resoure_language(args, lang_size_dict, sampling_weight):
    total_size = sum([v for k, v in lang_size_dict.items()])
    mean_size_for_each_lang = args.new_expected_size/len(args.languages)
    tot_low_resource_lang_size = 0
    print("Low resource languages :", end="")
    for lang, size in lang_size_dict.items():
        if size < mean_size_for_each_lang:
            sampling_weight[lang] = 1.0
            tot_low_resource_lang_size += size
            print(" {}({})".format(lang, size), end="")
    print("")
    print("Total size consumed by low resource languages {}".format(tot_low_resource_lang_size))
    return tot_low_resource_lang_size

def calc_multinomial_sampling_prob_with_penalty(dataset_size, alpha=.5):
    tot_size = 0
    probs = OrderedDict()
    for lang, size in dataset_size.items():
        tot_size += size
    for lang, size in dataset_size.items():
        probs[lang] = size/tot_size

    pen_prob = OrderedDict()
    tot_pen_prob = 0.0
    for lang, prob in probs.items():
        tot_pen_prob += (prob**alpha)
    sum_ = 0.0
    for lang, prob in probs.items():
        pen_prob[lang] =  (prob**alpha)/tot_pen_prob
        sum_ += pen_prob[lang]
    assert math.fabs(1-sum_) < 1e-6
    return pen_prob

def distribute_high_resoure_language(args, lang_dict, sampling_probability, total_size_capacity):
    lang_size_dict = copy.deepcopy(lang_dict)
    total_high_resource_capacity = total_size_capacity
    for lang, prob in sampling_probability.items():
        if prob == 1.0:
            del lang_size_dict[lang] 
    high_resource_sampling_prob = calc_multinomial_sampling_prob_with_penalty(lang_size_dict, alpha=args.alpha)
    print("Sampling High resource language based on multinomial distribution with alpha {}".format(args.alpha))
    print("-"*80)
    total_high_resource_lang_size = 0
    lang_fixed, high_resource_size = {}, {}
    for lang, prob in high_resource_sampling_prob.items():
        new_prob = prob
        new_prob_str = ""
        new_size = lang_size_dict[lang] * new_prob
        if new_size < args.min_high_resource_size:
            lang_fixed[lang] = True
            new_size = args.min_high_resource_size
            new_size = min(lang_size_dict[lang], new_size)
            new_prob = new_size/lang_size_dict[lang]
            new_prob_str="-> {}".format(round(new_prob, 2))
        if new_size > args.max_high_resource_size:
            new_size = args.max_high_resource_size
            new_prob = new_size/lang_size_dict[lang]
            new_prob_str="-> {}".format(round(new_prob, 2))
        high_resource_sampling_prob[lang] = new_prob
        high_resource_size[lang] = new_size
        sampling_probability[lang] = prob
        print("Language : {}, Sampling prob : {} {}, ({} -> {} GB)".format(
            lang, round(prob,2), new_prob_str, lang_size_dict[lang], round(new_size) )
        )
        total_size_capacity -= new_size
        total_high_resource_lang_size += new_size
    print("Expected high resource size {}, Total Size : {}".format(total_high_resource_capacity, total_high_resource_lang_size))
    adjustment = total_size_capacity
    if adjustment > 0:
        print("Performing adjustment ...")
        for lang, size in high_resource_size.items():
            if size ==  args.max_high_resource_size:
                lang_fixed[lang] = True
        _flag = True
        while adjustment > 0 and _flag:
            _flag = False
            for lang, size in high_resource_size.items():
                if lang not in lang_fixed and adjustment > 0:
                    if size < lang_size_dict[lang]:
                        _dist_val = min(1, lang_size_dict[lang]-size)
                        _dist_val = min(_dist_val, adjustment)
                        high_resource_size[lang] += _dist_val
                        adjustment -= _dist_val
                        _flag = True
        for lang, size in high_resource_size.items():
            _sampling_prob = high_resource_size[lang]/lang_size_dict[lang]
            sampling_probability[lang] = _sampling_prob
    return sampling_probability

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True,
                        help='Name of the dataset.',
                        choices=['c4', 'mc4'])
    parser.add_argument('--languages', nargs='+', required=True,
                        help='Name of the langugae.')
    parser.add_argument('--cache-dir', type=str, required=True,
                        help='Path to the cache dir. (The directory may require very large space)')
    parser.add_argument('--size-format', type=str, required=True,
                        help='Calculation will be done in byte, mega-byte, giga-byte or tera-byte',
                        choices=['B', 'MB', 'GB', 'TB'])
    parser.add_argument('--new-expected-size', type=int, required=True,
                        help='Total amount of data to be selected.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory where data will be saved.')
    parser.add_argument('--alpha', type=float, required=True,
                        help='Sampling penalty.')
    parser.add_argument('--min_high_resource_size', type=int, required=True,
                        help='Sampling penalty.')
    parser.add_argument('--max_high_resource_size', type=int, required=True,
                        help='Sampling penalty.')
    args = parser.parse_args()
    
    total_size_capacity = args.new_expected_size

    lang_size_dict = get_size_stats(args)
    print_stat(args, lang_size_dict)

    sampling_probability = {lang: -1 for lang in args.languages}
    low_resource_size_consumed = find_and_distribute_low_resoure_language(args, lang_size_dict, sampling_probability)
    total_size_capacity = total_size_capacity - low_resource_size_consumed
    distribute_high_resoure_language(args, lang_size_dict, sampling_probability, total_size_capacity)
    
    total_size = 0
    print("\nFinal Breakdown")
    print("-"*15)
    for lang, prob in sampling_probability.items():
        _size = lang_size_dict[lang]*prob
        print("Language : {}, Sampling prob : {}, ({} -> {} GB)".format(
            lang, round(prob,2), lang_size_dict[lang], round(_size, 2) )
        )
        total_size += _size
    print("Expected resource size {}, Total Size : {}".format(args.new_expected_size, round(total_size,1)))
    open(os.path.join(args.output_dir, 'lang_dict.json'), "w").write(
        json.dumps(sampling_probability, indent=4)
    )


if __name__ == '__main__':
    main()
