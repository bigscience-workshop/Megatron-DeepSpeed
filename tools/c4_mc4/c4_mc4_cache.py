import os
import datasets
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True,
                        help='Name of the dataset.',
                        choices=['c4', 'mc4'])
    parser.add_argument('--lang', type=str, required=True,
                        help='Name of the langugae.')
    parser.add_argument('--cache-dir', type=str, required=True,
                        help='Path to the cache dir. (The directory may require very large space)')
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    lang_cache_log = os.path.join(args.cache_dir, args.lang+".log")
    if not os.path.exists(lang_cache_log):
        open(lang_cache_log, 'w').write("Data downloading and processing.\n")
        try:
            print("downloading {}".format(args.lang))
            dataset_name="mc4"
            if args.lang == "en":
                dataset_name="c4"
            print('Running \"mc4_dataset = datasets.load_dataset({}, {}, cache_dir={})\"'.format(
                args.dataset_name, args.lang, args.cache_dir
            ))
            mc4_dataset = datasets.load_dataset(args.dataset_name, args.lang, cache_dir=args.cache_dir)
        except:
            raise Exception("Download failed for {} lang".format(args.lang))
            open(lang_cache_log, 'a').write("Data caching failed.\n")
        open(lang_cache_log, 'a').write("Data caching for {} language completed.\n".format(args.lang))
    else:
        print("Data processing ofr {} language started or completed.".format(args.lang))
        

if __name__ == '__main__':
    main()