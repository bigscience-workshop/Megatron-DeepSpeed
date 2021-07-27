# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time

import torch
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)

        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()
    
    # this depends on a queue in the main process to read the lines and write to disk
    def encode_yield(self, job):
      with  open(self.args.input, 'r', encoding='utf-8') as fin:
        fin.seek(job[0])
        all_jsonl = fin.readlines(job[1])
      while all_jsonl:
        json_line all_jsonl[0]
        all_jsonl = all_jsonl[1:]
        data = json.loads(json_line)
        ids = {}
        for key in self.args.json_keys:
            text = data[key]
            doc_ids = []
            for sentence in Encoder.splitter.tokenize(text):
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
        yield ids, len(json_line)

    # this does writing to shard files, and then merge on disk
    def encode_shard(self, job):
      with  open(self.args.input, 'r', encoding='utf-8') as fin:
        fin.seek(job[0])
        all_jsonl = fin.readlines(job[1])
      level = "document"
      if self.args.split_sentences:
          level = "sentence"
      output_bin_files = {}
      output_idx_files = {}
      builders = {}
      for key in self.args.json_keys:
          output_bin_files[key] = "{}_{}_{}_shard_{}.bin".format(self.args.output_prefix,
                                                        key, level, job[0])
          output_idx_files[key] = "{}_{}_{}_shard_{}.idx".format(self.args.output_prefix,
                                                        key, level, job[0])
          builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                impl=self.args.dataset_impl,
                                                vocab_size=Encoder.tokenizer.vocab_size)
      while all_jsonl:
        json_line all_jsonl[0]
        all_jsonl = all_jsonl[1:]
        data = json.loads(json_line)
        ids = {}
        for key in self.args.json_keys:
            text = data[key]
            doc_ids = []
            for sentence in Encoder.splitter.tokenize(text):
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
        doc, bytes_processed = ids, len(json_line)
        if doc:
          for key, sentences in doc.items():
              if len(sentences) == 0:
                  continue
              for sentence in sentences:
                  builders[key].add_item(torch.IntTensor(sentence))
              builders[key].end_document()
      for key in self.args.json_keys:
          builders[key].finalize(output_idx_files[key])

      return [output_bin_files[key].replace(".bin", "") for key in self.args.json_keys], job[1]

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, default=None,
                       help='Path to input JSON')
    group.add_argument('--shard_and_merge_input', type=bool, default=False,
                       help='Shard the input and process in jobs, and then merge')
    group.add_argument('--shard_size', type=int, default=100,
                       help='Size of shards in MBs')
    group.add_argument('--datasets', nargs='+', default=None,
                       help='Paths to one or more input datasets to merge')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, default=None, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'PretrainedFromHF'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                       help="Name or path of the huggingface tokenizer.")

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    if args.input is None and args.datasets is None:
      raise RuntimeError("Either an input file or one or more datasets to merge must be provided.")
    if args.input is not None and args.datasets is not None:
      raise RuntimeError("Both input and datasets are set. Either an input file is processed or a list of datasets are merged, but not both.")
    if args.datasets is not None and len(args.json_keys) > 1:
      raise RuntimeError("Merging datasets are performed only for one key at a time.")

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

       
def get_file_shard_ranges(input_file_path, shard_size, num_proc, num_shards=None):
        file_size= os.stat(input_file_path).st_size        
        if num_shards is not None:
            shard_size = int(file_size/num_shards) 
        shard_size = min(shard_size, int(file_size/num_proc))
        with open(input_file_path, "rb") as f:
          file_segs = []
          file_pos = 0
          while file_pos < file_size:
                if file_size - file_pos <= shard_size:
                    file_segs.append((file_pos, file_size))
                    break
                f.seek(file_pos+shard_size, 0)
                seg_len = shard_size
                line = f.readline()
                if not line:
                    file_segs.append((file_pos, file_size))
                    break
                seg_len += len(line)
                if file_size-(file_pos+seg_len) < shard_size:
                    file_segs.append((file_pos, file_size))
                    break

                file_segs.append((file_pos, file_pos + seg_len))
                file_pos = f.tell()
          line = None
          return file_segs

# merge files
def merge_by_key(key, args, vocab_size):
  level = "document"
  if args.split_sentences:
      level = "sentence"
  datasets = [dataset for dataset in args.datasets if dataset.startswith(f"{args.output_prefix}_{key}_{level}")]
  if len(datasets) == 0: return
  output_bin_file = f"{args.output_prefix}_{key}_{level}.bin"
  output_idx_file = f"{args.output_prefix}_{key}_{level}.idx"
  builder = indexed_dataset.make_builder(output_bin_file,
                                                  impl=args.dataset_impl,
                                                  vocab_size=vocab_size)
  for dataset in datastes:
    builder.merge_file_(dataset)      
  builder.finalize(output_idx_file)
  for dataset in datasets:
    if dataset != f"{args.output_prefix}_{key}_{level}":
      os.unlink(f"{dataset}.bin")
      os.unlink(f"{dataset}.idx")

def main():
    args = get_args()
    startup_start = time.time()
    if args.input is not None:
      print("Opening", args.input)
      if nltk_available and args.split_sentences:
          nltk.download("punkt", quiet=True)
      shards = get_file_shard_ranges(args.input, args.shard_size*1000000,  num_proc=args.workers) 
      encoder = Encoder(args)
      vocab_size = build_tokenizer(args).vocab_size
      pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
      if not args.shard_and_merge_input:
        encoded_docs = pool.map_async(encoder.encode_yield, shards, chunksize=1)
        level = "document"
        if args.split_sentences:
            level = "sentence"

        print(f"Vocab size: {tokenizer.vocab_size}")
        print(f"Output prefix: {args.output_prefix}")
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        for key in args.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                          key, level)
            builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                  impl=args.dataset_impl,
                                                  vocab_size=vocab_size)

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        for i, (doc, bytes_processed) in enumerate(encoded_docs.get(), start=1):
            total_bytes_processed += bytes_processed
            for key, sentences in doc.items():
                if len(sentences) == 0:
                    continue
                for sentence in sentences:
                    builders[key].add_item(torch.IntTensor(sentence))
                builders[key].end_document()
            if i % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed/elapsed/1024/1024
                print(f"Processed {i} documents",
                      f"({i/elapsed} docs/s, {mbs} MB/s).",
                      file=sys.stderr)
        for key in args.json_keys:
            builders[key].finalize(output_idx_files[key])
        pool.close()
        pool.join()
      else:
        encoded_docs = pool.map_async(encoder.encode_shard, shards, chunksize=1)
        builders = None
        datasets =[]
        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        for i, (shard_files, bytes_processed) in enumerate(encoded_docs.get(), start=1):
            args.datasets.extend(shard_files)  
            total_bytes_processed += bytes_processed
            if i % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed/elapsed/1024/1024
                print(f"Processed {i} documents",
                      f"({i/elapsed} docs/s, {mbs} MB/s).",
                      file=sys.stderr)
        pool.close()
        pool.join()
        # we could have done the merge incrementally but the dataset builder doesn't really do incremental merge efficiently
        # now merge all the shards. 
        startup_start = time.time()
        curr_keys = list(set([a.split("_")[1] for a in args.datasets]))
        pool = multiprocessing.Pool(len(curr_keys))
        pool.starmap_async(merge_by_key, curr_keys, [args]* len(curr_keys), [vocab_size]* len(curr_keys)).get()
        pool.close()
        pool.join()   
        startup_end = time.time()
        print("Time to merge:", startup_end - startup_start)
        print(f"Merged {len(args.datasets)} datasets to {args.output_prefix}")

    elif args.datasets is not None:
      print("Merging",args.datasets)
      vocab_size = build_tokenizer(args).vocab_size
      print(f"Vocab size: {tokenizer.vocab_size}")
      print(f"Output prefix: {args.output_prefix}")
      curr_keys = list(set([a.split("_")[1] for a in args.datasets]))
      pool = multiprocessing.Pool(len(curr_keys))
      pool.starmap_async(merge_by_key, curr_keys, [args]* len(curr_keys), [vocab_size]* len(curr_keys)).get()
      pool.close()
      pool.join()   
      startup_end = time.time()
      print("Time to merge:", startup_end - startup_start)
      print(f"Merged {len(args.datasets)} datasets to {args.output_prefix}")


if __name__ == '__main__':
  #import cProfile
  #cProfile.run('main()',  'restats')
  main()
