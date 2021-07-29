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
import collections
import itertools
import json
import multiprocessing
import os
import sys
import threading

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
import threading
from multiprocessing.connection import Connection

from megatron.data.indexed_dataset import index_file_path, data_file_path

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


    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        for key in self.args.json_keys:
            text = data[key]
            tokenized_sentences = Encoder.splitter.tokenize(text)
            if self.args.old_gpt_tokenize:
              doc_ids = [
                  sentence_ids
                  for sentence_ids in Encoder.tokenizer.batch_tokenize_old(tokenized_sentences)
                  if len(sentence_ids) > 0
              ]
            else:
              doc_ids = [
                  sentence_ids
                  for sentence_ids in Encoder.tokenizer.batch_tokenize(tokenized_sentences)
                  if len(sentence_ids) > 0
              ]
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
        return ids, len(json_line)

    # this depends on a queue in the main process to read the lines and write to disk
    # if _iter is true, read line by line, instead in blocks
    def encode_batch(self, job):
      def process_line(self, json_line):
          try:
            data = json.loads(json_line)
          except:
            print ('json_line', json_line)
            raise 

          ids = {}
          for key in self.args.json_keys:
              text = data[key]
              tokenized_sentences = Encoder.splitter.tokenize(text)
              if self.args.old_gpt_tokenize:
                doc_ids = [
                    sentence_ids
                    for sentence_ids in Encoder.tokenizer.batch_tokenize_old(tokenized_sentences)
                    if len(sentence_ids) > 0
                ]
              else:
                doc_ids = [
                    sentence_ids
                    for sentence_ids in Encoder.tokenizer.batch_tokenize(tokenized_sentences)
                    if len(sentence_ids) > 0
                ]
              if len(doc_ids) > 0 and self.args.append_eod:
                  doc_ids[-1].append(Encoder.tokenizer.eod)
              ids[key] = doc_ids
          return (ids, len(json_line))
      _iter=self.args.read_iter
      ret = []
      if _iter:
        with open(self.args.input, 'r', encoding='utf-8') as fin:
          fin.seek(job[0])
          while fin.tell() < job[0]+job[1]:
            json_line = fin.readline()
            if not json_line: 
              break
            ret.append(process_line(self, json_line))
      else:
        with  open(self.args.input, 'r', encoding='utf-8') as fin:
          fin.seek(job[0])
          all_jsonl = fin.read(job[1]).split('\n')
        while all_jsonl:
          json_line = all_jsonl[0]
          all_jsonl = all_jsonl[1:]
          ret.append(process_line(self, json_line))
      return ret

    # this does writing to shard files
    # if _iter is true, read line by line, instead read in blocks
    def encode_shard(self, job):

      def process_line(self, json_line, builders):
        try:
            data = json.loads(json_line)
        except:
            print ('json_line', json_line)
            raise
        ids = {}
        for key in self.args.json_keys:
            text = data[key]
            tokenized_sentences = Encoder.splitter.tokenize(text)
            if self.args.old_gpt_tokenize:
                doc_ids = [
                    sentence_ids
                    for sentence_ids in Encoder.tokenizer.batch_tokenize_old(tokenized_sentences)
                    if len(sentence_ids) > 0
                ]
            else:
                doc_ids = [
                    sentence_ids
                    for sentence_ids in Encoder.tokenizer.batch_tokenize(tokenized_sentences)
                    if len(sentence_ids) > 0
                ]
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

      ###
      _iter=self.args.read_iter
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
      num_docs=0
      if _iter:
        with open(self.args.input, 'r', encoding='utf-8') as fin:
          fin.seek(job[0])
          while fin.tell() < job[0]+job[1]:
            json_line = fin.readline()
            if not json_line: 
              break
            num_docs += 1
            process_line(self, json_line, builders)
      else:
        with  open(self.args.input, 'r', encoding='utf-8') as fin:
          fin.seek(job[0])
          all_jsonl = fin.read(job[1]).split('\n')
        while all_jsonl:
          num_docs += 1
          json_line = all_jsonl[0]
          all_jsonl = all_jsonl[1:]
          process_line(self, json_line, builders)

      for key in self.args.json_keys:
          builders[key].finalize(output_idx_files[key])

      return num_docs, [output_bin_files[key].replace(".bin", "") for key in self.args.json_keys], job[1]

def process_samples(simple_queue, process_index, args, level, writer: Connection):
    encoder = Encoder(args)
    encoder.initializer()

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_filename = f"{args.output_prefix}_{key}_{level}_{process_index}"
        output_bin_files[key] = data_file_path(output_filename)
        output_idx_files[key] = index_file_path(output_filename)
        builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                     impl=args.dataset_impl,
                                                     vocab_size=encoder.tokenizer.vocab_size)

    json_lines = simple_queue.get()
    while json_lines is not None:
        try:
            process_json_lines(json_lines, encoder, builders, writer)
        except:
            # Debugging code in order to understand why the encoder can fail
            for json_line in json_lines:
                try:
                    if json_line.strip() == "":
                        continue
                    encoder.encode(json_line)
                except:
                    print(repr(json_line))
                    print(json_line.strip() == "")
                    raise

        json_lines = simple_queue.get()

    # in case finished, we still need to add None to signal to everyone else
    simple_queue.put(None)
    # we need to send EOFError
    writer.close()

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])


def process_json_lines(json_lines, encoder, builders, writer):
    total_bytes_processed = 0
    for json_line in json_lines:
        if json_line.strip() == "":
            continue

        doc, bytes_processed = encoder.encode(json_line)

        total_bytes_processed += bytes_processed

        for key, sentences in doc.items():
            if len(sentences) == 0:
                continue
            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))
            builders[key].end_document()

    writer.send((len(json_lines), total_bytes_processed))


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, default=None,
                       help='Path to input JSON')
    group.add_argument('--multiprocessing_type', type=str, default='one_reader_one_writer', required=False,
                       choices=['queue_reader_writer','many_readers',
                                'many_readers_many_writers', 'one_reader_one_writer'],
                       help='What type of multiprocessing strategy to use.')
    group.add_argument('--read_iter', type=bool, default=True,
                       help='Read the input file iteratively instead of in batch_shard_size blocks')
    group.add_argument('--batch_shard_size', type=int, default=10,
                       help='Size of batch/shards in MBs')
    group.add_argument('--chunksize', type=int, default=25,
                       help='If we are using one reader, this is the number of lines allocated to each worker')
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
    group.add_argument('--old_gpt_tokenize', type=bool, default=False,
                       help='Use the old gpt tokenize method')
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
    group.add_argument('--less_logs', type=bool, default=False,
                       help="Don't display so many logs.")
    args = parser.parse_args()
    args.keep_empty = False

    if args.old_gpt_tokenize and args.tokenizer_type != 'GPT2BPETokenizer':
      print ("warning: old_gpt_tokenize only available for tokenizer-type == GPT2BPETokenizer")
      args.old_gpt_tokenize = False
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
                    file_segs.append((file_pos, file_size-file_pos))
                    break
                f.seek(file_pos+shard_size, 0)
                seg_len = shard_size
                line = f.readline()
                if not line:
                    file_segs.append((file_pos, file_size-file_pos))
                    break
                seg_len += len(line)
                if file_size-(file_pos+seg_len) < shard_size:
                    file_segs.append((file_pos, file_size-file_pos))
                    break

                file_segs.append((file_pos, seg_len))
                file_pos = f.tell()
          line = None
          return file_segs



def fill_simple_queue(filename, simple_queue, chunksize:int):
    with open(filename, "r") as f:
        print("Start filling queue")
        while True:
            acc = tuple(itertools.islice(f, chunksize))
            if len(acc) == 0:
                simple_queue.put(None)
                return
            simple_queue.put(acc)

def log(readers, log_interval, less_logs):
    print("Start Logging")
    proc_start = time.time()
    total_bytes_processed = 0
    doc_processed = 0
    logged_docs = 0
    elapsed = 1
    # we want to compute a rolling average of bytes processed over last 10k documents (more or less)
    bytes_queue_max_length = 10_000 // log_interval + 1
    bytes_queue = collections.deque(maxlen= bytes_queue_max_length)
    # we fill the queue with (start_time, 0)
    bytes_queue.extend([(proc_start, total_bytes_processed)]*bytes_queue_max_length)

    while len(readers) != 0:
        for r in multiprocessing.connection.wait(readers):
            try:
                nb_of_docs, bytes_processed = r.recv()
                total_bytes_processed += bytes_processed
                doc_processed += nb_of_docs
            except EOFError:
                r.close()
                readers.remove(r)

            if ((doc_processed - logged_docs) >= log_interval):
                logged_docs = doc_processed
                current = time.time()
                elapsed = current - proc_start

                (old_start_time, old_bytes) = bytes_queue.popleft()
                bytes_queue.append((current, total_bytes_processed))
                mbs = (total_bytes_processed - old_bytes) / (current - old_start_time) / 1024 / 1024
                if not less_logs: print(f"Processed {doc_processed} documents",
                      f"({doc_processed / elapsed} docs/s, {mbs} MB/s).")
    if less_logs:
        print(f"Processed {doc_processed} documents",
              f"({doc_processed / elapsed} docs/s, {mbs} MB/s).")
def main():
    args = get_args()
    startup_start = time.time()
    if args.datasets is not None:
      startup_start = time.time()
      print("Merging",args.datasets)
      tokenizer = build_tokenizer(args)
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
                                                vocab_size=tokenizer.vocab_size)
          for dataset in args.datasets:
            builders[key].merge_file_(dataset)

      startup_end = time.time()
      print("Time to merge:", startup_end - startup_start)

      for key in args.json_keys:
          builders[key].finalize(output_idx_files[key])

      print(f"Merged {len(args.datasets)} datasets to {args.output_prefix}")
    else:
      startup_start = time.time()
      if  args.multiprocessing_type == 'queue_reader_writer': 
        print (f"Doing: {args.multiprocessing_type}")
        print("Opening", args.input)
        simple_queue = multiprocessing.Queue() # we can also limit the number of elements to reduce the memory footprint.
        chunksize = args.chunksize

        if nltk_available and args.split_sentences:
            nltk.download("punkt", quiet=True)

        level = "document"
        if args.split_sentences:
            level = "sentence"

        assert args.workers > 2, "One worker is used for logging, one for filling the queue"
        readers, writers = list(zip(*[multiprocessing.Pipe(duplex=False) for _ in range(args.workers - 2)]))
        processes = [multiprocessing.Process(target=process_samples, args=(simple_queue, i, args, level, writer)) for i, writer in enumerate(writers)]
        log_thread = threading.Thread(target=log, args=(list(readers), args.log_interval, args.less_logs))
        fill_thread = multiprocessing.Process(target=fill_simple_queue, args=(args.input, simple_queue, chunksize))

        fill_thread.start()
        log_thread.start()
        for i, process in enumerate(processes):
            process.start()

        # We close the writable end of the pipe now to be sure that
        # p is the only process which owns a handle for it.  This
        # ensures that when p closes its handle for the writable end,
        # wait() will promptly report the readable end as being ready.
        # https://docs.python.org/fr/3/library/multiprocessing.html#multiprocessing.connection.Connection
        for writer in writers:
            writer.close()

        fill_thread.join()
        fill_thread.close()
        for process in processes:
            process.join()
            process.close()
        log_thread.join()

        # TODO: this may be done after.
        print("Merging files together")

        tokenizer = build_tokenizer(args)

        print(f"Vocab size: {tokenizer.vocab_size}")
        print(f"Output prefix: {args.output_prefix}")
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        for key in args.json_keys:
            output_filename = f"{args.output_prefix}_{key}_{level}"
            output_bin_files[key] = data_file_path(output_filename)
            output_idx_files[key] = index_file_path(output_filename)
            builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                        impl=args.dataset_impl,
                                                        vocab_size=tokenizer.vocab_size)

        for key in args.json_keys:
            for process_index in range(len(processes)):
                output_filename = f"{args.output_prefix}_{key}_{level}_{process_index}"
                builders[key].merge_file_(output_filename)
                os.unlink(output_filename+".bin")
                os.unlink(output_filename+".idx")
            builders[key].finalize(output_idx_files[key])


      elif args.multiprocessing_type == 'many_readers': # many readers, and one writer
        # non-queue based methods
        print (f"Doing: {args.multiprocessing_type}")
        print("Opening", args.input)
        if nltk_available and args.split_sentences:
            nltk.download("punkt", quiet=True)
        level = "document"
        if args.split_sentences:
            level = "sentence"

        encoder = Encoder(args)
        vocab_size = build_tokenizer(args).vocab_size

        pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
        shards = get_file_shard_ranges(args.input, args.batch_shard_size*1000000,  num_proc=args.workers) 
        if args.workers == 1:
          encoded_docs = map(encoder.encode_batch, shards)  
        else:
          encoded_docs = pool.imap(encoder.encode_batch, shards, chunksize=1)

        print(f"Vocab size: {vocab_size}")
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
        i = 0
        for ret in encoded_docs:
          for (doc, bytes_processed) in ret:
            i+=1
            total_bytes_processed += bytes_processed
            for key, sentences in doc.items():
                if len(sentences) == 0:
                    continue
                for sentence in sentences:
                    builders[key].add_item(torch.IntTensor(sentence))
                builders[key].end_document()
            if not args.less_logs and (i % args.log_interval == 0):
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed/elapsed/1024/1024
                print(f"Processed {i} documents",
                      f"({i/elapsed} docs/s, {mbs} MB/s).",
                      file=sys.stderr)
        if args.less_logs:
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
      elif args.multiprocessing_type == 'many_readers_many_writers':# this is the many readers and many writers to shards and merge method
        print (f"Doing: {args.multiprocessing_type}")
        print("Opening", args.input)
        if nltk_available and args.split_sentences:
            nltk.download("punkt", quiet=True)
        level = "document"
        if args.split_sentences:
            level = "sentence"

        encoder = Encoder(args)
        vocab_size = build_tokenizer(args).vocab_size

        pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
        shards = get_file_shard_ranges(args.input, args.batch_shard_size*1000000,  num_proc=args.workers) 
        if args.workers == 1:
          encoded_docs = map(encoder.encode_shard, shards)  
        else:
          encoded_docs = pool.imap(encoder.encode_shard, shards, chunksize=1)
        print(f"Vocab size: {vocab_size}")
        print(f"Output prefix: {args.output_prefix}")
        args.datasets =[]
        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        total_docs=0
        for i, (num_docs, shard_files, bytes_processed) in enumerate(encoded_docs, start=1):
            total_docs+=num_docs
            args.datasets.extend(shard_files)  
            total_bytes_processed += bytes_processed
            if not args.less_logs and (i % args.log_interval == 0):
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed/elapsed/1024/1024
                print(f"Processed {total_docs} documents",
                      f"({total_docs/elapsed} docs/s, {mbs} MB/s).",
                      file=sys.stderr)
        if args.less_logs:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {total_docs} documents",
                      f"({total_docs/elapsed} docs/s, {mbs} MB/s).",
                      file=sys.stderr)
        pool.close()
        pool.join()
        # let's merge
        # we could have done the merge incrementally but the dataset builder doesn't really do incremental merge efficiently
        merge_start = time.time()
        curr_keys = list(set([dataset.split(f"{args.output_prefix}_")[1].split("_")[0] for dataset in args.datasets]))
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        for key in curr_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                          key, level)

            builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                  impl=args.dataset_impl,
                                                  vocab_size=vocab_size)
            for dataset in args.datasets:
              if dataset.startswith(f"{args.output_prefix}_{key}"):
                print ("doing dataset", dataset)
                builders[key].merge_file_(dataset)

        merge_end = time.time()
        print("Time to merge:", merge_end - merge_start)

        for key in curr_keys:
            builders[key].finalize(output_idx_files[key])
    
        for dataset in args.datasets:
          if dataset != f"{args.output_prefix}_{key}_{level}":
            os.unlink(f"{dataset}.bin")
            os.unlink(f"{dataset}.idx")

        print(f"Merged {len(args.datasets)} datasets to {args.output_prefix}")

      else: # original method, one reader and one writer and many workers
        print (f"Doing: {args.multiprocessing_type}")
        print("Opening", args.input)
        if nltk_available and args.split_sentences:
            nltk.download("punkt", quiet=True)
        level = "document"
        if args.split_sentences:
            level = "sentence"

        encoder = Encoder(args)
        vocab_size = build_tokenizer(args).vocab_size

        fin = open(args.input, 'r', encoding='utf-8')
        pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
        if args.workers == 1:
          encoder.initializer()
          encoded_docs = map(encoder.encode, fin)
        else:
          encoded_docs = pool.imap(encoder.encode, fin, chunksize=args.chunksize)

        print(f"Vocab size: {vocab_size}")
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

        for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            for key, sentences in doc.items():
                if len(sentences) == 0:
                    continue
                for sentence in sentences:
                    builders[key].add_item(torch.IntTensor(sentence))
                builders[key].end_document()
            if not args.less_logs and (i % args.log_interval == 0):
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed/elapsed/1024/1024
                print(f"Processed {i} documents",
                      f"({i/elapsed} docs/s, {mbs} MB/s).",
                      file=sys.stderr)
        if args.less_logs:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {i} documents",
                      f"({i/elapsed} docs/s, {mbs} MB/s).",
                      file=sys.stderr)
        for key in args.json_keys:
            builders[key].finalize(output_idx_files[key])

    print("Finished everything in:", time.time() - startup_start)

if __name__ == '__main__':
  #import cProfile
  #cProfile.run('main()',  'restats')
  main()
