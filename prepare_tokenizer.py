from transformers import AutoTokenizer, AddedToken

tokenizer = AutoTokenizer.from_pretrained('bigscience/tokenizer')

tokenizer.add_special_tokens({
        'additional_special_tokens': [
            AddedToken(
                '<extra_id_{}>'.format(str(idx).zfill(3)),
                lstrip=False,
                rstrip=False,
                normalization=False
                ) for idx in reversed(range(0,200))
        ]
    })

tokenizer.save_pretrained('bigscience-tokenizer-padded')

# python tools/preprocess_data.py \
#     --input data/oscar-en-10k.jsonl \
#     --output-prefix data/meg-gpt2-oscar-en-10k \
#     --dataset-impl mmap \
#     --tokenizer-type PretrainedFromHF \
#     --tokenizer-name-or-path bigscience-tokenizer-padded \
#     --append-eod \
#     --workers 4