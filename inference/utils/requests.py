from typing import List, Union

from pydantic import BaseModel


def parse_bool(value: str) -> bool:
    if (value.lower() == "true"):
        return True
    elif (value.lower() == "false"):
        return False
    else:
        raise ValueError("{} is not a valid boolean value".format(value))


def parse_field(kwargs: dict, field: str, dtype: int, default_value: Any = None) -> Any:
    if (field in kwargs):
        if (dtype == bool):
            return parse_bool(kwargs[field])
        else:
            return dtype(kwargs[field])
    else:
        return default_value


class GenerateRequest(BaseModel):
    text: Union[List[str], str]
    min_length: int = None
    do_sample: bool = None
    early_stopping: bool = None
    num_beams: int = None
    temperature: float = None
    top_k: int = None
    top_p: float = None
    typical_p: float = None
    repitition_penalty: float = None
    bos_token_id: int = None
    pad_token_id: int = None
    eos_token_id: int = None
    length_penalty: float = None
    no_repeat_ngram_size: int = None
    encoder_no_repeat_ngram_size: int = None
    num_return_sequences: int = None
    max_time: float = None
    max_new_tokens: int = None
    decoder_start_token_id: int = None
    num_beam_groups: int = None
    diversity_penalty: float = None
    forced_bos_token_id: int = None
    forced_eos_token_id: int = None
    exponential_decay_length_penalty: float = None
    bad_words_ids: List[int] = None
    force_words_ids: Union[List[int], List[List[int]]] = None
    remove_input_from_output: bool = False

    def __init__(self, text: Union[List[str], str], kwargs: dict) -> None:
        self.text = text
        self.min_length = parse_field(kwargs, "min_length", int)
        self.do_sample = parse_field(kwargs, "do_sample", bool)
        self.early_stopping = parse_field(kwargs, "early_stopping", bool)
        self.num_beams = parse_field(kwargs, "num_beams", int)
        self.temperature = parse_field(kwargs, "temperature", float)
        self.top_k = parse_field(kwargs, "top_k", int)
        self.top_p = parse_field(kwargs, "top_p", float)
        self.typical_p = parse_field(kwargs, "typical_p", float)
        self.repitition_penalty = parse_field(
            kwargs, "repitition_penalty", float)
        self.bos_token_id = parse_field(kwargs, "bos_token_id", int)
        self.pad_token_id = parse_field(kwargs, "pad_token_id", int)
        self.eos_token_id = parse_field(kwargs, "eos_token_id", int)
        self.length_penalty = parse_field(kwargs, "length_penalty", float)
        self.no_repeat_ngram_size = parse_field(
            kwargs, "no_repeat_ngram_size", int)
        self.encoder_no_repeat_ngram_size = parse_field(
            kwargs, "encoder_no_repeat_ngram_size", int)
        self.num_return_sequences = parse_field(
            kwargs, "num_return_sequences", int)
        self.max_time = parse_field(kwargs, "max_time", float)
        self.max_new_tokens = parse_field(kwargs, "max_new_tokens", int)
        self.decoder_start_token_id = parse_field(
            kwargs, "decoder_start_token_id", int)
        self.num_beam_group = parse_field(kwargs, "num_beam_group", int)
        self.diversity_penalty = parse_field(
            kwargs, "diversity_penalty", float)
        self.forced_bos_token_id = parse_field(
            kwargs, "forced_bos_token_id", int)
        self.forced_eos_token_id = parse_field(
            kwargs, "forced_eos_token_id", int)
        self.exponential_decay_length_penalty = parse_field(
            kwargs, "exponential_decay_length_penalty", float),
        self.remove_input_from_output = parse_field(
            kwargs, "remove_input_from_output", bool, False)


class GenerateResponse(BaseModel):
    text: Union[List[str], str] = None
    num_generated_tokens: Union[List[int], int] = None
    query_id: int = None
    total_time_taken: float = None
