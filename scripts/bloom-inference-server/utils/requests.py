from typing import Any, List

from pydantic import BaseModel


class BaseResponse(BaseModel):
    query_id: int = None
    total_time_taken: str = None


class GenerateRequest(BaseModel):
    text: List[str] = None
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
    remove_input_from_output: bool = False
    method: str = "generate"


class GenerateResponse(BaseResponse):
    text: List[str] = None
    num_generated_tokens: List[int] = None
    method: str = "generate"


class TokenizeRequest(BaseModel):
    text: List[str] = None
    padding: bool = False
    method: str = "tokenize"


class TokenizeResponse(BaseResponse):
    token_ids: List[List[int]] = None
    attention_mask: List[List[int]] = None
    method: str = "tokenize"


def parse_bool(value: str) -> bool:
    if (value.lower() == "true"):
        return True
    elif (value.lower() == "false"):
        return False
    else:
        raise ValueError("{} is not a valid boolean value".format(value))


def parse_field(kwargs: dict,
                field: str,
                dtype: int,
                default_value: Any = None) -> Any:
    if (field in kwargs):
        if (type(kwargs[field]) == dtype):
            return kwargs[field]
        elif (dtype == bool):
            return parse_bool(kwargs[field])
        else:
            return dtype(kwargs[field])
    else:
        return default_value


def parse_generate_kwargs(text: List[str], kwargs: dict) -> GenerateRequest:
    return GenerateRequest(
        text=text,
        min_length=parse_field(kwargs, "min_length", int),
        do_sample=parse_field(kwargs, "do_sample", bool),
        early_stopping=parse_field(kwargs, "early_stopping", bool),
        num_beams=parse_field(kwargs, "num_beams", int),
        temperature=parse_field(kwargs, "temperature", float),
        top_k=parse_field(kwargs, "top_k", int),
        top_p=parse_field(kwargs, "top_p", float),
        typical_p=parse_field(kwargs, "typical_p", float),
        repitition_penalty=parse_field(kwargs, "repitition_penalty", float),
        bos_token_id=parse_field(kwargs, "bos_token_id", int),
        pad_token_id=parse_field(kwargs, "pad_token_id", int),
        eos_token_id=parse_field(kwargs, "eos_token_id", int),
        length_penalty=parse_field(kwargs, "length_penalty", float),
        no_repeat_ngram_size=parse_field(kwargs, "no_repeat_ngram_size", int),
        encoder_no_repeat_ngram_size=parse_field(
            kwargs, "encoder_no_repeat_ngram_size", int),
        num_return_sequences=parse_field(kwargs, "num_return_sequences", int),
        max_time=parse_field(kwargs, "max_time", float),
        max_new_tokens=parse_field(kwargs, "max_new_tokens", int),
        decoder_start_token_id=parse_field(
            kwargs, "decoder_start_token_id", int),
        num_beam_group=parse_field(kwargs, "num_beam_group", int),
        diversity_penalty=parse_field(kwargs, "diversity_penalty", float),
        forced_bos_token_id=parse_field(kwargs, "forced_bos_token_id", int),
        forced_eos_token_id=parse_field(kwargs, "forced_eos_token_id", int),
        exponential_decay_length_penalty=parse_field(
            kwargs, "exponential_decay_length_penalty", float),
        remove_input_from_output=parse_field(
            kwargs, "remove_input_from_output", bool, False)
    )


def get_filter_dict(d: BaseModel) -> dict:
    d = dict(d)
    q = {}
    for i in d:
        if (d[i] != None):
            q[i] = d[i]
    del q["text"]
    return q
