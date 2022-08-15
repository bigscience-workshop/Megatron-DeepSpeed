import argparse
from typing import List, Tuple, Union

import torch


class Model:
    def __init__(self, args: argparse.Namespace) -> None:
        self.tokenizer = None
        self.model = None
        self.input_device = None
        raise NotImplementedError("This is a dummy class")

    def generate(self,
                 text: Union[str, List[str]],
                 generate_kwargs: dict,
                 remove_input_from_output: bool = False) -> Union[Tuple[str, int],
                                                                  Tuple[List[str], List[int]]]:
        return_type = type(text)
        if (return_type == str):
            text = [text]

        input_tokens = self.tokenizer(text, return_tensors="pt", padding=True)

        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.input_device)

        with torch.no_grad():
            output_tokens = self.model.generate(
                **input_tokens,
                **generate_kwargs
            )

        input_token_lengths = [x.shape[0] for x in input_tokens.input_ids]
        output_token_lengths = [x.shape[0] for x in output_tokens]
        generated_tokens = [
            o - i for i, o in zip(input_token_lengths, output_token_lengths)]

        if (remove_input_from_output):
            output_tokens = [x[-i:]
                             for x, i in zip(output_tokens, generated_tokens)]

        output_text = self.tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True)

        if (return_type == str):
            output_text = output_text[0]
            generated_tokens = generated_tokens[0]

        return output_text, generated_tokens

    def shutdown(self) -> None:
        exit()
