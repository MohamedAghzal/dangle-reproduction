#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys

from collections import Counter
from multiprocessing import Pool

from fairseq.data.encoders.gpt2_bpe import get_encoder

from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')


def main():
    """
    Helper script to encode raw text with the GPT-2 BPE using multiple processes.

    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json",
        help='path to encoder.json',
    )
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help='path to vocab.bpe',
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=['-'],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=['-'],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs), \
        "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        # encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

        inputs = list(zip(*inputs))
        inputs = [i[0].strip() for i in inputs]
        # for inputs_idx, (_inputs) in enumerate(inputs):
        #     print(f"  inputs[{inputs_idx}]: {_inputs}")
            

        # encoded_lines = inputs[:10]
        encoded_lines = inputs
        encoded_lines = [f"{l}".strip() for l in encoded_lines]
        # for encoded_lines_idx, (_encoded_lines) in enumerate(encoded_lines):
        #     print(f"  pre encoded_lines[{encoded_lines_idx}]: {_encoded_lines}")

        encoded_lines = [tokenizer.encode(line)[1:] for line in encoded_lines]

        # for encoded_lines_idx, (_encoded_lines) in enumerate(encoded_lines):
        #     print(f"  encoded_lines[{encoded_lines_idx}]: {_encoded_lines}")
        encoded_lines = [f"{l}" for l in encoded_lines]
        # for encoded_lines_idx, (_encoded_lines) in enumerate(encoded_lines):
        #     print(f"  encoded_lines[{encoded_lines_idx}]: {_encoded_lines}")

        stats = Counter()
        for i, (enc_lines) in enumerate(encoded_lines,):
            #   enc_line: "39565 7505 357 1988 837 1635 24808 1267"
            def clean(x):
                replace_chars = '[],'
                for ch in replace_chars:
                    x = x.replace(ch, '')
                return x.strip()
            
            enc_lines = f"{enc_lines}"
            enc_lines = clean(enc_lines)
            enc_lines = f"\"{enc_lines}\""
            if i < 5:
                print(f"*" * 20,)
                print(f"  inputs[i]: {inputs[i]}")
                print(f"  enc_lines: {enc_lines}")
                print(f"*" * 20,)


            for output_h in outputs:
                print(enc_lines, file=output_h)

            # for enc_line, output_h in zip(enc_lines, outputs):
            #     print(enc_line, file=output_h)
            # # if filt == "PASS":
            # #     for enc_line, output_h in zip(enc_lines, outputs):
            # #         print(enc_line, file=output_h)
            # #         print(f"  enc_line: \"{enc_line}\"")
            # else:
            #     stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)

def monolingual_main():
    """
    Helper script to encode raw text with the GPT-2 BPE using multiple processes.

    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json",
        help='path to encoder.json',
    )
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help='path to vocab.bpe',
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=['-'],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=['-'],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs), \
        "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)
        encoded_lines = list(encoded_lines)[:10]
        print(f"*" * 60,)
        print(f"  encoded_lines: {encoded_lines}")

        stats = Counter()
        for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(f"  output_h: {output_h}")
                    print(enc_line, file=output_h)
                    print(f"  enc_line: \"{enc_line}\"")
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)

## Uncomment to use the original monolingual tokenizer
# main = monolingual_main
class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]

# """Byte pair encoding utilities"""
# import os
# import sentencepiece as spm

# class Encoder:
#     def __init__(self, filename):
#         self.sp = spm.SentencePieceProcessor()
#         self.sp.Load(filename)

#     def encode(self, text):
#         return self.sp.EncodeAsIds(text)

#     def decode(self, tokens):
#         return self.sp.DecodeIds(tokens.tolist()).replace('<|n|>', '\n')

# def get_encoder(model_name, models_dir):
#     return Encoder(os.path.join(models_dir, model_name, 'sp.model'))

if __name__ == "__main__":
    main()
