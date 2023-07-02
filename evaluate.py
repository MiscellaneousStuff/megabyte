import numpy as np

import torch
import megabyte
from torch.utils.data import DataLoader

from train import generate, load_dataset, SEQ_LEN, TextSamplerDataset, cycle, BATCH_SIZE, PRIME_LEN, WrappedDataset
from train import DIM_HEAD, \
                  HEADS, \
                  NUM_TOKENS, \
                  DIM, \
                  DEPTH, \
                  MAX_SEQ_LEN, \
                  FLASH_ATTN, \
                  decode_tokens

MODEL_CHECKPOINT = "./megabyte_2700_1.5388739109039307.pt"

def generate(model, prompt, PRIME_LEN=100):
    model.eval()
    inp = torch.tensor([ord(c) for c in prompt]).cuda()
    prime_inp = inp[:PRIME_LEN]
    # print("prime_inp:", prime_inp.shape)
    prime = decode_tokens(prime_inp)
    print(f'%s \n\n %s', (prime, '*' * 100))

    sample = model.generate(prime_inp[None, :])
    sample = sample.flatten(1)

    output_str = decode_tokens(sample[0][PRIME_LEN:])
    print(output_str)

def main():
    raw_ds = load_dataset("togethercomputer/RedPajama-Data-1T-Sample")
    ds = raw_ds["train"].train_test_split(test_size=0.01)
    train_dataset = WrappedDataset(ds["train"], SEQ_LEN, infinite=False)
    val_dataset   = WrappedDataset(ds["test"], SEQ_LEN, infinite=False)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE), infinite=False)
    val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE), infinite=False)

    model = megabyte.MEGABYTE(
        dim_head=DIM_HEAD,
        heads=HEADS,
        num_tokens=NUM_TOKENS,
        dim=DIM,
        depth=DEPTH,
        max_seq_len=MAX_SEQ_LEN,
        flash_attn=FLASH_ATTN
    ).cuda()

    model.load_state_dict(torch.load(MODEL_CHECKPOINT))

    generate(model, "import math", PRIME_LEN)

if __name__ == "__main__":
    main()