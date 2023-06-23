import os
import neptune
import math

import gzip
import torch
import numpy as np

import contextlib
import random
import tqdm

from torch.utils.data import DataLoader, Dataset

import megabyte
from lr_sched import polynomial_decay_lr_schedule

# Hyperparameters
NUM_BATCHES               = int(1e5)
BATCH_SIZE                = 4
GRADIENT_ACCUMULATE_EVERY = 4
VALIDATE_EVERY            = 100
GENERATE_EVERY            = 500
PRIME_LEN                 = 100
SEQ_LEN                   = 8192        # Taken from from MegaByte paper
WARMUP_ITERS              = 500         # Taken from from MegaByte paper
ADAM_BETA_1               = 0.9         # Taken from from MegaByte paper
ADAM_BETA_2               = 0.98        # Taken from from MegaByte paper
LR_DECAY_ITERS            = NUM_BATCHES # max_iters per Chinchilla?
DECAY_LR                  = True
MAX_LEARNING_RATE         = 2e-4 # Taken from from MegaByte paper
MIN_LEARNING_RATE         = 2e-5 # Common to do MAX_LEARNING_RATE * 0.1

# Model Parameters
DIM_HEAD    = 64
HEADS       = 8
NUM_TOKENS  = 256
DIM         = (768, 512, 256)
DEPTH       = (6, 4, 2)
MAX_SEQ_LEN = (512, 4, 4)
FLASH_ATTN  = False

def load_env(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            # Skip empty lines and comments
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

load_env(".env")

run = neptune.init_run(
    project=os.getenv("NEPTUNE_PROJECT"),
    api_token=os.getenv("NEPTUNE_KEY"))

# Taken from: https://github.com/karpathy/nanoGPT/blob/master/train.py
# Learning rate decay scheduler (Cosine with Warmup)
def get_lr(it):
    # 1) Linear warmup for warmup_iters steps
    if it < WARMUP_ITERS:
        return MAX_LEARNING_RATE * it / WARMUP_ITERS
    
    # 2) If it > lr_decay_iters, return min learning rate
    if it > NUM_BATCHES:
        return MIN_LEARNING_RATE
    
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - WARMUP_ITERS) / (NUM_BATCHES - WARMUP_ITERS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return MIN_LEARNING_RATE + coeff * (MAX_LEARNING_RATE - MIN_LEARNING_RATE)

def cycle(loader):
        while True:
            for data in loader:
                yield data

def decode_token(token):
    if 32 <= token <= 126:
        return str(chr(token))
    else:
        return ''

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len
    
# model = torch.compile(model)

# amp = True

def train(model, train_loader):
    model.train()
    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(next(train_loader), return_loss = True)
            loss.backward()
            # scaler.scale(loss).backward()

    run["loss"].append(loss)
    
    return loss

def test(model, val_loader):
    model.eval()
    with torch.no_grad():
        loss = model(next(val_loader), return_loss = True)
        run["vloss"].append(loss)
        run["perplexity"].append(torch.exp(loss))
        return loss

def generate(model, val_dataset, PRIME_LEN=100):
    model.eval()
    inp = random.choice(val_dataset)[:-1]
    prime_inp = inp[:PRIME_LEN]
    prime = decode_tokens(prime_inp)
    print(f'%s \n\n %s', (prime, '*' * 100))
    run["prompts"].append(prime)

    sample = model.generate(prime_inp[None, :])
    sample = sample.flatten(1)

    output_str = decode_tokens(sample[0][PRIME_LEN:])
    print(output_str)
    run["generated_outputs"].append(output_str)

def go(model,
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
        NUM_BATCHES,
        GRADIENT_ACCUMULATE_EVERY,
        VALIDATE_EVERY,
        GENERATE_EVERY,
        PRIME_LEN):

    best_val = float("inf")

    optim = torch.optim.Adam(
        model.parameters(),
        lr=MAX_LEARNING_RATE,
        betas=(ADAM_BETA_1, ADAM_BETA_2))
    # scaler = GradScaler()

    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
        # Set LR
        lr = get_lr(i) if DECAY_LR else MAX_LEARNING_RATE
        for param_group in optim.param_groups:
            param_group['lr'] = lr
        run["lr"].append(lr)

        # Perform one GRADIENT_ACCUMULATE number of forward and backward passes, then update model
        loss = train(model, train_loader)
        print(f'training loss: {loss.item()}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        # scaler.step(optim)
        # scaler.update()

        # Zero gradients
        optim.zero_grad()

        # Validate every `n` steps (because it's time consuming)
        if i % VALIDATE_EVERY == 0:
            vloss = test(model, val_loader)
            print(f'validation loss: {vloss.item()}')

        # Save best model every `n` steps. Set this to be high as the models are huge
        if vloss < best_val:
            best_val = vloss
            torch.save(
                model.state_dict(),
                f"./megabyte_{i}_{vloss}.pt")
            torch.save(
                model.state_dict(),
                f"./megabyte_{i}_{vloss}.pt")
        
        # Generate a sequence from a prompt every once in a while
        if i != 0 and i % GENERATE_EVERY == 0:
            generate(model, val_dataset)

def load_dataset(src):
    with gzip.open(src) as file:
        x = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()

        train_x, valid_x = np.split(x, [int(90e6)])
        data_train, data_val = map(torch.from_numpy, (train_x, valid_x))

    return data_train, data_val

def main():
    # Set hyperparameters in log
    config_keys = [k for k, v in globals().items()
                   if k.isupper() and isinstance(v, (int, float, bool, str))]
    for k in config_keys:
        run[k] = globals()[k]

    data_train, data_val = load_dataset("./data/enwik8.gz")

    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

    model = megabyte.MEGABYTE(
        dim_head=64,
        heads=8,
        num_tokens=256,
        dim=(768, 512, 256),
        depth=(6, 4, 2),
        max_seq_len=(512, 4, 4),
        flash_attn=False
    ).cuda()

    with open('output.txt', 'w') as f:
        with contextlib.redirect_stdout(f):
            go(model=model,
               train_loader=train_loader,
               val_loader=val_loader,
               train_dataset=train_dataset,
               val_dataset=val_dataset,
               NUM_BATCHES=NUM_BATCHES,
               GRADIENT_ACCUMULATE_EVERY=GRADIENT_ACCUMULATE_EVERY,
               VALIDATE_EVERY=VALIDATE_EVERY,
               GENERATE_EVERY=GENERATE_EVERY,
               PRIME_LEN=PRIME_LEN)

    run.stop()

if __name__ == "__main__":
    main()