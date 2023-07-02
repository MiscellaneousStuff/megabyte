import os
import neptune
import math
import random

import gzip
import torch
import numpy as np

import contextlib
import random
import tqdm

from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.cuda.amp.grad_scaler import GradScaler

import megabyte
from sophiag import SophiaG
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
# import tiktoken

# Optimisers
optimiser_adam            = "adam"
optimiser_sophiag         = "sophiag"

# Tokenisation
token_char                = "char"
token_tiktoken            = "tiktoken"
token_gpt_neo             = "gpt_neo"

# Hyperparameters
AMP                       = True
RANDOM_SEED               = 42
NUM_BATCHES               = int(30_000) # int(1e5)
BATCH_SIZE                = 4 * 1 # 1 # 4 * 2
GRADIENT_ACCUMULATE_EVERY = 4
VALIDATE_EVERY            = 100
GENERATE_EVERY            = 500
PRIME_LEN                 = 100 
SEQ_LEN                   = 512         # Taken from from MegaByte paper
WARMUP_ITERS              = 500 # 500   # Taken from from MegaByte paper
ADAM_BETA_1               = 0.9         # Taken from from MegaByte paper
ADAM_BETA_2               = 0.98        # Taken from from MegaByte paper
LR_DECAY_ITERS            = 0 # NUM_BATCHES # max_iters per Chinchilla?
DECAY_LR                  = True
MAX_LEARNING_RATE         = 2e-4 # 6e-4 # 2e-4 # Taken from from MegaByte paper
MIN_LEARNING_RATE         = 2e-5 # 6e-5 # 2e-5 # Common to do MAX_LEARNING_RATE * 0.1
WEIGHT_DECAY              = 0.1 # 0.2
OPTIMISER                 = optimiser_adam # optimiser_sophiag
TOKENISATION              = token_gpt_neo

# Model Parameters
DIM_HEAD    = 32
HEADS       = 8 # 12
NUM_TOKENS  = 256
DIM         = (768, 512, 256) # (768, 512, 256)
DEPTH       = (12, 4, 2) # (12, 4, 2)
MAX_SEQ_LEN = (32, 4, 4)
FLASH_ATTN  = False

SAVE        = False

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

#' tiktoken_enc = tiktoken.get_encoding("cl100k_base")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
if TOKENISATION == token_gpt_neo:
    NUM_TOKENS = tokenizer.vocab_size

def get_reserved_mem_gb():
    device = torch.cuda.current_device()
    reserved = torch.cuda.memory_reserved(device)
    reserved_gb = reserved / 1024 / 1024 / 1024
    return reserved_gb

def load_env(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            # Skip empty lines and comments
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

load_env(".env")

run = neptune.init_run(
    name="",
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

def cycle(loader, infinite=True):
    while True:
        for data in loader:
            yield data
        if not infinite:
            break

def decode_token(token):
    if 32 <= token <= 126:
        return str(chr(token))
    else:
        return ''

def decode_tokens(tokens):
    if TOKENISATION == token_char:
        return ''.join(list(map(decode_token, tokens)))
    elif TOKENISATION == token_gpt_neo:
        dec = tokenizer.decode(tokens, skip_special_tokens=True)
        dec = [ord(d) for d in dec]
        dec = ''.join(list(map(decode_token, dec)))
        return dec

class WrappedDataset(IterableDataset):
    def __init__(self, huggingface_dataset, seq_len, infinite=True):
        self.huggingface_dataset = huggingface_dataset
        self.seq_len = seq_len
        self.infinite = infinite
        
    def __iter__(self):
        buffer = torch.tensor([], dtype=torch.long)
        while True:  # Infinite loop over the dataset
            for row in self.huggingface_dataset:
                formatted_text = row['text']
                if TOKENISATION == token_char:
                    x = np.frombuffer(formatted_text.encode(), dtype=np.uint8).copy()
                elif TOKENISATION == token_gpt_neo:
                    x = np.array(
                        tokenizer.encode(formatted_text)).copy()
                buffer = torch.cat((buffer, torch.from_numpy(x)), dim=0)
                while len(buffer) >= self.seq_len:
                    yield buffer[:self.seq_len]\
                        .long() \
                        .pin_memory() \
                        .to("cuda", non_blocking=True)
                    buffer = buffer[self.seq_len:]
            if not self.infinite:
                if len(buffer):
                    yield buffer\
                        .long() \
                        .pin_memory() \
                        .to("cuda", non_blocking=True)
                break

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

def train(model, scaler, train_loader):
    model.train()
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            loss = model(next(train_loader), return_loss = True)
            # loss.backward()
            scaler.scale(loss).backward()

    run["loss"].append(loss)
    
    return loss

def test(model, val_loader):
    model.eval()
    with torch.no_grad():
        loss = model(next(val_loader), return_loss = True)
        run["vloss"].append(loss)
        run["perplexity"].append(torch.exp(loss))
        return loss

def generate(model, gen_loader, PRIME_LEN=100):
    model.eval()
    if TOKENISATION == token_char:
        inp = next(gen_loader)[:-1]
        prime_inp = inp[:PRIME_LEN]
        # print("prime_inp:", prime_inp.shape)
        prime = decode_tokens(prime_inp)
    elif TOKENISATION == token_gpt_neo:
        inp = next(gen_loader)[-1:]
        prime_inp = inp[:, :PRIME_LEN]
        # print("prime_inp:", prime_inp.shape)
        # print(prime_inp, type(prime_inp), prime_inp.shape)
        prime = decode_tokens(prime_inp[0])
    print("final prime:", prime)
    print(f'%s \n\n %s', (prime, '*' * PRIME_LEN))
    run["prompts"].append(prime)

    if TOKENISATION == token_char:
        sample = model.generate(prime_inp[None, :])
        sample = sample.flatten(1)
    else:
        print("ello:", prime_inp)
        sample = model.generate(prime_inp)
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
        PRIME_LEN,
        gen_loader):

    best_val = float("inf")

    if OPTIMISER == optimiser_adam:
        optim = torch.optim.Adam(
        model.parameters(),
        lr=MAX_LEARNING_RATE,
        betas=(ADAM_BETA_1, ADAM_BETA_2))
    else:
        optim = SophiaG(
            model.parameters(),
            lr=MAX_LEARNING_RATE,
            betas=(ADAM_BETA_1, ADAM_BETA_2),
            weight_decay=0.1)
    
    scaler = GradScaler()

    # optim.load_state_dict(torch.load("./megabyte_4200_1.292237401008606.pt"))

    pbar = tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training')
    for i in pbar:
        # Set LR
        lr = get_lr(i) if DECAY_LR else MAX_LEARNING_RATE
        for param_group in optim.param_groups:
            param_group['lr'] = lr
        run["lr"].append(lr)

        # Perform one GRADIENT_ACCUMULATE number of forward and backward passes, then update model
        loss = train(model, scaler, train_loader)
        print(f'training loss: {loss.item()}')
        mem_gb = get_reserved_mem_gb()
        pbar.set_description(f"Reserved Memory (GB): {mem_gb}, loss: {loss.item()}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # (same as paper?) ,0.5)
        optim.step()
        scaler.step(optim)
        scaler.update()

        # Zero gradients
        optim.zero_grad()

        # Validate every `n` steps (because it's time consuming)
        if i % VALIDATE_EVERY == 0:
            vloss = test(model, val_loader)
            print(f'validation loss: {vloss.item()}')
            pbar.set_description(f"Reserved Memory (GB): {mem_gb}, loss: {loss.item()}, vloss: {vloss.item()}")

        # Save best model every `n` steps. Set this to be high as the models are huge
        if vloss < best_val:
            best_val = vloss
            if SAVE:
                torch.save(
                    model.state_dict(),
                    f"./megabyte_{i}_{vloss}.pt")
                torch.save(
                    optim.state_dict(),
                    f"./megabyte_{i}_{vloss}_optim.pt")
        
        # Generate a sequence from a prompt every once in a while
        # if i != 0 and i % GENERATE_EVERY == 0:
        if i % GENERATE_EVERY == 0:
            print("GENERATE")
            generate(model, gen_loader)

# def load_dataset(src, load_bytes=95e6, split_bytes=90e6):
#     with gzip.open(src) as file:
#         x = np.frombuffer(file.read(int(load_bytes)), dtype=np.uint8).copy()

#         train_x, valid_x = np.split(x, [int(split_bytes)])
#         data_train, data_val = map(torch.from_numpy, (train_x, valid_x))

#     return data_train, data_val

# def load_dataset(src, load_bytes=1_000_000_000, split_bytes=990_000_000):
#     with open(src, "br") as file:
#         # file.read(int(load_bytes))
#         x = np.memmap(src, dtype=np.uint8, mode="r").copy()

#         train_x, valid_x = np.split(x, [int(split_bytes)])
#         data_train, data_val = map(torch.from_numpy, (train_x, valid_x))

#     return data_train, data_val

# def load_dataset():
#     pass

def main():
    # Set hyperparameters in log
    config_keys = [k for k, v in globals().items()
                   if k.isupper() and isinstance(v, (int, float, bool, str))]
    for k in config_keys:
        run[k] = globals()[k]

    # skeskinen/TinyStories-Instruct-hf

    raw_ds = load_dataset("skeskinen/TinyStories-Instruct-hf")
    ds = raw_ds["train"].train_test_split(test_size=0.01)
    train_dataset = WrappedDataset(ds["train"], SEQ_LEN, infinite=True)
    val_dataset   = WrappedDataset(ds["test"], SEQ_LEN, infinite=True)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE), infinite=True)
    val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE), infinite=True)
    gen_loader    = iter(val_loader)

    # raw_ds = load_dataset("Cohere/wikipedia-22-12-simple-embeddings")
    # ds = raw_ds["train"].train_test_split(test_size=0.01)
    # train_dataset = WrappedDataset(ds["train"], SEQ_LEN, infinite=False)
    # val_dataset   = WrappedDataset(ds["test"], SEQ_LEN, infinite=False)
    # train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE), infinite=True)
    # val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE), infinite=True)

    # enwik8
    # data_train, data_val = load_dataset("./data/enwik8.gz")
    # train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    # val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
    # train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
    # val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

    # HuggingFace: RedPajama-Data-1T-Sample
    # raw_ds = load_dataset("togethercomputer/RedPajama-Data-1T-Sample")
    # ds = raw_ds["train"].train_test_split(test_size=0.01)
    # train_dataset = WrappedDataset(ds["train"], SEQ_LEN, infinite=False)
    # val_dataset   = WrappedDataset(ds["test"], SEQ_LEN, infinite=False)
    # train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE), infinite=False)
    # val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE), infinite=False)

    model = megabyte.MEGABYTE(
        dim_head=DIM_HEAD,
        heads=HEADS,
        num_tokens=NUM_TOKENS,
        dim=DIM,
        depth=DEPTH,
        max_seq_len=MAX_SEQ_LEN,
        flash_attn=FLASH_ATTN
    ).cuda()

    # model.load_state_dict(torch.load("./PATH_TO_MODEL_CHECKPOINT.pt"))
    # optim.load_state_dict(torch.load("./PATH_TO_OPTIM_CHECKPOINT.pt"))

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
               PRIME_LEN=PRIME_LEN,
               gen_loader=gen_loader)

    run.stop()

if __name__ == "__main__":
    main()