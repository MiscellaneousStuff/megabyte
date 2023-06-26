import torch
import megabyte
from torch.utils.data import DataLoader

from train import generate, load_dataset, SEQ_LEN, TextSamplerDataset, cycle, BATCH_SIZE, PRIME_LEN
from train import DIM_HEAD, \
                  HEADS, \
                  NUM_TOKENS, \
                  DIM, \
                  DEPTH, \
                  MAX_SEQ_LEN, \
                  FLASH_ATTN

MODEL_CHECKPOINT = "./megabyte_3000_1.4545425176620483.pt"

def main():
    data_train, data_val = load_dataset("./data/enwik8.gz")

    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

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

    generate(model, val_dataset, PRIME_LEN)

if __name__ == "__main__":
    main()