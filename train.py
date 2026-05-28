import argparse
import random

import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from data import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="google/flan-t5-base", type=str)
parser.add_argument("--max-steps", default=10_000, type=int)
parser.add_argument("--log-every", default=20, type=int)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--hub-id", default="lecslab/dialog-inpainter-OQ")
args = parser.parse_args()

wandb.init(project="inpainting", config=vars(args))

device = "cuda" if torch.cuda.is_available() else "mps"
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.model)
model = model.to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

dataset = load_dataset()


def collate(batch):
    texts: list[str] = []
    targets: list[str] = []
    for d in batch:
        # Use different masks per epoch
        masked_idx = random.randint(0, len(d["turns"]) - 1)
        text = " ".join(
            [
                f"0:{turn['question'] if idx != masked_idx else '<extra_id_0>'} 1:{turn['response']}"
                for idx, turn in enumerate(d["turns"])
            ]
        )
        texts.append(text)
        targets.append(d["turns"][masked_idx]["question"])
    return tokenizer(
        texts, text_target=targets, padding=True, truncation=True, return_tensors="pt"
    ).to(device)


dataloaders = {
    split: DataLoader(
        exs,  # type:ignore
        batch_size=args.batch_size,
        shuffle=split == "train",
        collate_fn=collate,
    )
    for split, exs in dataset.items()
}

cur_step = 0
cur_epoch = 0
pbar = tqdm(total=args.max_steps, desc="Training")


@torch.no_grad
def eval(model):
    model.eval()
    eval_loss = 0
    for eval_batch in dataloaders["test"]:
        batch_loss = model(**eval_batch).loss.item()
        eval_loss += batch_loss / len(dataloaders["test"])
    return eval_loss


def grad_norm(model):
    # Log grad norm
    grad_norm = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        grad_norm += param_norm.item() ** 2
    grad_norm = grad_norm**0.5
    return grad_norm


train_loss = 0
train_grad_norm = 0

with torch.autocast(device_type=device, dtype=torch.bfloat16):
    while cur_step < args.max_steps:
        print(f"Beginning epoch {cur_epoch}")
        for batch_idx, batch in enumerate(dataloaders["train"]):
            model.train()
            out = model(**batch)
            out.loss.backward()
            unclipped_grad_norm = grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            # Running average since last log
            steps_since_last_log = cur_step % args.log_every
            train_loss = (
                steps_since_last_log / (steps_since_last_log + 1) * train_loss
                + (1 / (steps_since_last_log + 1)) * out.loss.detach().item()
            )
            train_grad_norm = (
                steps_since_last_log / (steps_since_last_log + 1) * train_grad_norm
                + (1 / (steps_since_last_log + 1)) * unclipped_grad_norm
            )

            if (cur_step + 1) % args.log_every == 0:
                eval_loss = eval(model)
                print(f"Train loss={train_loss}\tEval loss={eval_loss}")
                wandb.log(
                    {
                        "train": {"loss": train_loss, "grad_norm": train_grad_norm},
                        "eval": {"loss": eval_loss},
                    },
                    step=cur_step,
                )
            cur_step += 1
            pbar.update()
            if cur_step >= args.max_steps:
                break
        cur_epoch += 1

model.push_to_hub(args.hub_id)
tokenizer.push_to_hub(args.hub_id)
