from gpt2 import GPT, GPTConfig
from eval import get_belebele, evaluate
# from scratch import check_dtypes

import os
import math
import tiktoken
import torch
import torch.nn as nn
import numpy as np
import pytz
import wandb
from datetime import datetime
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, inputs=None, process_rank=0, num_processes=1, verbose=False, split=None):
        self.B = B
        self.T = T
        self.inputs = inputs
        self.process_rank = process_rank
        self.num_processes = num_processes

        if inputs == "shakespeare":
            enc = tiktoken.get_encoding("gpt2")
            with open("input.txt", "r") as file:
                text = file.read()
            tokens = enc.encode(text)
            self.tokens = torch.tensor(tokens)
            if verbose:
                print(f"Loaded a total of {len(tokens)} tokens")
                print(f"1 epoch = {len(tokens) // (B * T)} batches")

            self.max_batches = len(tokens) // (B * T)
            self.reset()
        elif inputs == "fineweb":
            assert split in ("train", "val"), "split must be in ['train', 'val']"
            data_root = "/root/Llama/CPT/wikipedia"
            shards = os.listdir(data_root)
            shards = [shard for shard in shards if split in shard]
            shards = sorted(shards)
            shards = [os.path.join(data_root, shard) for shard in shards]
            assert len(shards) > 0, "no shards found"
            self.shards = shards
            if master_process:
                print(f"Found {len(shards)} shards for split {split}")
            self.reset()

    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes # im not entirely sure of this, like intuition wise, but this seems similar to how CUDA kernels do it
        # reset if current_position > tokens length
        if self.inputs == "shakespeare":
            if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
                self.current_position = self.B * self.T * self.process_rank
        elif self.inputs == "fineweb":
            if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
                self.current_position = B * T * self.process_rank
        return x, y
    
    def reset(self):
        self.current_position = self.B * self.T * self.process_rank
        if self.inputs == "fineweb":
            self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
        return self
    

# Distributed Data Parallel (DDP) setup
# 1. simple run: python train_gpt2.py
# 2. ddp run: torchrun --standalone --nproc_per_node=2 train_gpt2.py
# standalone is used for single node multi-gpu training
ddp = int(os.environ.get("RANK", -1)) != -1 
# ^if doesnt exist then we return -1, and by -1 or else decide whether ddp
if ddp:
    assert torch.cuda.is_available(), "no CUDA device available"
    init_process_group(backend="nccl") # on backends: https://pytorch.org/docs/stable/distributed.html
    # these env variables are set by torchrun
    ddp_rank = int(os.environ["RANK"]) # RANK is the gloval process id, ie. if you have two machines (nodes) with 4 GPUs each, you will have 8 RANKs 
    ddp_local_rank = int(os.environ["LOCAL_RANK"]) # LOCAL_RANK is an id given to this specific process, ie. if you have two machines with 4 GPUs each, you will have 4 LOCAL_RANKs
    ddp_world_size = int(os.environ["WORLD_SIZE"]) # WORLD_SIZE is the  number of GPUs running the DDP
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # master process is responsible for logs, checkpoints, etc.
    if master_process:
        print(f"DDP: rank={ddp_rank}, local_rank={ddp_local_rank}, world_size={ddp_world_size}")
else:
    # run without DDP
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # infer device type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device("mps")
    print(f"DDP: disabled")
    print(f"Device: {device}")

# torch is sometimes really not fun to work with if the device is not raw cuda, ie cuda:2, cuda:3, etc.
# even though there is no difference, :x is only a mapping to the physical device
device_type = "cuda" if "cuda" in device else "cpu"

# reproducibility
seed = 1337
torch.manual_seed(seed)
if device == "cuda":
    torch.cuda.manual_seed(seed)
elif device == "mps":
    torch.mps.manual_seed(seed)
if master_process:
    print("Torch seed is", seed)


# gradient accumulation
B = 64  # this controls the accumulating batch size
T = 1024 # this does not, if you're short in token length this is not a way to do it
total_batch_size = 2**19 # 2**19=~0.5M tokens but a power of 2 as well
assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size needs to be divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


# precision of matmuls FP32 -> TF32
torch.set_float32_matmul_precision("high")


# dataloader
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, verbose=master_process, inputs="fineweb", split="train")
val_loader   = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, verbose=master_process, inputs="fineweb", split="val") 

# init model
model = GPT(GPTConfig(vocab_size=50304)) # random model initialization, will still produce some readable sentence parts due to tokenizer construction
model.to(device)
model = torch.compile(model) #if device.type == "cuda" else model # cpu compile is stuck on MBP
if ddp:
    DDP(model, device_ids=[ddp_local_rank]) # hmm local rank instead of rank, interesting.....
raw_model = model #if not ddp else model.module # DDP wraps the model in a module, so we have to unwrap it, for some reason it worked for me without this
if master_process: 
    print("Model initialized successfully!")


# create optimizer
max_lr = 6e-4
min_lr = max_lr * 0.1
max_steps = 2400
warmup_steps = 50
def get_lr(step):
    # linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # constat min lr for when cosine decay ends
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
# apparently AdamW is bugfixed Adam according to Andrej
optimizer = model.configure_optimizers(weight_decay=0.01, lr=6e-4, device_type=device_type, verbose=master_process)


# local logging
eval_resolution  = 1000
model_dir        = "models"
log_dir          = "logs"
run_dir          = datetime.now(pytz.timezone("Europe/Warsaw")).strftime("%Y-%m-%d_%H-%M-%S")
if master_process: os.makedirs(os.path.join(log_dir, run_dir), exist_ok=True)
if master_process: os.makedirs(os.path.join(model_dir, run_dir), exist_ok=True)
train_file       = os.path.join(log_dir, run_dir, "train_log.txt")
val_file         = os.path.join(log_dir, run_dir, "val_log.txt")
generations_file = os.path.join(log_dir, run_dir, "generations_log.txt")
run_model_dir    = os.path.join(model_dir, run_dir)

# wandb logging
if master_process:
    wandb_logging = wandb.login()
    if wandb_logging:
        generations_table = []
        wandb.init(
            project="gpt2-wiki-pl", name=run_dir, 
            config={
                "B": B, "T": T, "total_batch_size": total_batch_size, 
                "grad_accum_steps": grad_accum_steps, "max_lr": max_lr, 
                "min_lr": min_lr, "max_steps": max_steps, "warmup_steps": warmup_steps, 
                "device": device, "ddp": ddp, "ddp_rank": ddp_rank, 
                "ddp_local_rank": ddp_local_rank, "ddp_world_size": ddp_world_size, "seed": seed,
                "eval_resolution": eval_resolution})

# huggingface upload
enable_hf = True
if master_process:
    if enable_hf:
        from huggingface_hub import HfApi
        api = HfApi()

# validation ds
if master_process:
    ds = get_belebele("pol_Latn")


# training loop
start_total = datetime.now()
metrics = dict(loss=[], tokens_per_sec=[], batch_time=[])
for step in range(max_steps):
    last_step = (step == max_steps - 1)

    # validation
    if step % eval_resolution == 0 or last_step:
        model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for val_step in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation loss: {val_loss_accum:.6f}")
            with open(val_file, "a") as file:
                file.write(f"{step}: {val_loss_accum:.4f}\n")
            if wandb_logging:
                wandb.log({"step": step, "val_loss": val_loss_accum})

    # sanity check sampling
    if step % eval_resolution == 0 and master_process:
        model.eval()
        # test_sentence = "The meaning of life is"
        test_sentence = "Cześć, jestem modelem językowym, "
        num_return_sequences = 4
        max_length = 32
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        generations = model.generate(
            test_sentence, 
            max_length=max_length, 
            num_return_sequences=num_return_sequences, 
            generator=sample_rng, printer=False, device=device)
        with open(generations_file, "a") as file:
            file.write(f"Step: {step}\n")
            for i, generation in enumerate(generations):
                file.write(f"{i}: ```{generation}```\n")
            file.write("\n")
        if wandb_logging:
            generations_table.append(generations)
            eval_generations_table = wandb.Table(
                columns=["generation_1", "generation_2", "generation_3", "generation_4"], 
                data=generations_table)
            wandb.log({"generations": eval_generations_table})

    if step % eval_resolution == 0 and master_process:
        model.eval()
        with torch.no_grad():
            acc = evaluate(
                model, 
                tiktoken.get_encoding("gpt2"),
                "cuda:0",
                ds,
                return_acc=True,
                no_dot_notation=True
            )
        wandb.log({"belebele": acc})

        

    # training
    model.train()
    start = datetime.now()
    start_rank = datetime.now()
    optimizer.zero_grad()
    loss_accum = 0.0 # only metric
    for micro_step in range(grad_accum_steps):
        if ddp:
            # boolean telling backward() to sync gradients across all ranks if true, we only want it to happen
            # on the last micro_step of the grad_accum_steps
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if "cuda" in device:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        # we need a mean of all grad_accum_steps batch elements, which is
        # grad_acum_steps * batch_size, by defualt the loss will be divided
        # by batch_size due to reduce mode "mean", so if we accumulate gradients
        # we still need to divide each of its grad_accums_steps by the number of
        # grad_accum_steps
        loss = loss / grad_accum_steps
        loss_accum += loss.detach() # only a printing metric!
        loss.backward() # it does += automatically, which is why we normally have to zero_grad
    if ddp:
        # average loss across all ranks, god I love this
        # if we didnt do this, we would only get master process loss
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        batch_time_rank = torch.tensor((datetime.now() - start_rank).total_seconds()).to(device)
        if master_process:
            batch_time_per_rank = [torch.zeros_like(batch_time_rank) for _ in range(dist.get_world_size())]
        else:
            batch_time_per_rank = None
        dist.gather(batch_time_rank, batch_time_per_rank, dst=0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    # torch.mps.synchronize() # useful for per epoch timings, only lets cpu continue after gpu finishes work
    if device_type == "cuda": torch.cuda.synchronize()
    end = datetime.now()
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (end-start).total_seconds()
    loss = loss.item()
    batch_time = end-start
    metrics["loss"].append(loss_accum), metrics["tokens_per_sec"].append(tokens_per_sec), metrics["batch_time"].append(batch_time)
    if master_process: 
        log_string = f"{step}, {loss_accum:.4f}, {norm:.4f}, {lr:.4e}, {batch_time}, {tokens_per_sec:.2f}, {batch_time_per_rank}"
        print(f"Step: {step}, Loss: {loss_accum:.6f}, Norm: {norm:.4f}, lr: {lr:.4e}, Batch time: {batch_time}, Tokens/sec: {tokens_per_sec:.2f}")
        [print(f"   Rank {i}: {batch_time:.4f}") for i, batch_time in enumerate(batch_time_per_rank)]
        if (step % eval_resolution == 0 or last_step) and step != 0:
            model_path = os.path.join(run_model_dir, f"model_{step}.pth")
            torch.save(raw_model.state_dict(), model_path)
            print(f"Model saved at step {step}!")
            if enable_hf:
                api.upload_file(
                    path_or_fileobj=model_path,
                    path_in_repo=f"model_{step}.pth",
                    repo_id="laz4rz/GPT-2-wiki-pl",
                    repo_type="model",
                )
        with open(train_file, "a") as file:
            file.write(f"{log_string}\n")
        if wandb_logging:
            wandb.log({
                "step": step, "loss": loss_accum, "norm": norm, "lr": lr, 
                "batch_time": batch_time.total_seconds(), "tokens_per_sec": tokens_per_sec,
                } | dict([(f"rank_{i}_batch_time", batch_time_per_rank[i]) for i in range(len(batch_time_per_rank))]))


end_total = datetime.now()

if master_process:
    mean_batch_time = sum(map(lambda x: x.total_seconds(), metrics["batch_time"])) / len(metrics["batch_time"])
    mean_tokens_per_sec = sum(metrics["tokens_per_sec"]) / len(metrics["tokens_per_sec"])
    print(f"Runtime: {(end_total-start_total)}\nDevice: {device}\nMean Batch time: {mean_batch_time:.2f}s\nMean tokens/sec: {mean_tokens_per_sec:.2f}")


# destroy process group
if ddp:
    destroy_process_group()
    print("Process group destroyed:", ddp_local_rank)


# save model
if master_process:
    torch.save(raw_model.state_dict(), f"model_{step}.pth")
    print("Model saved successfully!")

