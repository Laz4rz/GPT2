import math
import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import tiktoken
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# does 2 passes but is susceptible to overflow
class softmax(nn.Module):
    """Only handles flat input"""
    def forward(self, x, exp_module=math):
        d = 0
        for i in range(len(x)):
            d += exp_module.exp(x[i])
        y = []
        for i in range(len(x)):
            y.append(exp_module.exp(x[i]) / d)
        return torch.tensor(y)

# is not susceptible to overflow, but does 3 passes (1 for max_x)
class safe_softmax(nn.Module):
    """Only handles flat input"""
    def forward(self, x, exp_module=math):
        m = float("-inf")
        for i in range(len(x)):
            if x[i] > m:
                m = x[i]
        d = 0
        for i in range(len(x)):
            d += exp_module.exp(x[i]-m)
        y = []
        for i in range(len(x)):
            y.append(exp_module.exp(x[i]-m) / d)
        return torch.tensor(y)

# is not susceptible to overflow, by using additional computation
# but reducing the number of data transfers, overall reducing
# the runtime
class online_safe_softmax(nn.Module):
    """Only handles flat input"""
    def forward(self, x, exp_module=math):
        m0 = float("-inf")
        d = 0 
        for i in range(len(x)):
            m1 = m0
            if x[i] > m0:
                m1 = m0
                m0 = x[i]
            d = d * exp_module.exp(m1 - m0) + exp_module.exp(x[i] - m0)
        y = []
        for i in range(len(x)):
            y.append(exp_module.exp(x[i] - m0) / d)
        return torch.tensor(y)

# online softmax again, but noob version using list appends
class online_safe_softmax_list(nn.Module):
    """Only handles flat input"""
    def forward(self, x, exp_module=math):
        m = [float("-inf")]
        d = 0 
        for i in range(len(x)):
            if x[i] > m[-1]:
                m.append(x[i])
            else:
                m.append(m[-1])
            d = d * exp_module.exp(m[-2] - m[-1]) + exp_module.exp(x[i] - m[-1])
        y = []
        for i in range(len(x)):
            y.append(exp_module.exp(x[i] - m[-1]) / d)
        return torch.tensor(y)

class TanhGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

# sanity check if I understand cross entropy docs, weight can be used for unbalanced classes
def cross_entropy(t, target, weights: dict | None = None):
    if weights is not None:
        return -1 * torch.log(torch.exp(t[target]) / torch.exp(t).sum()) * weights[target]
    return -1 * torch.log(torch.exp(t[target]) / torch.exp(t).sum())


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # why im not sure at first glance
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # batched q, k, v projections for (all heads) this im not sure why for all 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANO_GPT_SCALE_INIT = 1
        # regularization (why??? where??? what???)
        # more of a mask than bias, but follows HF/openai naming (also not sure why)
        # buffers are non-learnable parameters
        # when using flash attention, we don't need this buffer -- we no longer create the large (T, T) matrix that needs masking
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                      .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch, sequence length, embedding dimension (n_embd)
        # calculate q, k, v for all heads in batch and move head forward (to be the batch???)
        # nh is the number of heads, hs is the head size, C (number of channels) = nh * hs
        # in this GPT-2 (124M) model, nh = 12, hs = 64, so nh * hs = C = 768 channels in transformer
        qkv = self.c_attn(x) # TODO: bias of self.c_attn.bias does not match
        q, k, v = qkv.split(self.n_embd, dim=2) # chech what it does, what are the shapes
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        # attention -- materialize the large (T, T) for all queries and keys
        
        # raw attention calculatio
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        
        # flash attention calculation
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # contiguous() places the tensor in a contiguous block of memory, used after transpose, view, etc.
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANO_GPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257 # number of tokens, 50k merges BPE, 256 bytes tokens, 1 <|endoftext|>
    n_layer: int = 12       # number of layers
    n_head: int = 12        # number of heads
    n_embd: int = 768       # embedding dimension


class GPT(nn.Module):

    def __init__(self, config, lower_matmul_precision=True, return_dict=False):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # GPT uses no bias

        # weights sharing (https://arxiv.org/pdf/1608.05859)
        self.transformer.wte.weight = self.lm_head.weight
        
        # init params with specific parameters
        self.apply(self._init_weights)

        # set dtype of matmul operation to TF32 (or treat F32 as sum of two BF16, if none available fallsback to F32)
        # if I understood correctly, on MPS it will use the BFloat16 solution, TF32 is not supported
        if lower_matmul_precision:
            torch.set_float32_matmul_precision("high")

        print("return dict:", return_dict)
        self.return_dict = return_dict

 # parameters specific to GPT2/GPT3 papers
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # the scaled initialization of stacked residual layers
            if hasattr(module, "NANO_GPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** (-0.5) # times 2 due to two residual concats for each block made by n_layers (i guess) (yeah literally confirmed 10 secs later)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None): # why is this called idx dunno 
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence length ({T=}) is greater than the pretrained position embeddings value ({self.config.block_size=})"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # positional embeddings (T, n_emb)
        tok_emb = self.transformer.wte(idx) # token embeddings (batch size, seq length, n_emb)
        x = pos_emb + tok_emb               # superposition of positions and token embeddings
        for block in self.transformer.h:    # pass sequentially through all attention blocks
            x = block(x)
        x = self.transformer.ln_f(x)        # pass through layernorm
        logits = self.lm_head(x)                 # pass through projection from high-dim (latent?) to vocab_size
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) 
            # quite counterintuitive for me, cause id think that we only want to compare the last logits??
            # since we only generate them?? but we can probably just compare all since, the correctly matched
            # do not add to lose either way, at least if my poor mind math skills are ok
        if self.return_dict:
            return SimpleNamespace(logits=logits, loss=loss)
        return logits, loss                      # return logits, loss

    def configure_optimizers(self, weight_decay, lr, device_type, verbose=False):
        # getting all parameters that require grad
        # these are basically just elements like "transformer.h.6.attn.c_attn.bias"
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # we only want to weight decay matmuls (linear, embedding etc), not biases and norms
        # we can do that by filtering by the dimension, any thing >= 2D is weigth decayed
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weigth_decay": weight_decay},
            {"params": nondecay_params, "weigth_decay": 0.0},
        ]
        num_decay_params = sum([p.numel() for p in decay_params])
        num_nondecay_params = sum([p.numel() for p in nondecay_params])
        if verbose:
            print(f"Decayed {len(decay_params)} layers, {num_decay_params} parameters, with weight decay {weight_decay}")
            print(f"Non-decayed {len(nondecay_params)} layers, {num_nondecay_params} parameters")
        # newer version of pytorch supports a fused AdamW optimizer, we can check for it
        # and if available use it for another optimization
        # inspect.signature allows us to check what parameters are in function signature
        # neat
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if verbose:
            print(f"Using fused AdamW optimizer:", use_fused)
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    @classmethod
    def from_pretrained(cls, model_name, verbose=False, local=False, **kwargs):
        if not local:
            assert model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], "model_name should be one of ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']"
            from transformers import GPT2LMHeadModel
            print(f"Loading {model_name} model weights from transformers")

            # n_head, n_layer, n_embd are model specific
            config_args = {
                "gpt2"         : dict(n_head=12, n_layer=12, n_embd=768),  # 124M
                "gpt2-medium"  : dict(n_head=16, n_layer=16, n_embd=1024), # 350M
                "gpt2-large"   : dict(n_head=36, n_layer=20, n_embd=1280), # 774M
                "gpt2-xl"      : dict(n_head=48, n_layer=25, n_embd=1600), # 1558M
            }[model_name]

            # load architecture
            config_args["vocab_size"] = 50257 # constant for all GPT-2 models
            config_args["block_size"] = 1024  # constant for all GPT-2 models
            config  = GPTConfig(**config_args)
            model   = GPT(config, **kwargs)
            sd      = model.state_dict()
            sd_keys = sd.keys()
            sd_keys = [k for k in sd_keys if ".attn.bias" not in k] # remove attn.bias buffer, we omit buffers for some reason i dont know why, it doesnt even matter how hardy I try

            # load huggingface model weights
            hf_model = GPT2LMHeadModel.from_pretrained(model_name)
            sd_hf    = hf_model.state_dict()

            # choose, match and copy weights
            sd_keys_hf = sd_hf.keys()
            sd_keys_hf = [k for k in sd_keys_hf if ".attn.bias" not in k and "attn.masked_bias" not in k] # remove attn.bias and attn.masked_bias
            transposed = [".attn.c_attn.weight", ".attn.c_proj.weight", ".mlp.c_fc.weight", ".mlp.c_proj.weight"]
            # some of the original weights use Conv1D, but we want to load vanilla
            # which is why we need to transpose some of the weights
            assert len(sd_keys) == len(sd_keys_hf), f"mismatch in number of keys: custom {len(sd_keys)} vs hf {len(sd_keys_hf)}"
            for k in sd_keys_hf:
                if verbose: print(f"   Loading: {k}")
                if verbose: print("     from: {:15s}\n       to: {:15s}".format(str(sd_hf[k].shape), str(sd[k].shape)))
                if any(t in k for t in transposed):
                    assert sd_hf[k].shape[::-1] == sd[k].shape, f"mismatch in shape for special: {k}"
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k].t().contiguous()) # .t() works only for 2D weights, .T for any
                else:
                    assert sd_hf[k].shape == sd[k].shape, f"mismatch in shape for {k}"
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k])

            return model
        else:
            print(f"Loading GPT-2 model weights from local file: {model_name}")
            config = GPTConfig(vocab_size=50304)
            model = GPT(config, **kwargs)
            sd = model.state_dict()
            sd_keys = sd.keys()
            sd_keys = [k for k in sd_keys if ".attn.bias" not in k]
            # load local model weights
            sd_local = torch.load(model_name, map_location="cpu")
            sd_local_keys = sd_local.keys()
            sd_local_keys = [k for k in sd_local_keys if ".attn.bias" not in k]
            # naive way of handling torch.compile orig_mod
            sd_local_keys = [k.replace("_orig_mod.", "") for k in sd_local_keys]
            sd_local = {key.replace("_orig_mod.", ""): value for key, value in sd_local.items()}
            assert len(sd_keys) == len(sd_local_keys), f"mismatch in number of keys: custom {len(sd_keys)} vs local {len(sd_local_keys)}"
            for k in sd_local_keys:
                if verbose: print(f"   Loading: {k}")
                if verbose: print("     from: {:15s}\n       to: {:15s}".format(str(sd_local[k].shape), str(sd[k].shape)))
                assert sd_local[k].shape == sd[k].shape, f"mismatch in shape for {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_local[k])
        return model

    def generate(self, input, max_length=30, num_return_sequences=5, device="cpu", generator=None, printer=True):
        # encode
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(input)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        x = tokens.to(device)

        # generating loop
        if generator is None:
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            generator = torch.Generator(device=device)
            generator.manual_seed(42)
        while x.size(1) < max_length:
            with torch.no_grad():
                logits, loss = self(x)
                logits = logits[:, -1, :]
                probs  = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix      = torch.multinomial(topk_probs, 1, generator=generator)
                xcol    = torch.gather(topk_indices, -1, ix)
                x       = torch.cat((x, xcol), dim=1)
        
        # decode generated
        all_decoded = []
        for i in range(num_return_sequences):
            tokens  = x[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            all_decoded.append(decoded)
            if not printer:
                print(f"> {decoded}")
        if not printer:
            return all_decoded

# class method allows constructing a class through a class method call
# GPT.from_pretrained('gpt2') would be a way to instantiate a GPT model from a pre-trained checkpoint

if __name__ == "__main__":
    # init model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT.from_pretrained('gpt2', verbose=True, lower_matmul_precision=False)
    # model = GPT(GPTConfig()) # random model initialization, will still produce some readable sentence parts due to tokenizer construction
    print(f"Device: {device}")
    # print(model) # buffers are not visible here, to show them we need to look at model.buffers()
    print("Model loaded successfully!")

    # manual generation
    print("Manual generation:")
    model.eval() # put model in eval mode, works for layers like: Dropout, BatchNorm, etc.
    model.to(device)

    num_return_sequences = 5
    max_length = 30

    # encode
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
    x = tokens.to(device)

    # generating loop
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, loss = model(x)                   # get logits from non-grad forward pass
            logits = logits[:, -1, :]           # take last logits from each batch
            probs  = F.softmax(logits, dim=-1)  # get probabilities from logits
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)    # get top50 probs and its indices
            ix      = torch.multinomial(topk_probs, 1)   # get random idx from topk distribution (not really dist, since sum=/=1, but yknow)
            xcol    = torch.gather(topk_indices, -1, ix) # get tokens corresponding to sampled ixs
            x       = torch.cat((x, xcol), dim=1)        # concat previous tokens, with sampled ixs tokens
            
    # decode generated
    for i in range(num_return_sequences):
        tokens  = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"> {decoded}") 

    # automatic generation
    print("Automatic generation:")
    model.generate("Hello, I'm a language model,", max_length=max_length, num_return_sequences=num_return_sequences, device=device)
