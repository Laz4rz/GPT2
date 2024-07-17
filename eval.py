import torch
import sys
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import tiktoken
from datasets import load_dataset
from gpt2 import GPT, GPTConfig


"""
seed: 42

Llama3 8B:
 - eng_Latn: 0.2811
 - pol_Latn: 0.2600
 - 
GPT-2 124M:
 - eng_Latn: 0.2778
 - pol_Latn: 0.2500
 
"""


def get_hellaswag(split="validation"):
    ds = load_dataset("Rowan/hellaswag", split=split)

    ds = ds.map(lambda x: {"label": int(x["label"])})

    return ds


def get_belebele(split="eng_Latn"):
    ds = load_dataset("facebook/belebele", split=split)

    # hellaswag format
    ds = ds.map(lambda x: {
        "ctx": x["question"],
        "label": int(x["correct_answer_num"]) - 1,
        "endings": [x["mc_answer1"], x["mc_answer2"], x["mc_answer3"], x["mc_answer4"]]
    })

    return ds


def render_example(example, enc):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        # if model.__str__().split("(")[0] != "LlamaForCausalLM":
        #     end_tokens = enc.encode(" " + end) # note: prepending " " because GPT-2 tokenizer -- may be treated as a sentence start and get a special token
        # else:
        #     end_tokens = enc.encode(end)   
        end_tokens = enc.encode(" " + end)     
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_examples(ds):
    for example in ds:
        yield example


@torch.no_grad()
def evaluate(model, tokenizer, device, ds, return_acc=False, no_dot_notation=False):

    torch.set_float32_matmul_precision('high') # use tf32
    model.to(device)

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples(ds):
        data, tokens, mask, label = render_example(example, tokenizer)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        if no_dot_notation:
            logits = model(tokens)[0]
        else:
            logits = model(tokens).logits # size: sentences x tokens x vocab
        shift_logits = (logits[..., :-1, :]).contiguous() # size: sentences x (tokens-1) x vocab
        shift_tokens = (tokens[..., 1:]).contiguous()     # size: sentences x (tokens-1)
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)) # size: (sentences*tokens) x vocab
        flat_shift_tokens = shift_tokens.view(-1)                        # size: (sentences*tokens)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1) # size: sentences x tokens
        shift_mask = (mask[..., 1:]).contiguous() # size: sentences x (tokens-1)
        masked_shift_losses = shift_losses * shift_mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}, acc: {num_correct/num_total:.4f} -- pred: {pred}, label: {label}")

        # print examples
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")
    
    # print("model:", name)
    print(f"Final acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")
    
    if return_acc:
        return num_correct_norm/num_total


if __name__ == "__main__":
    # seeds should not matter here
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # name = "meta-llama/Meta-Llama-3-8B"
    name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    enc = tokenizer

    # eval local models
    # 1. my pretrained GPT-2 models are at https://huggingface.co/Laz4rz/GPT-2/tree/main
    # 2. you can download with snapshot_download(repo_id="Laz4rz/GPT-2", allow_patterns="*.pth") (huggingface_hub library)
    # 3. they will be place in the .cache folder, so the path will be something like ~/.cache/huggingface and the follow by what you see in the example path below
    # 4. if you want to use custom model, make sure it returns dot notation accessible logits (model(x).logits, used SimpleNamespace for that)
    # name = "/root/.cache/huggingface/hub/models--Laz4rz--GPT-2/snapshots/7b11ebd22a44a487bda59e6c29d2ae144513a268/model_19000.pth"
    # model = GPT.from_pretrained(name, local=True, return_dict=True)
    # enc = tiktoken.get_encoding("gpt2")

    # ds = get_belebele("eng_Latn")
    ds = get_hellaswag("validation")

    evaluate(model, tokenizer, "cuda", ds)
