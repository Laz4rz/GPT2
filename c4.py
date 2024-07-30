import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

def open_json_gz(file_path):
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

path = "c4/multilingual/c4-pl.tfrecord-00511-of-01024.json.gz"
data = open_json_gz(path)

# ------------------------------------------
dataset_name = "chrisociepa/wikipedia-pl-20230401"
local_dir = "c4"
shard_size = int(1e8)
training_tokens_to_shard = 1_100_000_000
training_file_limit = training_tokens_to_shard // shard_size

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# load c4 from disk with ~= expected number of tokens
fw = load_dataset(dataset_name, split="train")

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
total_tokens_training = 0
total_tokens_validation = 0
training_file_count = 0
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"wikipedia_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder
            # track the number of tokens
            if split == "val":
                total_tokens_validation += len(all_tokens_np)
            else:
                total_tokens_training += len(all_tokens_np)
                training_file_count += 1
                if training_file_count == training_file_limit:
                    break

    # write any remaining tokens as the last shard
    if token_count != 0 and training_file_count < training_file_limit:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"wikipedia_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
        if split == "val":
            total_tokens_validation += len(all_tokens_np[:token_count])
        else:
            total_tokens_training += len(all_tokens_np[:token_count])
    
progress_bar.close()
print("Done sharding!")
print("Training tokens:", total_tokens_training)
print("Validation tokens:", total_tokens_validation)
