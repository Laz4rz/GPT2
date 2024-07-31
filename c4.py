import os
import multiprocessing as mp
import numpy as np
import tiktoken
import gzip
import json 
import random
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

def open_json_gz(file_path):
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# path = "c4/multilingual/c4-pl.tfrecord-00511-of-01024.json.gz"
# data = open_json_gz(path)

# ------------------------------------------
# choose where:
# dataset_path - c4 files (ie. c4-pl.tfrecord-00750-of-01024.json.gz) are located
# local_dir    - where to save shards
# subdataset   - will look for this part in filenames, used to filter specific langauge files
# shard_size   - tokens per shard file
# training_... - how many tokens to shard total
# approx_ch... - approximation for how many chars = 1 token, used for cache load
dataset_path = "c4/multilingual"
local_dir = "c4"
subdataset = "c4-pl"
shard_size = int(1e8)
training_tokens_to_shard = 1_100_000_000
total_tokens_to_shard = training_tokens_to_shard + 100_000_000 # + validation tokens
training_file_limit = training_tokens_to_shard // shard_size
approx_char_token = 3
random_seed = 42
random.seed(random_seed)

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# load c4 from disk with ~= expected number of tokens
# we need to overload ram a little so that we don't have to
# load on demand later
files = list(os.walk(dataset_path))[0][2]
files = [f for f in files if "c4-pl" in f]
files_shuffle = random.shuffle(files)
print(files[:10])
approx_tokens = 0
fw = []
for f in files:
    f_data = open_json_gz(os.path.join(dataset_path, f))
    for line in f_data:
        approx_tokens += len(line["text"]) // approx_char_token
    fw += f_data
    print(f"Added file {f}, approx tokens: {approx_tokens}/{training_tokens_to_shard} ({approx_tokens/training_tokens_to_shard*100:.2f}%)")
    if approx_tokens > training_tokens_to_shard:
        break


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
            filename = os.path.join(DATA_CACHE_DIR, f"c4_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar.close()
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
        filename = os.path.join(DATA_CACHE_DIR, f"c4_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
        if split == "val":
            total_tokens_validation += len(all_tokens_np[:token_count])
        else:
            total_tokens_training += len(all_tokens_np[:token_count])

print("Done sharding!")
print("Training tokens:", total_tokens_training)
print("Validation tokens:", total_tokens_validation)
