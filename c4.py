import gzip
import json

def open_json_gz(file_path):
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

path = "c4/multilingual/c4-pl.tfrecord-00511-of-01024.json.gz"
data = open_json_gz(path)


