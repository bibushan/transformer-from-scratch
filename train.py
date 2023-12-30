import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.model import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

## gives absolute path given relative path
from pathlib import Path 



def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]



def build_tokenizer(config, ds, lang):
    # config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token=['UNK']))
        tokenizer.pre_tokenizer = Whitespace() ## we split by whitespace
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer



def get_dataset(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # buiding the tokenizer
    tokenizer_src = build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = build_tokenizer(config, ds_raw, config['lang_tgt'])

    # split dataset into 90% train and 10% test
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

