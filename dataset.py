import os
import torch
import torchtext

from config import config
from torchtext.data import get_tokenizer
from torch.utils.data import Dataset


def get_qa_set():
    if os.path.exists('./dat/qa_dat.pt'):
        return torch.load('./dat/qa_dat.pt')

    dat = open('./dat/Shakespeare_preprocessed.txt')
    lines = []
    qa_dat = []
    for paragraph in dat:
        if paragraph != '\n':
            lines.append(paragraph)
    half_lines_len = len(lines) - 1
    for i in range(half_lines_len):
        qa_dat.append(('<sos> ' + str(lines[i]), str(lines[i + 1]) + ' <eos>'))
    torch.save(qa_dat, './dat/qa_dat.pt')
    return qa_dat


def get_vocab(n):
    if os.path.exists('./model/vocab.pt'):
        return torch.load('./model/vocab.pt')
    tokenizer = get_tokenizer('basic_english')
    qa_set = get_qa_set()
    all_tokens = dict()
    for q, a in qa_set:
        tokens = tokenizer(q + a)
        for token in tokens:
            all_tokens[token] = all_tokens.get(token, 0) + 1
    sorted_tokens = sorted(all_tokens.items(), key=lambda x: x[1], reverse=True)
    most_frequent_words = dict(sorted_tokens[:n - 2])
    vocab = torchtext.vocab.vocab(
        most_frequent_words,
        specials=['<unk>', '<sos>', '<eos>', '<pad>']
    )
    torch.save(vocab, './model/vocab.pt')
    return vocab


def clip(val_list):
    new_val_list = [config.pad_id] * config.max_len
    pad_len = min(len(val_list), config.max_len)
    new_val_list[:pad_len] = val_list[:pad_len]
    return new_val_list


def collate_fn(batch):
    input_ids, output_ids = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    output_ids = torch.stack(output_ids, dim=0)
    return input_ids.to(config.device), output_ids.to(config.device)


class QADataset(Dataset):
    def __init__(self):
        self.data = get_qa_set()
        self.stoi = get_vocab(config.vocab_size).vocab.get_stoi()
        self.qa_index = self.get_index()

    def get_index(self):
        qa_index = list()
        for q, a in self.data:
            tokens = q + a
            tokens = self.get_token_ids(tokens)
            qa_index.append((clip(tokens[:-1]), clip(tokens[1:])))
        return qa_index

    def get_token_ids(self, tokens):
        tokenizer = get_tokenizer('basic_english')
        tokens = tokenizer(tokens)
        tokens = [self.stoi.get(token, config.unk_id) for token in tokens]
        return tokens

    def __len__(self):
        return len(self.qa_index)

    def __getitem__(self, idx):
        dec_input, dec_output = self.qa_index[idx]
        dec_input = torch.tensor(dec_input, dtype=torch.long, device=config.device)
        dec_output = torch.tensor(dec_output, dtype=torch.long, device=config.device)
        return dec_input, dec_output


if __name__ == '__main__':
    qa_set = get_qa_set()
    vocab = get_vocab(config.vocab_size)

