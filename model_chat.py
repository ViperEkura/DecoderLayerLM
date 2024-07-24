import torch
import random
from torchtext.data import get_tokenizer

from config import config
from dataset import get_vocab
from model import DecoderTransformerLM


def gen_chat(model, vocab, question, k, max_len=30):
    tokenizer = get_tokenizer('basic_english')
    stoi, itos = vocab.get_stoi(), vocab.get_itos()
    model.eval()
    tokens = tokenizer('<sos>' + question)
    tokens = [stoi.get(token, config.pad_id) for token in tokens]
    start_pos = len(tokens)

    while len(tokens) - start_pos < max_len:
        input_ids = torch.tensor(tokens).view(1, -1)
        output = model(input_ids.to(config.device))
        output = torch.softmax(output, dim=-1)
        m_val, m_indices = torch.topk(output[0, -1].view(-1), k=k, dim=-1)
        k_val = m_val.tolist()
        summa = sum(k_val)
        k_val = [val / summa for val in k_val]
        k_indices = m_indices.tolist()

        target_index, lim_cnt = 0, 0
        while target_index < 4 and lim_cnt < k:
            lim_cnt += 1
            target_index = random.choices(k_indices, k_val, k=1)[0]
        if lim_cnt >= k and target_index < 4:
            break

        tokens.append(target_index)
    answer_str = [itos[index] for index in tokens[start_pos:]]
    print(' '.join(answer_str))
    print(tokens[start_pos:], 'len:', len(tokens[start_pos:]))


if __name__ == '__main__':
    model = DecoderTransformerLM()
    model.load_state_dict(torch.load("./model/model.pt"))
    model.to(config.device)
    model.eval()

    vocab = get_vocab(config.vocab_size)
    question = "  You were pretty lordings then? "
    gen_chat(model, vocab, question, 1)
