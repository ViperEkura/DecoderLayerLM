import torch
import warnings


class BaseConfig:
    d_model = 256
    vocab_size = 12000
    max_len = 256
    drop_rate = 0.1
    feedforward_dim = 4 * d_model
    n_head = 4
    decoder_layer = 4
    learning_rate = 5e-4
    batch_size = 32
    num_epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unk_id, sos_id, eos_id, pad_id = 0, 1, 2, 3


warnings.filterwarnings('ignore')
config = BaseConfig()
