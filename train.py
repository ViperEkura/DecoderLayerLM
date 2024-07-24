import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from config import config
from dataset import QADataset, collate_fn
from model import DecoderTransformerLM


qa_dataset = QADataset()
dataloader = DataLoader(qa_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
model = DecoderTransformerLM().to(config.device)
criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

if __name__ == '__main__':
    dataloader_len = len(dataloader)
    interval = dataloader_len // 10
    start_time = time.time()
    loss_list = []

    for epoch in range(1, config.num_epochs + 1):
        sum_loss = 0
        for idx, dat in enumerate(dataloader):
            optimizer.zero_grad()
            input_ids, target_ids = dat
            pred = model(input_ids)
            loss = criterion(pred.view(-1, config.vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

            if (idx + 1) % interval == 0:
                avg_loss = sum_loss / interval
                diff_time = time.time() - start_time
                print(f"epoch {epoch} \t{idx / dataloader_len * 100:.2f}% \tloss {avg_loss:.2f}\t using {diff_time:.2f}s")
                loss_list.append(avg_loss)
                sum_loss = 0

    plt.figure()
    plt.ylim(0, 8)
    plt.ylabel('avg loss')
    plt.xlabel('iter')
    plt.plot(loss_list)
    plt.savefig('dat/loss.png')
    torch.save(model.state_dict(), 'model/model.pt')
