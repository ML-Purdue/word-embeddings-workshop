            #!/usr/bin/env python3

from data.skipgram import Dataset
from data.cbow import Dataset
from .arch import Word2Vec
from IPython import embed
import const
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, optimizer, loss, data):
    training_loss = []
    for epoch in range(const.EPOCHS):
        print('-' * 10)
        training_loss.append([])

        for batch in data:
            optimizer.zero_grad()

            X, y = batch
            y_pred = model(X.to(device))

            batch_loss = loss(y_pred, y.to(device))
            batch_loss.backward()

            optimizer.step()

            training_loss[-1].append(batch_loss)
        print(f'Epoch: {epoch+1}\tLoss: {sum(training_loss[-1]) / const.BATCH_SIZE}')
    print('-' * 10)


if __name__ == '__main__':
    dataset = Dataset(const.TOY_DATASET_PATH)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=True,
                                             batch_size=const.BATCH_SIZE)

    model = Word2Vec(len(dataset), const.VECTOR_DIMENSIONS).to(device)
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=const.LEARNING_RATE,
                                momentum=const.MOMENTUM)
    loss = torch.nn.CrossEntropyLoss()
    train(model, optimizer, loss, dataloader)
    torch.save(model, const.SAVE_MODEL_PATH / 'w2v.pt')

    embed()
