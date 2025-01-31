import torch
import text_init


class SimpleNN(torch.nn.Module):
    """Initialization AI architecture
    """
    def __init__(self, in_tensor_lenth, out_tensor_lenth):
        super(SimpleNN, self).__init__()

        self.embeding = torch.nn.Embedding(1024, 512)
        # self.pos_enc = torch.nn.Parameter(torch.zeros(1, 1024, 100))

        self.transformer = torch.nn.Transformer()

        self.out_layer = torch.nn.Linear(in_tensor_lenth, out_tensor_lenth)
        self.out_func = torch.nn.Softmax()

    def forward(self, src, tgt):
        src = self.embeding(src)
        tgt = self.embeding(tgt)

        result = self.transformer(src, tgt)
        result = self.out_layer(result)
        result = self.out_func(result)

        return result


AImodel = SimpleNN(512, 1)
optimizer = torch.optim.Adam(AImodel.parameters(), lr=0.01)
loss = torch.nn.CrossEntropyLoss()
LEARNING_RATE = 1

for _ in range(LEARNING_RATE):
    for batch in text_init.learn_text:
        src = torch.tensor(batch[: -1], dtype=torch.float32)
        tgt = torch.tensor(batch[1:], dtype=torch.float32)

        optimizer.zero_grad()
        result = AImodel.forward(src, tgt).reshape(99)

        print(src.size(), tgt.size(), result.size(), tgt, result)

        val_loss = loss(result, tgt)

        val_loss.backward()
        optimizer.step()
