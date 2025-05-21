import torch
import AI_init
import text_init as learn_t
# import matplotlib.pyplot as plt


source = learn_t.src[: -1]
AI_init.transformer.train()
loses = []

for epoch in range(1):
    for _, batch in enumerate(source):
        src = torch.tensor(source[batch][: -1], dtype=int)
        tgt = torch.tensor(source[batch][1:], dtype=int)

        AI_init.optimizer.zero_grad()
        output = AI_init.transformer(src=src,
                                     tgt=tgt)

        print(torch.argmax(output, dim=-1))

        loss = AI_init.loss(output.view(-1, len(learn_t.learn_dict)),
                            tgt.view(-1))
        loss.backward()
        AI_init.optimizer.step()

    loses.append(loss.detach().numpy())

    print(epoch, loss, src.size(), tgt.size(), output.size())
