import torch
import AI_init
import text_init as learn_t


source = learn_t.src
AI_init.transformer.train()

for epoch in range(100):
    for batch in range(len(learn_t.src)):
        src = torch.tensor(source[batch][: -1], dtype=int)
        tgt = torch.tensor(source[batch][1:], dtype=int)

        AI_init.optimizer.zero_grad()
        output = AI_init.transformer(src=src,
                                     tgt=tgt)

        loss = AI_init.loss(output.view(-1, len(learn_t.learn_list)),
                            tgt.view(-1))
        loss.backward()
        AI_init.optimizer.step()

    print(epoch,
          loss,
          src,
          tgt,
          output.size())
