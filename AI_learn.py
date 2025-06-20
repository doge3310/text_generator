import torch
import AI_init
import text_init as learn_t


source = learn_t.src[: -1]
AI_init.transformer.train()

for epoch in range(7):
    for batch, _ in enumerate(source):
        src = torch.tensor(source[batch][: -1], dtype=int)
        tgt = torch.tensor(source[batch][1:], dtype=int)

        AI_init.optimizer.zero_grad()
        output = AI_init.transformer(src=src,
                                     tgt=tgt)

        loss = AI_init.loss(output.view(-1, len(learn_t.learn_dict)),
                            tgt.view(-1))

        loss.backward()
        AI_init.optimizer.step()

    print(epoch, loss, src.size(), tgt.size(), output.size())
    print([f"{learn_t.learn_dict[i]} {i.item()}"
           for i in torch.argmax(output, dim=-1)[0]])

if __name__ == "__main__":
    torch.save(AI_init.transformer.state_dict(), "parameters.pkl")
    torch.save(AI_init.transformer, "architecture.pkl")
