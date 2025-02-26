import torch
import AI_init
import text_init as learn_t

LEARN_COEF: int = 1
output_data = []
source = learn_t.src
AI_init.transformer.train()

for epoch in range(100):
    for batch in range(learn_t.TEXT_LENTH):
        tgt = source[batch][1:]
        src = source[batch][: -1]

        AI_init.optimizer.zero_grad()
        output = AI_init.transformer(torch.tensor(src[0], dtype=int),
                                     torch.tensor(tgt[1], dtype=int))
        loss = AI_init.loss(output,
                            torch.tensor(src, dtype=torch.float32))
        loss.backward()
        AI_init.optimizer.step()

    print(batch, loss, src, tgt, output)

if __name__ == "__main__":
    for i in output_data:
        print(i)
