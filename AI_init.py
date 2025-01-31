import torch
import text_init as rlt


class SimpleNN(torch.nn.Module):
    """Initialization AI architecture
    """
    def __init__(self, number_neuron, start_neyron):
        super(SimpleNN, self).__init__()

        self.fc1 = torch.nn.Linear(start_neyron, number_neuron)
        self.fa1 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(number_neuron, number_neuron)
        self.do1 = torch.nn.Dropout(p=0.5)
        self.fa2 = torch.nn.ReLU()

        self.fc3 = torch.nn.Linear(number_neuron, number_neuron)
        self.do2 = torch.nn.Dropout(p=0.3)
        self.fa3 = torch.nn.ReLU()

        self.fc4 = torch.nn.Linear(number_neuron, number_neuron)
        self.do3 = torch.nn.Dropout(p=0.5)
        self.fa4 = torch.nn.Sigmoid()

        self.fc5 = torch.nn.Linear(number_neuron, number_neuron)
        self.do4 = torch.nn.Dropout(p=0.3)
        self.fa5 = torch.nn.Sigmoid()

        self.fc6 = torch.nn.Linear(number_neuron, number_neuron)
        self.do5 = torch.nn.Dropout(p=0.7)
        self.fa6 = torch.nn.Sigmoid()

        self.fc7 = torch.nn.Linear(number_neuron, number_neuron)
        self.do6 = torch.nn.Dropout(p=0.3)
        self.fa7 = torch.nn.ReLU()

        self.fc8 = torch.nn.Linear(number_neuron, number_neuron)
        self.do7 = torch.nn.Dropout(p=0.4)
        self.fa8 = torch.nn.ReLU()

        self.fc9 = torch.nn.Linear(number_neuron, 1)

    def forward(self, x):
        """initialization back propagation
        """
        x = self.fc1(x)
        x = self.fa1(x)

        x = self.fc2(x)
        x = self.do1(x)
        x = self.fa2(x)

        x = self.fc3(x)
        x = self.do2(x)
        x = self.fa3(x)

        x = self.fc4(x)
        x = self.do3(x)
        x = self.fa4(x)

        x = self.fc5(x)
        x = self.do4(x)
        x = self.fa5(x)

        x = self.fc6(x)
        x = self.do5(x)
        x = self.fa6(x)

        x = self.fc7(x)
        x = self.do6(x)
        x = self.fa7(x)

        x = self.fc8(x)
        x = self.do7(x)
        x = self.fa8(x)

        x = self.fc9(x)

        return x


NN = SimpleNN(2, rlt.TEXT_LENTH)
optimizer = torch.optim.Adam(NN.parameters(), lr=0.01)
loss = torch.nn.MSELoss()
LEARN_COEF: int = 3
output_data = []

for index in range(len(rlt.learn_text) * LEARN_COEF):
    text_index = int(index / LEARN_COEF)

    for ind in range(len(rlt.learn_text[text_index]) * LEARN_COEF):
        word_index = int(ind / LEARN_COEF)

        x = rlt.learn_text[text_index][: word_index]
        x = x + [0] * (rlt.TEXT_LENTH - len(x))
        y = rlt.learn_text[text_index][word_index]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor([y], dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = NN.forward(x)

        val_loss = loss(y_pred, y)
        output_data.append([val_loss, x, y, y_pred])

        val_loss.backward()
        optimizer.step()

if __name__ == "__main__":
    for i in output_data:
        print(i)

    len(rlt.learning_text)
