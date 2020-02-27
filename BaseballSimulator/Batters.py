import torch
import csv
import statistics

def load_data(file="temp_data.csv"):
    training_set = []

    with open(file, newline='') as csv_data:
        reader = csv.reader(csv_data)
        training_set = list(reader)[1:]

    for row in training_set:
        for i, num in enumerate(row):
            row[i] = float(num)

    data_size = len(training_set)
    init = [training_set[i][:3] + training_set[i+1][:3] for i in range(data_size - 2)]
    final = [training_set[i+2][:3] for i in range(data_size - 2)]

    init = torch.FloatTensor(init)
    final = torch.FloatTensor(final)

    training_set = []
    training_set.append(init)
    training_set.append(final)
    return training_set


def train(fitter, init_pos, final_pos, epochs=5000, learning_rate=0.01):
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(fitter.parameters(),lr=learning_rate)

    print('INITIAL PARAMETERS')
    for p in fitter.parameters():
        print(p)

    for epoch in range(epochs):
        pred = fitter(init_pos)
        loss = loss_func(pred, final_pos)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print('\nFINAL PARAMETERS')
    for p in fitter.parameters():
        print(p)


def test(fitter, init_pos, final_pos):
    accuracy = []
    for i in range(len(init_pos)):
        pred = fitter(init_pos[i])
        ratio = (pred / final_pos[i])
        accuracy_temp = []
        for r in ratio:
            accuracy_temp.append(r.item())

        accuracy.append(statistics.mean(accuracy_temp))
    
    avg = statistics.mean(accuracy)
    err = abs(avg - 1) * 100
    return err



class HitModels:

    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()

    class LinearFit(Base):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Linear(in_features=6, out_features=3)

        def forward(self, pos):
            return self.f(pos)

    class TwoLayerNet(Base):
        def __init__(self, mid_layer_size):
            super().__init__()
            self.l1 = torch.nn.Linear(in_features=6, out_features=mid_layer_size)
            self.l2 = torch.nn.Linear(in_features=mid_layer_size, out_features=3)

        def forward(self, pos):
            prediction = self.l2(torch.relu(self.l1(pos)))
            return prediction

data = load_data()
testing_data = load_data() # ideally, this dataset would be a different one

print("~~~~~Linear~~~~~")
fit1 = HitModels.LinearFit()
train(fit1, data[0], data[1], learning_rate=0.001)
err = test(fit1, testing_data[0], testing_data[1])
print('final error = '+str(err)+" %")

print("\n~~~~~NN~~~~~")
fit2 = HitModels.TwoLayerNet(20)
train(fit2, data[0], data[1], epochs=5000, learning_rate=0.001)
err = test(fit2, testing_data[0], testing_data[1])
print('final error = '+str(err)+" %")
