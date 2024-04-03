from ESN_with_readout import EchoStateNetworkWithObserver
from ESN import EchoStateNetworkWithoutObserver

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Загрузка MNIST датасета
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


# Параметры сети
input_size = 28 * 28  # размер входных данных (размер изображения MNIST)
reservoir_size = 100  # размер резервуарного слоя
output_size = 10  # размер выходного слоя (число классов)

# Инициализация сетей
esn_without_observer = EchoStateNetworkWithoutObserver(input_size, reservoir_size, output_size)
esn_with_observer = EchoStateNetworkWithObserver(input_size, reservoir_size, output_size)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer_without_observer = optim.Adam(esn_without_observer.parameters(), lr=0.001)
optimizer_with_observer = optim.Adam(esn_with_observer.parameters(), lr=0.001)

# Обучение сетей
epochs = 5
for epoch in range(epochs):
    running_loss_without_observer = 0.0
    running_loss_with_observer = 0.0
    for images, labels in trainloader:
        # Обучение сети без наблюдателя
        optimizer_without_observer.zero_grad()
        outputs_without_observer = esn_without_observer(images)
        loss_without_observer = criterion(outputs_without_observer, labels)
        loss_without_observer.backward()
        optimizer_without_observer.step()
        running_loss_without_observer += loss_without_observer.item()

        # Обучение сети с наблюдателем
        optimizer_with_observer.zero_grad()
        outputs_with_observer = esn_with_observer(images)
        loss_with_observer = criterion(outputs_with_observer, labels)
        loss_with_observer.backward()
        optimizer_with_observer.step()
        running_loss_with_observer += loss_with_observer.item()

    print(f"Epoch {epoch+1}, Loss without observer: {running_loss_without_observer}, Loss with observer: {running_loss_with_observer}")

print('Training finished')

# Тестирование сетей
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

correct_without_observer = 0
correct_with_observer = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        # Тестирование сети без наблюдателя
        outputs_without_observer = esn_without_observer(images)
        _, predicted_without_observer = torch.max(outputs_without_observer.data, 1)
        correct_without_observer += (predicted_without_observer == labels).sum().item()

        # Тестирование сети с наблюдателем
        outputs_with_observer = esn_with_observer(images)
        _, predicted_with_observer = torch.max(outputs_with_observer.data, 1)
        correct_with_observer += (predicted_with_observer == labels).sum().item()

        total += labels.size(0)

print(f'Accuracy of the network without observer on the 10000 test images: {100 * correct_without_observer / total}%')
print(f'Accuracy of the network with observer on the 10000 test images: {100 * correct_with_observer / total}%')



# Функция для отображения изображений и их предсказаний
def visualize_predictions(model, testloader, num_images=5):
    # Выведем изображения и предсказания
    plt.figure(figsize=(10, 4))
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            if i >= 1:  # Остановимся после первого батча изображений
                break
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for j in range(num_images):
                plt.subplot(2, num_images, i*num_images+j+1)  # Общее количество изображений = 2*num_images
                plt.imshow(images[j].squeeze(), cmap='gray')
                plt.title(f'Predicted: {predicted[j]}, True: {labels[j]}')
                plt.axis('off')
    plt.tight_layout()
    plt.show()

# Вывод нескольких изображений и предсказаний обеих моделей
visualize_predictions(esn_without_observer, testloader)
visualize_predictions(esn_with_observer, testloader)
