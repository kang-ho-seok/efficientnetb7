# efficientnetb7
cifar10을 이용하여 커스텀 efficientnetb7 학습

optimizer : RMSprop, lr=0.001, alpha=0.9 ,weight_decay=1e-4

scheduler = StepLR, 2epoch마다 0.97감소

loss function : CrossEntropyLoss

batch size : 64

epoch : 100

결과 : test acc 90%
