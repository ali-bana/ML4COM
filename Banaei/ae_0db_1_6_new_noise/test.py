#%%
import matplotlib.pyplot as plt


def schedule(epoch, lr):
    if epoch < 640:
        lr = 0.001
    elif epoch < 2700:
        lr = 0.0001
    else:
        lr = 0.00001
    return lr


x = [_ for _ in range(3000)]
y = [schedule(_, 0) for _ in x]

plt.plot(x[2500:], y[2500:])
plt.show()
