#%%
import matplotlib.pyplot as plt


def schedule(epoch, lr):
    lr = 0.001
    if epoch > 600 and epoch < 2000:
        lr = 0.0001
    elif epoch>1000:
        lr = 0.0001 - (0.0001-0.00001) * ((epoch-2000)/3000)
    return lr


x = [_ for _ in range(5000)]
y = [schedule(_, 0) for _ in x]

plt.plot(x, y)
plt.show()
