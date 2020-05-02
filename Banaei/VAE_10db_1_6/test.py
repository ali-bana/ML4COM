#%%
import matplotlib.pyplot as plt


def schedule(epoch, lr):
    if epoch > 8000:
        lr = 0.000001
    else:
        decay = (1 - (epoch / float(1000))) ** 4
        lr = 0.0001 * decay

    return lr


x = [_ for _ in range(1000)]
y = [schedule(_, 0) for _ in x]

plt.plot(x, y)
plt.show()
