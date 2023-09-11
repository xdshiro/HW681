# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import my_functions.plotings as pl
import matplotlib.pyplot as plt


def H(p):
    return -p*np.log(p) - (1-p)* np.log(1-p)


def error(p):
    return p

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ticksFontSize = 20
    xyLabelFontSize = 20
    fig, ax = plt.subplots(figsize=(8, 6))
    xArray = np.linspace(0.0001, 0.5, 200)

    plt.plot(xArray, H(xArray), lw=5, color='b')
    plt.plot(xArray, error(xArray), lw=5, color='r')
    plt.xticks(fontsize=ticksFontSize)
    plt.yticks(fontsize=ticksFontSize)
    ax.set_xlabel('p', fontsize=xyLabelFontSize)
    # ax.set_ylabel(yname, fontsize=xyLabelFontSize)
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
