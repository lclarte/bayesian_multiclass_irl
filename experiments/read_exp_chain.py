import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def main(input_file):
    x = np.load(input_file)

    plt.scatter(x[:, 0], x[:, 1])
    plt.title('File = ' + input_file + ': true means are [1 0] and [0 100]')
    plt.show()

if __name__ == "__main__":
    args = sys.argv
    try:
        input_file = args[1]
        main(input_file)
    except Exception as e:
        print(e)
        print('Input file not provided')
    