import matplotlib.pyplot as plt
import csv
import sys

if __name__ == "__main__":
    cvsfile_name = sys.argv[1]
    x=[]
    y=[]

    with open(cvsfile_name, 'r') as csvfile:
        plots= csv.reader(csvfile, delimiter=',')
        next(plots, None)  # skip the headers

        i = 0
        for row in plots:
            x.append(i)
            y.append(float(row[0]))
            i += 1


    plt.plot(x, y, label='NLL')

    plt.title('NLL')

    plt.xlabel('Epoch')
    plt.ylabel('NLL')

    plt.savefig('results/NLL.png')
    plt.close()

