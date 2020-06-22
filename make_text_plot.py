import matplotlib.pyplot as plt
import csv
import sys

if __name__ == "__main__":
    cvsfile_name = sys.argv[1]
    title = sys.argv[2]
    legend_position = sys.argv[3]
    rolling = int(sys.argv[4])

    x=[]
    y1=[]
    y2=[]

    # Open file
    with open(cvsfile_name, 'r') as csvfile:
        plots= csv.reader(csvfile, delimiter=',')
        next(plots, None)  # skip the headers

        # Read each line
        i = 0
        for row in plots:
            x.append(i)
            y1.append(float(row[0]))
            y2.append(float(row[1]))
            i += 1

    # Calculate moving average 
    xx=[]
    yy1=[]
    yy2=[]

    i = 0
    for row in range(len(x)-rolling+1):
        xx.append(i)
        s1 = 0.0
        s2 = 0.0
        for j in range(rolling):
            s1 += y1[row+j]
            s2 += y2[row+j]

        yy1.append(s1)
        yy2.append(s2)
        i += 1

    plt.plot(xx, yy1, label='Pooling CNN')
    plt.plot(xx, yy2, label='Pooling CNN (FL γ=1)')

    plt.title(title)
    plt.legend(loc=legend_position)

    plt.xlabel('Epoch')
    plt.ylabel(title)

    plt.savefig('results/' + title + '.png')
    plt.close()

