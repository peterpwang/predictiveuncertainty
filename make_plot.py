import matplotlib.pyplot as plt
import csv
import sys

if __name__ == "__main__":
    cvsfile_name = sys.argv[1]
    title = sys.argv[2]
    legend_position = sys.argv[3]

    x=[]
    y1=[]
    y2=[]
    y3=[]
    y4=[]
    y5=[]
    y6=[]
    y7=[]

    with open(cvsfile_name, 'r') as csvfile:
        plots= csv.reader(csvfile, delimiter=',')
        next(plots, None)  # skip the headers

        i = 0
        for row in plots:
            x.append(i)
            y1.append(float(row[0]))
            y2.append(float(row[1]))
            y3.append(float(row[2]))
            y4.append(float(row[3]))
            y5.append(float(row[4]))
            y6.append(float(row[5]))
            y7.append(float(row[6]))
            i += 1


    plt.plot(x, y1, label='Resnet50')
    plt.plot(x, y2, label='Resnet50 (FL γ=1)')
    plt.plot(x, y3, label='Densenet121')
    plt.plot(x, y4, label='Densenet121 (FL γ=1)')
    plt.plot(x, y5, label='EfficientNet B0')
    plt.plot(x, y6, label='EfficientNet B2')
    plt.plot(x, y7, label='EfficientNet B7')

    plt.title(title)
    plt.legend(loc=legend_position)

    plt.xlabel('Epoch')
    plt.ylabel(title)

    plt.savefig('results/' + title + '.png')
    plt.close()

