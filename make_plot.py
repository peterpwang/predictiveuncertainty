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
    y3=[]
    y4=[]
    y5=[]
    y6=[]
    y7=[]
    y8=[]

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
            y3.append(float(row[2]))
            y4.append(float(row[3]))
            y5.append(float(row[4]))
            y6.append(float(row[5]))
            y7.append(float(row[6]))
            y8.append(float(row[7]))
            i += 1

    # Calculate moving average 
    xx=[]
    yy1=[]
    yy2=[]
    yy3=[]
    yy4=[]
    yy5=[]
    yy6=[]
    yy7=[]
    yy8=[]

    i = 0
    for row in range(len(x)-rolling+1):
        xx.append(i)
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        s5 = 0.0
        s6 = 0.0
        s7 = 0.0
        s8 = 0.0
        for j in range(rolling):
            s1 += y1[row+j]
            s2 += y2[row+j]
            s3 += y3[row+j]
            s4 += y4[row+j]
            s5 += y5[row+j]
            s6 += y6[row+j]
            s7 += y7[row+j]
            s8 += y8[row+j]

        yy1.append(s1)
        yy2.append(s2)
        yy3.append(s3)
        yy4.append(s4)
        yy5.append(s5)
        yy6.append(s6)
        yy7.append(s7)
        yy8.append(s8)
        i += 1

    plt.plot(xx, yy1, label='Resnet50')
    plt.plot(xx, yy2, label='Resnet50 (FL γ=1)')
    plt.plot(xx, yy3, label='Densenet121')
    plt.plot(xx, yy4, label='Densenet121 (FL γ=1)')
    plt.plot(xx, yy5, label='EfficientNet B0')
    plt.plot(xx, yy6, label='EfficientNet B0 (FL γ=1)')
    plt.plot(xx, yy7, label='EfficientNet B7')
    plt.plot(xx, yy8, label='EfficientNet B7 (FL γ=1)')

    plt.title(title)
    plt.legend(loc=legend_position)

    plt.xlabel('Epoch')
    plt.ylabel(title)

    plt.savefig('results/' + title + '.png')
    plt.close()

