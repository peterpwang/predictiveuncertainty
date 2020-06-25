import matplotlib.pyplot as plt
import csv
import sys

if __name__ == "__main__":
    cvsfile_name = sys.argv[1]
    title = sys.argv[2]
    legend_position = sys.argv[3]
    rolling = int(sys.argv[4])
    y1_label = sys.argv[5];
    y2_label = sys.argv[6];
    y3_label = ''
    y4_label = ''
    y5_label = ''
    y6_label = ''
    y7_label = ''
    y8_label = ''
    if (len(sys.argv)>=7):
        y3_label = sys.argv[7];
    if (len(sys.argv)>=8):
        y4_label = sys.argv[8];
    if (len(sys.argv)>=9):
        y5_label = sys.argv[9];
    if (len(sys.argv)>=10):
        y6_label = sys.argv[10];
    if (len(sys.argv)>=11):
        y7_label = sys.argv[11];
    if (len(sys.argv)>=12):
        y8_label = sys.argv[12];

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
            if(y3_label != ''):
                y3.append(float(row[2]))
            if(y4_label != ''):
                y4.append(float(row[3]))
            if(y5_label != ''):
                y5.append(float(row[4]))
            if(y6_label != ''):
                y6.append(float(row[5]))
            if(y7_label != ''):
                y7.append(float(row[6]))
            if(y8_label != ''):
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
            if(y3_label != ''):
                s3 += y3[row+j]
            if(y4_label != ''):
                s4 += y4[row+j]
            if(y5_label != ''):
                s5 += y5[row+j]
            if(y6_label != ''):
                s6 += y6[row+j]
            if(y7_label != ''):
                s7 += y7[row+j]
            if(y8_label != ''):
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

    plt.plot(xx, yy1, label=y1_label)
    plt.plot(xx, yy2, label=y2_label)
    if (y3_label != ''):
        plt.plot(xx, yy3, label=y3_label)
    if (y4_label != ''):
        plt.plot(xx, yy4, label=y4_label)
    if (y5_label != ''):
        plt.plot(xx, yy5, label=y5_label)
    if (y6_label != ''):
        plt.plot(xx, yy6, label=y6_label)
    if (y7_label != ''):
        plt.plot(xx, yy7, label=y7_label)
    if (y8_label != ''):
        plt.plot(xx, yy8, label=y8_label)

    plt.title(title)
    plt.legend(loc=legend_position)

    plt.xlabel('Epoch')
    plt.ylabel(title)

    plt.savefig('results/' + title + '.png')
    plt.close()

