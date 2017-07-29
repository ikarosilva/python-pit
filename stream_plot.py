import sys
import re
import numpy as np
import matplotlib.pyplot as plt


def main():
    if sys.stdin.isatty():
        return 0
    #regex = re.compile('time=(\d+.\d+)')
    regex = re.compile('(\d+.\d+)')
    data = []
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_yscale('linear') # or log
    ax.grid(True)

    l1, = ax.plot(data)
    fig.show()

    while True:
        #get ping, if no string: stream has ended
        line = sys.stdin.readline()
        if line == '':
            break
        #conversion: 64 bytes from 127.0.0.1: icmp_seq=0 ttl=45 time=100.873 ms --> 100.873
        match = regex.findall(line)
        number = 0.
        
        if len(match) > 1:
            raise ValueError()
        if len(match) == 1:
            number = float(match[0])
            #except ValueError as e:
                #print e
        #add number to array, plot the data
        print "line =%s "%number
        data.append(number)
        l1.set_xdata(range(len(data)))
        l1.set_ydata(data)

        ax.relim()
        ax.autoscale()

        plt.draw()
        fig.canvas.flush_events()
    return 0

if __name__ == '__main__':
    sys.exit(main())
