import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

class OscillogramNode:
    def __init__(self, init_data):
        self.number = ""
        for i in range(3, -1, -1):
            self.number += bin(init_data[i])[2:]
        self.number = np.uint32(int(self.number, base=2))


        self.start_time = ""
        for i in range(7, 3,-1):
            self.start_time += bin(init_data[i])[2:].zfill(8)
        self.start_time = np.uint32(int(self.start_time, base=2))


        init_data = init_data[8:]
        init_dataA = init_data[:-8]
        init_dataD = init_data[-8:]


        self.channelsA = []
        for x in range(int(len(init_dataA)/2)):
            tempValue = ""
            for i in range(1,-1,-1):
                tempValue += bin(init_dataA[i])[2:].zfill(8)
            tempValue = np.int16(int(tempValue, base=2))
            self.channelsA.append(tempValue)
            init_dataA = init_dataA[2:]


        self.channelsD = []
        for x in range(int(len(init_dataD)/2)):
            tempValue = ""
            for i in range(1,-1,-1):
                tempValue += bin(init_dataD[i])[2:].zfill(8)
            self.channelsD.append(tempValue)
            init_dataD = init_dataD[2:]


class InfoOs:
    def __init__(self, init_info):
        init_info = init_info.split('\n')
        LineChannels=init_info[1].split(',')
        self.AllChannels = LineChannels[0]
        self.channelsA = int(LineChannels[1][:-1])
        self.channelsD = int(LineChannels[2][:-1])
        self.size_line_data = 8 + (self.channelsA + math.ceil(self.channelsD/16))*2
        self.coffTransfor = []
        self.name = []
        for i in range(2, self.channelsA+2):
            temp = init_info[i].split(',')
            self.coffTransfor.append(float(temp[-2]))
            LineChannels=init_info[i].split(',')
            self.name.append(LineChannels[1])
        print(self.name)

def size_binline(init_info):
    init_info = init_info.split('\n')
    LineChannels=init_info[1].split(',')
    AllChannels = LineChannels[0]
    channelsA = int(LineChannels[1][:-1])
    channelsD = int(LineChannels[2][:-1])
    size_line_data = 8 + (channelsA + math.ceil(channelsD/16))*2
    return size_line_data

def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)


def get_bin_data_from_file(filename):
    with open(filename, "rb") as f:
        bin_data =f.read()
    return bin_data


def get_info_from_file(filename):
    with open(filename, "r") as f:
        info =f.read()
    return info


if __name__ == '__main__':
    input_data =  get_bin_data_from_file("24e61716a082f03660813bc0fd6b3f56458cac13.dat")
    input_info =  get_info_from_file("24e61716a082f03660813bc0fd6b3f56458cac13.cfg")
    #print(input_info)
    oscillogram_node_list = []
    x=[]
    constLineSize = InfoOs(input_info)
    number_of_lines = int(len(input_data)/constLineSize.size_line_data)
    channelsValue=[[] for i in range(constLineSize.channelsA)]

    for i in range(number_of_lines):
        oscillogram_node_list.append(OscillogramNode(input_data[:constLineSize.size_line_data]))
        x.append(int(oscillogram_node_list[i].start_time))
        input_data = input_data[constLineSize.size_line_data:]
        if (i<20):
            print(i ,"- Number: ", oscillogram_node_list[i].number, "; begin time: ", oscillogram_node_list[i].start_time,"; A: ", oscillogram_node_list[i].channelsA, "; D: ", oscillogram_node_list[i].channelsD, sep="")
    for n in range(constLineSize.channelsA):
        for i in range(number_of_lines):
            channelsValue[n].append(float(oscillogram_node_list[i].channelsA[n])/constLineSize.coffTransfor[n])

    # Create a dtype with the binary data format and the desired column names
    # dt = np.dtype([('a', 'i4'), ('b', 'i4'), ('c', 'i2'), ('d', 'i2'),('e', 'i2'),('f', 'i2'),('a1', 'i2'),('b1', 'i2'),('c1', 'i2'),('d1', 'i2'),('e1', 'i2'), ('f1', 'i2'), ('a2', 'i2'),('f3', 'i2'), ('f4', 'i2'),('f5', 'i2'), ('f6', 'i2')])
    # data = np.fromfile("24e61716a082f03660813bc0fd6b3f56458cac13.dat", dtype=dt)
    # df = pd.DataFrame(data)
    # Or if you want to explicitly set the column names
    # df = pd.DataFrame(data, columns=data.dtype.names)
    # print(df)
    for i in range(7,10):
        plt.plot(x, channelsValue[i], label=constLineSize.name[i])
    plt.legend(loc=2)
    plt.show()
    # plt.xlim (0, 40000)
