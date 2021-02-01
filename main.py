import numpy as np
import matplotlib.pyplot as plt
import math

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

        # self.channels = []
        # for x in range(int(len(init_data)/2)):
        #     tempValue = ""
        #     for i in range(2):
        #         tempValue += bin(init_data[i])[2:].zfill(8)
        #     tempValue = np.int16(int(tempValue, base=2))
        #     self.channels.append(tempValue)
        #     init_data = init_data[2:]
        #     a = np.array(self.channels)



        a = int.from_bytes(init_dataA, byteorder='little')



        print(a,"    ")
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
    def __init__(self, init_info, OscillogramNode):
        init_info = init_info.split('\n')
        LineChannels=init_info[1].split(',')
        self.AllChannels = LineChannels[0]
        self.channelsA = int(LineChannels[1][:-1])
        self.channelsD = int(LineChannels[2][:-1])
        self.size_line_data = 8 + (self.channelsA + math.ceil(self.channelsD/16))*2


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
    #oscillogram_info_list = []
    V1=[]
    V2=[]
    V3=[]
    x=[]
    constLineSize = size_binline(input_info)
    for i in range(int(len(input_data)/constLineSize)):
        oscillogram_node_list.append(OscillogramNode(input_data[:constLineSize]))
        V1.append(oscillogram_node_list[i].channelsA[0])
        V2.append(oscillogram_node_list[i].channelsA[1])
        V3.append(oscillogram_node_list[i].channelsA[2])
        x.append(oscillogram_node_list[i].start_time)
        input_data = input_data[constLineSize:]
        if (i<20):
            print(i ,"- Number: ", oscillogram_node_list[i].number, "; begin time: ", oscillogram_node_list[i].start_time,"; A: ", oscillogram_node_list[i].channelsA, "; D: ", oscillogram_node_list[i].channelsD, sep="")
    print(len(V1))
    y1=V1
    y2=V2
    y3=V3
    x=x
    #plt.xlim (0, 40000)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.show()
