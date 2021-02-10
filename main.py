import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.fft import rfft, rfftfreq


class InfoOs:
    def __init__(self, init_info):
        init_info = init_info.split('\n')
        LineChannels=init_info[1].split(',')
        #self.AllChannels = LineChannels[0]
        self.channelsA = int(LineChannels[1][:-1])
        self.channelsD = int(LineChannels[2][:-1])
        self.size_line_data = (self.channelsA + math.ceil(self.channelsD/16))*2
        self.coffTransforList = []
        self.formats = []
        self.formats.append(np.uint32)
        self.formats.append(np.uint32)
        for i in range(self.size_line_data):
            self.formats.append(np.int16)
        self.name = []
        self.name.append('Nomber')
        self.name.append('time')
        for i in range(2, self.channelsA+2):
            temp = init_info[i].split(',')
            self.coffTransforList.append(float(temp[-2]))
            LineChannels=init_info[i].split(',')
            self.name.append(LineChannels[1])
        for i in range(math.ceil(self.channelsD/16)):
            self.name.append('D'+str(i+1))
            #print(self.name)
        self.coffTransfor = dict(zip(self.name[2:], self.coffTransforList))
        self.simpleRate = int(init_info[self.channelsA+self.channelsD+4].split(',')[0])

def get_info_from_file(filename):
    with open(filename, "r") as f:
        info =f.read()
    return info



if __name__ == '__main__':
    input_info =  get_info_from_file("24e61716a082f03660813bc0fd6b3f56458cac13.cfg")
    constLineSize = InfoOs(input_info)
    dt = np.dtype({'names': constLineSize.name,
               'formats': constLineSize.formats})
    data = np.fromfile("24e61716a082f03660813bc0fd6b3f56458cac13.dat", dtype=dt)
    df = pd.DataFrame(data)
    df = pd.DataFrame(data, columns=data.dtype.names)

    #Задаем интервал на котором происходит преобразования Фурье
    df=df.loc[5000:10000]
    print(df)

    name_graf_to_print = ['Ix2']#'Ix1','Ix2','Ix3'
    timePoint = df['time'].tolist()
    graf_values=[[] for i in range(len(name_graf_to_print))]
    for i in range(len(name_graf_to_print)):
        graf_values[i] =  df[name_graf_to_print[i]].tolist()
        for el, item in enumerate(graf_values[i]):
            graf_values[i][el] = float(item)/constLineSize.coffTransfor[name_graf_to_print[i]]
        plt.plot(timePoint, graf_values[i], label=name_graf_to_print[i])
    plt.legend(loc=2)
    plt.show()

    #Блок Фурье
    myarray = np.asarray((graf_values[0]))
    normalized_tone = np.int16((myarray / myarray.max()) * 32767)
    # число точек в normalized_tone
    N = len(myarray)
    yf = rfft(normalized_tone)
    print(len(yf))
    xf = rfftfreq(N, 1 / constLineSize.simpleRate)
    plt.plot(xf, np.abs(yf))
    plt.show()

    #plt.xlim (0, 5000)
    #print(df['time'][:5])
