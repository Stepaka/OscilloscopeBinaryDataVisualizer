import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq


FILE_DAT_NAME = "24e61716a082f03660813bc0fd6b3f56458cac13.dat"
FILE_CFG_NAME = "24e61716a082f03660813bc0fd6b3f56458cac13.cfg"


class AnalyzerRPA:
    def __init__(self, init_data):
        init_data = init_data.split('\n')

        # Получение информации о каналах.
        channels_info = init_data[1].split(',')
        self.channels_A = int(channels_info[1][:-1])    # Количество аналоговых каналов.
        self.channels_D = int(channels_info[2][:-1])    # Количество дискретных каналов.

        # Вычисление размера строки одной выборки, без учета ее номера и времени.
        self.data_line_size = (2 * (self.channels_A + math.ceil(self.channels_D / 16)))

        # Формат бинарных данных в одной строке.
        self.list_format = []
        self.list_format.append(np.uint32)   # Номер выборки.
        self.list_format.append(np.uint32)   # Время выборки.

        # Добавление формата для значений каналов.
        for i in range(self.data_line_size):
            self.list_format.append(np.int16)

        # Список коэффициентов трансформации аналоговый каналов.
        self.transform_coef_listS = []
        self.transform_coef_listP = []###
        self.transform_coef_listA = []###

        # Названия столбцов для DataFrame.
        self.cols_names = []
        self.cols_names.append('Number')
        self.cols_names.append('Time')

        # Добавление названий столбцов аналоговых каналов для DataFrame.
        for i in range(self.channels_A):
            temp = init_data[i + 2].split(',')
            self.cols_names.append(temp[1])

            # Добавление коэффициента трансформации текущего аналогового канала.
            self.transform_coef_listS.append(float(temp[-2]))
            self.transform_coef_listP.append(float(temp[-3]))
            self.transform_coef_listA.append(float(temp[-8]))

        # Добавление названий столбцов дискретных каналов для DataFrame.
        for i in range(math.ceil(self.channels_D / 16)):
            self.cols_names.append('D' + str(i + 1))

        # Частота дискретизации.
        self.simple_rate = int(init_data[self.channels_A + self.channels_D + 4].split(',')[0])### надо бы переделать
        self.freq = int(init_data[self.channels_A + self.channels_D + 2])
        self.sample = int(self.simple_rate/self.freq)

        # Преобразование списка коэффициентов трансформации в словарь ("имя" -> коэфф. трансформации).
        self.transform_coef_dictS = dict(zip(self.cols_names[2:], self.transform_coef_listS))
        self.transform_coef_dictP = dict(zip(self.cols_names[2:], self.transform_coef_listP))###
        self.transform_coef_dictA = dict(zip(self.cols_names[2:], self.transform_coef_listA))###

        # Создание DataFrame на основе полученных данных.
        temp_data = np.fromfile(FILE_DAT_NAME,
                                dtype=np.dtype({'names': self.cols_names,
                                                'formats': self.list_format}))
        self.data_frame = pd.DataFrame(temp_data, columns=temp_data.dtype.names)

        # Создание DataFrame с реальными значениями.
        self.true_df = df1 =self.data_frame.dropna()
        cols_names_to_print = self.cols_names[2:-math.ceil(self.channels_D / 16)]
        print(cols_names_to_print)
        graf_values=[[] for i in range(len(cols_names_to_print))]
        for i in range(len(cols_names_to_print)):
            graf_values[i] =  self.true_df[cols_names_to_print[i]].tolist()
            for el, item in enumerate(graf_values[i]):
                graf_values[i][el] = float(item) * self.transform_coef_dictA[cols_names_to_print[i]] * self.transform_coef_dictS[cols_names_to_print[i]] / self.transform_coef_dictP[cols_names_to_print[i]]
            self.true_df[cols_names_to_print[i]]=np.array(graf_values[i])


def get_data_from_file(filename):
    with open(filename, "r") as f:
        content = f.read()
    return content

# Создание графиков из DataFrame и полученных именен его столбцов.
def print_graf(cols_names_to_print,df):
    timePoint = df['Time'].tolist()
    graf_values=[[] for i in range(len(cols_names_to_print))]
    for i in range(len(cols_names_to_print)):
        graf_values[i] =  df[cols_names_to_print[i]].tolist()
        plt.plot(timePoint, graf_values[i], label=cols_names_to_print[i])
    plt.legend(loc=2)
    plt.show()

# Фурье.
def fourier_transform(df, name_column):
    myarray = np.asarray((df[name_column]))
    N = len(myarray)
    yf = rfft(myarray)
    xf = rfftfreq(N, 1 / analyzer.simple_rate)
    #plt.plot(xf, np.abs(yf))
    #plt.show()
    return yf[1]


def argand(a):
    xi=[]
    yi=[]
    for x in range(len(a)):
        xi.append(a[x].real)
        yi.append(a[x].imag)
        #plt.plot([0,a[x].real],[0,a[x].imag],'ro',label='python')#раскоментить если нужен вектор с центра
    plt.plot(xi,yi,'ro-',label='python')
    limit=np.max(np.ceil(np.absolute(a))) # установка пределов для осей
    plt.minorticks_on()
    #  Определяем внешний вид линий основной сетки:
    plt.grid(which='major',
            color = 'k',
            linewidth = 1)
    #  Определяем внешний вид линий вспомогательной сетки:
    plt.grid(which='minor',
            color = 'k',
            linestyle = ':')
    # plt.grid(axis = 'both')
    # plt.grid(which='minor', color = 'k', linewidth = 2)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.xlim((-limit,limit))
    plt.ylim((-limit,limit))
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    plt.show()


if __name__ == '__main__':
    input_data = get_data_from_file(FILE_CFG_NAME)
    analyzer = AnalyzerRPA(input_data)
    df = analyzer.true_df#.loc[0:35]
    columns = ['V','I', 'R']
    cd = pd.DataFrame(columns=columns)
    #print(cd)
    #print(max(df['Ix1']))

    cols_names_to_print = ['Ix1']
    print_graf(cols_names_to_print,df)

    #fourier_transform(df,'V1')

    for i in range(7608):#7608
        # Задаем интервал, на котором происходят преобразования Фурье.
        df = analyzer.true_df.loc[i:analyzer.sample-1+i]
        a={'V':fourier_transform(df,'V1'),'I':fourier_transform(df,'Ix1')}
        cd=cd.append(a, ignore_index = True)
    cd['R']= cd['V']  / cd['I']
    #print(cd)


    argand(cd['R'])
