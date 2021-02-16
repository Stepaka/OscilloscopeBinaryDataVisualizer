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
        self.transform_coef_list = []

        # Названия столбцов для DataFrame.
        self.cols_names = []
        self.cols_names.append('Number')
        self.cols_names.append('Time')

        # Добавление названий столбцов аналоговых каналов для DataFrame.
        for i in range(self.channels_A):
            temp = init_data[i + 2].split(',')
            self.cols_names.append(temp[1])

            # Добавление коэффициента трансформации текущего аналогового канала.
            self.transform_coef_list.append(float(temp[-2]))

        # Добавление названий столбцов дискретных каналов для DataFrame.
        for i in range(math.ceil(self.channels_D / 16)):
            self.cols_names.append('D' + str(i + 1))

        # Частота дискретизации.
        self.simple_rate = int(init_data[self.channels_A + self.channels_D + 4].split(',')[0])

        # Преобразование списка коэффициентов трансформации в словарь ("имя" -> коэфф. трансформации).
        self.transform_coef_dict = dict(zip(self.cols_names[2:], self.transform_coef_list))

        # Создание DataFrame на основе полученных данных.
        temp_data = np.fromfile(FILE_DAT_NAME,
                                dtype=np.dtype({'names': self.cols_names,
                                                'formats': self.list_format}))
        self.data_frame = pd.DataFrame(temp_data, columns=temp_data.dtype.names)


def get_data_from_file(filename):
    with open(filename, "r") as f:
        content = f.read()
    return content


if __name__ == '__main__':
    input_data = get_data_from_file(FILE_CFG_NAME)
    analyzer = AnalyzerRPA(input_data)

    # Задаем интервал, на котором происходят преобразования Фурье.
    df = analyzer.data_frame.loc[0:35]

    print(df)

    cols_names_to_print = ['Ix1']
    timePoint = df['Time'].tolist()
    graf_values=[[] for i in range(len(cols_names_to_print))]
    for i in range(len(cols_names_to_print)):
        graf_values[i] =  df[cols_names_to_print[i]].tolist()
        for el, item in enumerate(graf_values[i]):
            graf_values[i][el] = float(item)/analyzer.transform_coef_dict[cols_names_to_print[i]]
        plt.plot(timePoint, graf_values[i], label=cols_names_to_print[i])
    plt.legend(loc=2)
    plt.show()

    # Блок Фурье.
    myarray = np.asarray((graf_values[0]))

    # Число точек в normalized_tone.
    N = len(myarray)
    yf = rfft(myarray)
    print(len(yf))
    xf = rfftfreq(N, 1 / analyzer.simple_rate)
    plt.plot(xf, np.abs(yf))
    plt.show()
