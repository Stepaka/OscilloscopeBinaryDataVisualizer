import math
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import MultiPolygon
from shapely.geometry import MultiLineString
from shapely.geometry import LineString

FILE_DAT_NAME = "SA-2.DAT" #"24e61716a082f03660813bc0fd6b3f56458cac13.dat"
FILE_CFG_NAME = "SA-2.CFG" #24e61716a082f03660813bc0fd6b3f56458cac13.cfg"

POLYGON_F1 = 85
POLYGON_F2 = 22
POLYGON_F3 = 120

class AnalyzerRPA:
    def __init__(self):

        with open(FILE_CFG_NAME, "r") as f:
            init_data = f.read()
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

        # В каком значении записаны данные выборок (первичка (P) или вторичка (S))
        self.type_value_dat = init_data[2].split(',')[-1]

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
        self.simple_rate = float(init_data[self.channels_A + self.channels_D + 4].split(',')[0])### надо бы переделать
        self.freq = float(init_data[self.channels_A + self.channels_D + 2])
        self.sample = int(self.simple_rate/self.freq)

        # Преобразование списка коэффициентов трансформации в словарь ("имя" -> коэфф. трансформации).
        self.transform_coef_dictS = dict(zip(self.cols_names[2:], self.transform_coef_listS))
        self.transform_coef_dictP = dict(zip(self.cols_names[2:], self.transform_coef_listP))###
        self.transform_coef_dictA = dict(zip(self.cols_names[2:], self.transform_coef_listA))###

        # Тип чтения из dat файла .
        self.type = init_data[self.channels_A + self.channels_D + 7]

        # Создание DataFrame на основе полученных данных.
            # Если тип BINARY.
        if self.type == 'BINARY':
            temp_data = np.fromfile(FILE_DAT_NAME,
                                    dtype=np.dtype({'names': self.cols_names,
                                                    'formats': self.list_format}))
            self.data_frame = pd.DataFrame(temp_data, columns=temp_data.dtype.names)

            # Если тип ASCII.
        elif self.type == 'ASCII':
            with open(FILE_DAT_NAME,'r') as csvfile:
                plots = csv.reader(csvfile, delimiter=',')
                self.data_frame = pd.DataFrame(columns=self.cols_names[:- math.ceil(self.channels_D / 16)])
                iter=0
                for row in plots:
                    self.data_frame.loc[iter]=row[:-self.channels_D]
                    iter+=1
        print(self.data_frame)

        # Создание DataFrame с реальными значениями.
        self.true_df = df1 =self.data_frame.dropna()
        cols_names_to_print = self.cols_names[2:-math.ceil(self.channels_D / 16)]
        graf_values=[[] for i in range(len(cols_names_to_print))]
        for i in range(len(cols_names_to_print)):
            graf_values[i] =  self.true_df[cols_names_to_print[i]].tolist()
            for el, item in enumerate(graf_values[i]):
                if self.type_value_dat == 'P':
                    graf_values[i][el] = float(item) * self.transform_coef_dictA[cols_names_to_print[i]]# * self.transform_coef_dictS[cols_names_to_print[i]] / self.transform_coef_dictP[cols_names_to_print[i]]
                elif self.type_value_dat == 'S':
                    graf_values[i][el] = float(item) * self.transform_coef_dictA[cols_names_to_print[i]] * self.transform_coef_dictP[cols_names_to_print[i]] / self.transform_coef_dictS[cols_names_to_print[i]]
            self.true_df[cols_names_to_print[i]]=np.array(graf_values[i])

        # Число для построения полоного графика годографа
        self.full_int_hodograph = math.ceil(len(self.true_df)/2) - self.sample +1

    # Создание графиков из DataFrame и полученных именен его столбцов.
    def print_graf(self,cols_names_to_print,df):
        timePoint = df['Time'].tolist()
        graf_values=[[] for i in range(len(cols_names_to_print))]
        for i in range(len(cols_names_to_print)):
            graf_values[i] =  df[cols_names_to_print[i]].tolist()
            plt.plot(timePoint, graf_values[i], label=cols_names_to_print[i])
        plt.legend(loc=2)
        plt.show()

    # Фурье.
    def fourier_transform(self,df, name_column):
        myarray = np.asarray((df[name_column]))
        N = len(myarray)
        yf = rfft(myarray)
        xf = rfftfreq(N, 1 / analyzer.simple_rate)
        #plt.plot(xf, np.abs(yf))
        #plt.show()
        return yf[1]

    # Годограф
    def argand(self,poly = 'NULL'):
        a = self.calculate_resistance()
        xi=[]
        yi=[]
        for x in range(len(a)):
            xi.append(a[x].real)
            yi.append(a[x].imag)
            p1 = Point(a[x].real, a[x].imag)
            # if p1.within(poly):# дает 1 когда точка попадает в характеристику
            #     print("1")
            #plt.plot([0,a[x].real],[0,a[x].imag],'ro',label='python')#раскоментить если нужен вектор с центра
        plt.plot(xi,yi,'ro-',label='python')
        limit=np.max(np.ceil(np.absolute(a))) # установка пределов для осей
        plt.minorticks_on()
        #  Определяем внешний вид линий основной сетки:
        plt.grid(which='major', color = 'k', linewidth = 1)
        #  Определяем внешний вид линий вспомогательной сетки:
        plt.grid(which='minor', color = 'k', linestyle = ':')
        # plt.grid(axis = 'both')
        # plt.grid(which='minor', color = 'k', linewidth = 2)
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')
        plt.xlim((-limit,limit))
        plt.ylim((-limit,limit))
        plt.ylabel('Imaginary')
        plt.xlabel('Real')
        plt.show()


    def plot_coords(self,coords):
            pts = list(coords)
            x,y = zip(*pts)
            plt.plot(x,y)


    def plot_polys(self,polys):
        for poly in polys:
            if (not getattr(poly, "exterior", None)):
                print("got line?")

            self.plot_coords(poly.exterior.coords)

            for hole in poly.interiors:
                self.plot_coords(hole.coords)

    # Возвращает кординаты точки пересечения двух прямых
    def calculate_coord_point(self,start_point1, angle1, start_point2, angle2):
        intersection_of_lines = []
        intersection_of_lines.append((start_point2[1] - start_point1[1] - np.tan(angle2)*start_point2[0] + np.tan(angle1)*start_point1[0]) / (np.tan(angle1) - np.tan(angle2)))
        intersection_of_lines.append(np.tan(angle1) * (intersection_of_lines[0] + start_point1[0]) + start_point1[1])
        return intersection_of_lines

    # Создает ступень характерисики
    def create_polygon(self,f_l, f_4, R , X):
        coords = []
        coords.append([0,0])
        coords.append(self.calculate_coord_point( [0,0], np.deg2rad(POLYGON_F3), [0,X], np.deg2rad(0) ))
        coords.append( [X * np.tan(np.deg2rad(90-f_l)), X] )
        coords.append(self.calculate_coord_point( [X * np.tan(np.deg2rad(5)),X], np.deg2rad(-f_4), [R,0], np.deg2rad(POLYGON_F1) ))
        coords.append(self.calculate_coord_point( [0,0], np.deg2rad(-POLYGON_F2), [R,0], np.deg2rad(POLYGON_F1) ))
        poly = Polygon(coords)
        self.plot_polys([poly])
        return poly

    # Возвращает столбец с сопротивлением
    def calculate_resistance(self):
        columns = ['V','I', 'R']
        cd = pd.DataFrame(columns=columns)
        for i in range(self.full_int_hodograph):#self.full_int_hodograph
            # Задаем интервал, на котором происходят преобразования Фурье.
            df = analyzer.true_df.loc[i:analyzer.sample-1+i]
            a={'V':self.fourier_transform(df,'uL1')-self.fourier_transform(df,'uL2'),'I':self.fourier_transform(df,'iL1')*2}
            cd=cd.append(a, ignore_index = True)
        cd['R']= cd['V']  / cd['I']
        return cd['R']



if __name__ == '__main__':
    analyzer = AnalyzerRPA()
    df = analyzer.true_df#.loc[0:35]
    #print(max(df['uL1']))

    cols_names_to_print = ['uL1']
    analyzer.print_graf(cols_names_to_print,df)

    #analyzer.fourier_transform(df,'V1')

    analyzer.argand(analyzer.create_polygon(85, 20, 2.4, 4.4))# (fл, f4, R, X)

