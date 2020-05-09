# -*-coding=utf-8-*-

"""
构建一个简单势能面
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class SimpleSurface():
    def __init__(self):
        self.n = 2  # 变量个数
        self.delta = 0.001  # 计算梯度时使用
        self.point = []
        self.cal_num = [0, 0, 0]  # 能量，梯度，hessian计算次数

    def get_value(self, position):
        """
        计算函数
        """
        self.cal_num[0] += 1
        _x = position
        # return np.sin(_x[0]) + np.sin(_x[1])
        return _x[0]**4 + 4*_x[0]**2*_x[1]**2 - 2*_x[0]**2 + 2*_x[1]**2

    def get_diff(self, position):
        """
        计算一阶梯度
        """
        self.cal_num[1] += 1
        x = position
        # return np.array([np.cos(x[0]), np.cos(x[1])])
        return np.array([4*x[0]**3 + 8*x[0]*x[1]**2 - 4*x[0], 8*x[0]**2*x[1] + 4*x[1]])

    def get_hess(self, position):
        """
        Hessian矩阵
        """
        self.cal_num[2] += 1
        x = position
        return np.array([[-np.sin(x[0]), 0], [0, -np.sin(x[1])]])

    def show_surface_3d(self):
        """
        放在show_point后面
        :return:
        """
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        x_array = np.arange(-1, 1, 0.03)
        y_array = np.arange(-1, 1, 0.03)
        x, y = np.meshgrid(x_array, y_array)
        # z = [self.get_value([x_, y_]) for x_, y_ in zip(x, y)]
        z = self.get_value([x, y])
        # self.ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
        self.ax.scatter(x, y, z, c='r')
        # ax.contourf(x, y, z, zdir='z', offset=2)
        # plt.show()
        return

    def show_point_3d(self, x, y, z):
        """
        显示点
        """
        # ax = fig.add_subplot(111, projection='3d')
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        self.ax.scatter(x, y, z)
        self.ax.set_zlim(-4, 4)

        return

    def show_point_2d(self, x, color='ro'):
        plt.plot(x[:, 0], x[:, 1], color)
        plt.plot(x[-1, 0], x[-1, 1], 'k*')
        plt.annotate('Start Point', xy=(x[0, 0], x[0, 1]), fontsize=15, color='w')
        plt.annotate('TS', xy=(x[-1, 0], x[-1, 1]), fontsize=15, color='w')

    def show_surface_2d(self, x_min, x_max):
        # 建立步长为0.01，即每隔0.01取一个点
        step = 0.01
        x = np.arange(x_min, x_max, step)
        y = np.arange(x_min, x_max, step)
        plt.annotate('IS', xy=(-0.48, 0), fontsize=20, color='w')
        plt.annotate('FS', xy=(0.43, 0), fontsize=20, color='w')
        # 也可以用x = np.linspace(-10,10,100)表示从-10到10，分100份

        # 将原始数据变成网格数据形式
        X, Y = np.meshgrid(x, y)
        # 写入函数，z是大写
        Z = self.get_value((X, Y))
        # 设置打开画布大小,长10，宽6
        # plt.figure(figsize=(10,6))
        # 填充颜色，f即filled
        plt.contourf(X, Y, Z)
        # 画等高线
        plt.contour(X, Y, Z)

        return

    def show(self):
        plt.show()


if __name__ == "__main__":
    ss = SimpleSurface()
    x = [1, 1]
    y = [1, 2]
    z = [3, 3]
    # ss.show_surface_2d(-5, 5)
    ss.show_surface_3d()
    ss.show()
