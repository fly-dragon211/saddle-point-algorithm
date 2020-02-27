#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : dimer_vertical.py
# @Author: Fly_dragon
# @Date  : 2020/1/29
# @Desc  :
"""
本改动是建立于dimer在超球面上旋转
"""

import numpy as np
from simplesurface import SimpleSurface
import matplotlib.pyplot as plt
import random
import time


class Dimer:
    def __init__(self, n=2, ini_position=(np.pi/2, np.pi/2), ini_vector=np.random.rand(2)-0.5,
                 whether_print=False):
        # 是否打印中间数据
        self.whether_print = whether_print
        # 旋转力收敛值
        self.min_vertical_force = 1e-4

        self.min_value = 1e-20
        self.PES = SimpleSurface()
        self.n = n
        self.r = 0.1
        self.vector = np.zeros(n, 'f')
        self.normal = np.zeros(n, 'f')
        self.position = np.zeros(n, 'f')
        self.position[0] = ini_position[0]
        self.position[1] = ini_position[1]
        self.f1 = np.zeros(n, 'f')
        self.f2 = np.zeros(n, 'f')
        self.vector[:] = ini_vector
        # 向量归一化
        # 这里用=/会报错
        self.vector = self.vector / np.linalg.norm(self.vector)
        # 曲率
        self.c = 0
        self.update_c()
        self.delta = 0.001
        self.delta_angle = np.pi / 180
        # dimer速度
        self.v = np.zeros(n, 'f')
        # 给予一个初速度
        # self.v[:] = self.vector
        # dimer步进时间
        self.timer = 0.006
        self.f1[:] = self.force(self.position+self.vector*self.r)
        self.f2[:] = self.force(self.position-self.vector*self.r)
        self.f_rota = 0  # 旋转力
        self.update_normal()
        self.angle = 0
        self.position_list = [[], [], []]

    def get_value(self, position):
        """
        获取值
        """
        return self.PES.get_value(position)

    def force(self, position):
        """
        计算像点受力，一阶梯度，这里使用移动self.delta找梯度
        """
        value = self.get_value(position)
        delta_array = np.zeros(self.n, 'f')
        force_array = np.zeros(self.n, 'f')
        for i in range(self.n):
            delta_array[:] = 0
            delta_array[i] = self.delta
            new_position = position + delta_array
            new_value = self.get_value(new_position)
            # 负梯度为受力方向
            force_array[i] = (value-new_value) / self.delta
        return force_array
        # return self.PES.get_force(position)

    def vertical_force(self, force, vector):
        """
        计算垂直分力(旋转力)
        参数：
            force：力向量
            vector：dimer向量
        """
        # 平行分力
        f_parallel = np.dot(vector, force) * vector
        # 垂直分力
        f_vertical = force - f_parallel
        return f_vertical

    def get_rotate_angle(self):
        """
        计算dimer的旋转角
        """
        self.f1 = self.force(self.position+self.vector*self.r)
        self.f2 = self.force(self.position-self.vector*self.r)
        # 旋转力
        delta_f = (self.f1-self.f2) - np.dot(np.dot(self.f1-self.f2, self.vector), self.vector)
        f_abs = np.linalg.norm(delta_f)
        if f_abs < self.min_value:
            return 0
        # 垂直向量判断
        self.update_normal()
        if np.isnan(self.normal[0]):
            return 0
        # 这里是单位向量
        new_vector = self.vector*np.cos(self.delta_angle) + self.normal*np.sin(self.delta_angle)
        # 微旋转后的dimer受力
        new_f1 = self.force(self.position+new_vector * self.r)
        new_f2 = self.force(self.position-new_vector * self.r)
        delta_new_f = (new_f1-new_f2) - np.dot(new_f1-new_f2, new_vector) * new_vector
        temp = f_abs*np.cos(2*self.delta_angle)-np.dot(delta_new_f, delta_f) / f_abs
        angle = np.arctan((np.sin(2*self.delta_angle)*f_abs) / temp) / 2
        return angle

    def get_rotate_angle_v2(self):
        """
        计算dimer的旋转角
        """
        self.f1 = self.force(self.position+self.vector*self.r)
        self.f2 = self.force(self.position-self.vector*self.r)
        delta_f = (self.f1-self.f2) - np.dot(np.dot(self.f1-self.f2, self.vector), self.vector)
        # 力大小
        f_abs = np.linalg.norm(delta_f)
        if f_abs < self.min_value:
            return 0
        # 垂直向量判断
        self.update_normal()
        if np.isnan(self.normal[0]):
            return 0
        # 这里是单位向量
        new_vector = self.vector*np.cos(self.delta_angle) + self.normal*np.sin(self.delta_angle)
        # 微旋转后的dimer受力
        new_f1 = self.force(self.position+new_vector * self.r)
        new_f2 = self.force(self.position-new_vector * self.r)
        delta_new_f = (new_f1-new_f2) - np.dot(np.dot(new_f1-new_f2, new_vector), new_vector)
        new_f_abs = np.linalg.norm(delta_new_f)
        if new_f_abs < self.min_value:
            return self.delta_angle
        f_d = (f_abs - new_f_abs) / self.delta_angle
        angle = (f_abs + new_f_abs) / (-2 * f_d)
        return angle

    def rotate(self, angle):
        """
        旋转dimer
        """
        self.angle = angle
        # 垂直向量
        if np.isnan(self.normal[0]):
            return
        new_vector = self.vector*np.cos(self.angle) + self.normal*np.sin(self.angle)
        self.vector[:] = new_vector / np.linalg.norm(new_vector)
        self.f1 = self.force(self.position+self.vector*self.r)
        self.f2 = self.force(self.position-self.vector*self.r)
        # 计算曲率
        self.update_c()

    def translate(self):
        """
        移动dimer
        """
        self.angle = 0
        # dimer合力
        f_r = (self.f1 + self.f2) / 2
        # dimer平行力
        f_parallel = np.dot(self.vector, f_r) * self.vector
        # dimer指向鞍点力
        if self.c < 0:
            f_to_saddle = f_r - f_parallel * 2
        else:
            f_to_saddle = - f_parallel
        delta_v = f_to_saddle
        # delta_v = f_to_saddle * self.timer
        # 速度调整
        if np.dot(self.v, f_to_saddle) < 0:
            self.v = delta_v
        else:
            delta_v_abs = np.dot(delta_v, delta_v)
            if delta_v_abs < self.min_value:
                self.v[:] = 0
            else:
                self.v = delta_v * (1+np.dot(delta_v, self.v)/np.dot(delta_v, delta_v))
        self.position += (self.v * self.timer)
        # 计算新点相关数据
        self.f1[:] = self.force(self.position+self.vector*self.r)
        self.f2[:] = self.force(self.position-self.vector*self.r)

    def work(self):
        # 旋转到曲率最小
        # rotate_angle = np.pi / 180
        # self.position[:] = [np.random.random(), np.random.random()]
        x = [self.position[0]]
        y = [self.position[1]]
        z = [self.get_value(self.position)]
        times = []
        for i in range(1200):
            for j in range(300):
                rotate_angle = self.get_rotate_angle()
                self.rotate(rotate_angle)
                # print(rotate_angle)
                print('rotated force: ', self.f_rota)
                # 垂直力大小
                if np.linalg.norm(self.vertical_force(self.f1-self.f2, self.vector)) < self.min_vertical_force:
                    times.append(j)
                    break
                elif j == 299:
                    times.append(j)
            # print(self.vector, 'angle', self.angle)
            self.translate()
            # print(self.c)
            x.append(self.position[0])
            y.append(self.position[1])
            z.append(self.get_value(self.position))
            if (np.abs(self.f1 + self.f2) < 0.1).all():
                if self.c < 0.0:
                    if self.whether_print:
                        print('step: ', i)
                    break
        if self.whether_print:
            print('time', times)
            print(self.position/np.pi*180)
        self.PES.show_point_2d(x[:20], y[:20])
        # self.PES.show_surface_2d(-max(x) * 1.1, max(x) * 1.5)
        self.PES.show_surface_2d(-5, 5)
        self.PES.show()
        return x, y

    def update_c(self):
        """
        计算曲率
        """
        self.c = -np.dot(self.f1-self.f2, self.vector) / 2 / self.r
        if self.whether_print:
            print(self.c)

    def update_normal(self):
        """
        计算法向量
        """
        normal = self.vertical_force(self.f1-self.f2, self.vector)
        normal_abs = np.linalg.norm(normal)
        if normal_abs < self.min_value:
            # 如果垂直力为0，中断计算垂直向量
            # self.normal[:] = np.float('nan')
            self.normal[:] = 0
        else:
            self.normal[:] = normal / normal_abs

        # 旋转力
        self.f_rota = np.linalg.norm(self.f1 - self.f2 - np.dot(self.f1 - self.f2, self.vector) * self.vector)
        return

def test():
    # 正常收敛
    x1 = []
    y1 = []
    # 非正常收敛
    x2 = []
    y2 = []
    for i in range(60):
        np.random.seed(i)
        # plt.figure(i+1)
        ini_vector = np.random.rand(2) - 0.5
        ini_position = np.random.rand(2) * np.pi
        plt.title('initial vector:' + str(ini_vector))
        d = Dimer(2, ini_vector=ini_vector)
        d.work()
        print('curvature', d.c)
        if d.c == -0.0:
            x2.append(ini_vector[0])
            y2.append(ini_vector[1])
        else:
            x1.append(ini_vector[0])
            y1.append(ini_vector[1])
    plt.scatter(x1, y1, c='r')
    plt.scatter(x2, y2, c='b')
    plt.title('the effect of different initial direction')
    plt.show()
    return


if __name__ == "__main__":
    # test()
    np.random.seed(5)
    ini_vector = np.random.rand(2) - 0.5
    ini_position = (0, -0.5)
    angle = np.pi / 180 * 3
    ini_vector = [np.cos(angle), np.sin(angle)]
    d = Dimer(2, ini_vector=ini_vector, whether_print=True)
    d.work()


    # t = 100  # 等分成20份
    # x = []
    # y1 = []
    # y2 = []
    # for i in range(t):
    #     angle = np.pi * 2 * i / t
    #     ini_vector = [np.cos(angle), np.sin(angle)]
    #     ini_position = [5, 15]
    #     d = Dimer(2, ini_vector=ini_vector, ini_position=ini_position, whether_print=False)
    #     d.update_normal()
    #     d.update_c()
    #     x.append(angle * 180 / np.pi)
    #     y1.append(d.c)
    #     y2.append(d.f_rota)
    #     print('angle', angle * 180 / np.pi)
    #     print('curvature', d.c, 'rotate force', d.f_rota)
    # plt.plot(x, y2)
    # plt.xlabel('angle')
    # plt.ylabel('rotate force')

