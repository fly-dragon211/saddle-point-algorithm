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
    def __init__(self, n=2, ini_position=(np.pi/2, np.pi/2),
                 ini_vector=np.random.rand(2)-0.5,
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
        self.position[:] = ini_position[:]
        self.position_pre = np.zeros(n, 'f')
        self.f1 = np.zeros(n, 'f')
        self.f_r = np.zeros(n, 'f')  # the force of midpoint
        self.f2 = np.zeros(n, 'f')
        self.vector[:] = ini_vector
        # 向量归一化
        # 这里用=/会报错
        self.vector = self.vector / np.linalg.norm(self.vector)
        # 曲率
        self.c = 0
        self.delta = 0.001
        self.delta_angle = np.pi / 180
        # dimer速度
        self.v = np.zeros(n, 'f')
        # 给予一个初速度
        # self.v[:] = self.vector
        # dimer步进时间
        self.timer = 0.06
        self.update_f()  # 更新受力
        self.f_rota = 0  # 旋转力
        self.update_normal()
        self.update_c()
        self.angle = 0
        self.position_list = []

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
        # return self.PES.get_diff(position)

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
        self.update_f()
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
        # 更新受力
        self.update_f()
        # 计算曲率
        self.update_c()

    def translate(self):
        """
        移动dimer
        """
        self.angle = 0
        # dimer合力
        f_r = self.f_r
        # dimer平行力
        f_parallel = np.dot(self.vector, f_r) * self.vector
        # dimer指向鞍点力
        if self.c < 0:
            f_to_saddle = f_r - f_parallel * 2  # 平行力反向
        else:
            f_to_saddle = - f_parallel
        # f_to_saddle = f_r - f_parallel * 2  这样会增加旋转，减少移动
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
        self.update_f()

    def work(self):
        # 旋转到曲率最小
        times = []
        for i in range(1200):
            pre_c = self.c
            for j in range(200):
                rotate_angle = self.get_rotate_angle()
                self.rotate(rotate_angle)
                if self.whether_print:
                    print('rotated force: ', self.f_rota)
                # 垂直力大小
                if np.linalg.norm(self.vertical_force(self.f1-self.f2, self.vector)) < self.min_vertical_force:
                    if self.c > pre_c:  # 如果曲率上升，旋转pi/2 + angle, 重要
                        self.rotate(np.pi / 2)
                        continue
                    times.append(j)
                    break
                elif j == 199:
                    times.append(j)
            self.translate()
            self.position_list.append(self.position.copy())
            if (np.abs(self.f1 + self.f2) < 1).all():  # 所有方向都相反
                if self.c < 0.2:
                    if self.whether_print:
                        print('step: ', i)
                    break
        if self.whether_print:
            print('time', times)
            print(self.position/np.pi*180)
        # self.PES.show_point_2d(np.array(self.position_list))
        # self.PES.show_surface_2d(-5, 5)
        # plt.title('It rotates %d times and run %d times' % (sum(times), len(times)))
        # self.PES.show()
        return np.array(self.position_list), times

    def update_c(self):
        """
        计算曲率
        """
        self.c = -np.dot(self.f1-self.f2, self.vector) / 2 / self.r
        if self.whether_print:
            print('curvature: ', self.c)

    def update_normal(self):
        """
        计算法向量
        """
        normal = self.vertical_force(self.f1-self.f2, self.vector)
        normal_abs = np.linalg.norm(normal)
        if normal_abs < self.min_value:
            # 如果垂直力为0，中断计算垂直向量
            self.normal[:] = np.float('nan')
            # self.normal[:] = 0
        else:
            self.normal[:] = normal / normal_abs

        # 旋转力
        self.f_rota = np.linalg.norm(self.f1 - self.f2 - np.dot(self.f1 - self.f2, self.vector) * self.vector)
        return

    def update_f(self):
        """
        更新受力，利用像点是否移动进行效率优化
        """
        if (self.position == self.position_pre).all():  # 没有移动
            self.f1[:] = self.force(self.position + self.vector * self.r)
            self.f2[:] = 2 * self.f_r - self.f1
        else:
            self.f1[:] = self.force(self.position + self.vector * self.r)
            self.f_r[:] = self.force(self.position)
            self.f2[:] = 2 * self.f_r - self.f1
            self.position_pre[:] = self.position[:]
        return



def test_1():
    np.random.seed(7)
    for i in range(1, 7):
        plt.figure(i)
        ini_position = (np.random.rand(2) - 0.5) * 10
        angle = np.pi / 180 * 3
        ini_vector = [np.cos(angle), np.sin(angle)]
        ini_vector = np.random.rand(2)
        d = Dimer(2, ini_position, ini_vector, whether_print=True)
        position_d, times_d = d.work()  # 得到dimer运行轨迹和每一次的旋转数
        d.PES.show_point_2d(position_d)
        d.PES.show_surface_2d(-5, 5)
        plt.title('Dimer rotates %d times and run %d times \n '
                  % (sum(times_d), len(times_d)))

        x1 = d.position + d.vector * d.r
        plt.plot(x1[0], x1[1], 'ko')  # 画出点1位置


if __name__ == "__main__":
    test_1()
    # np.random.seed(7)
    # for i in range(0, 7):
    #     ini_position = (np.random.rand(2) - 0.5) * 10
    #     angle = np.pi / 180 * 30*i
    #     ini_vector = [np.cos(angle), np.sin(angle)]
    #     # ini_vector = np.random.rand(2)
    #     d = Dimer(2, ini_position, ini_vector, whether_print=False)
    #     print('angle:', 30*i, 'rotate_force:', d.f_rota, 'curvature:%.3f' % d.c)

