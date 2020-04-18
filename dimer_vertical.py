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
import pandas as pd
from simplesurface import SimpleSurface
import matplotlib.pyplot as plt
import random
import time


class Dimer:
    def __init__(self, PES=SimpleSurface(), n=2, ini_position=None,
                 ini_vector=None, ini_velocity=None,
                 whether_print=False):
        # 是否打印中间数据
        self.whether_print = whether_print
        # 旋转力收敛值
        self.min_vertical_force = 1e-5
        self.min_value = 1e-20
        self.PES = PES
        self.n = n
        self.r = 0.005  # 半径
        self.vector = np.zeros(n, 'f')
        self.normal = np.zeros(n, 'f')
        self.position = np.zeros(n, 'f')
        self.position_pre = np.zeros(n, 'f')
        self.f1 = np.zeros(n, 'f')
        self.f_r = np.zeros(n, 'f')  # the force of midpoint
        self.f_r_pre = np.zeros(n, 'f')
        self.f2 = np.zeros(n, 'f')
        self.f_vertical = np.zeros(n, 'f')  # 旋转力
        self.v = np.zeros(n, 'f')  # dimer初速度
        # 更新预设
        self.vector[:] = ini_vector if ini_vector is not None else np.random.rand(n) - 0.5
        self.position[:] = ini_position if ini_position is not None else np.random.rand(n)
        self.v[:] = ini_velocity if ini_velocity is not None else self.vector
        # 向量归一化
        # 这里用=/会报错
        self.vector = self.vector / np.linalg.norm(self.vector)
        # 曲率
        self.c = 0
        self.delta_angle = np.pi / 180
        self.timer = 0.05  # dimer步进时间
        self.deltax_max = 0.2
        self.translate_situation = [0, 0, 0, 0]  # 各种移动情况计算次数
        self.bk = np.eye(n)  # Hessian矩阵近似
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
        # value = self.get_value(position)
        # delta_array = np.zeros(self.n, 'f')
        # force_array = np.zeros(self.n, 'f')
        # for i in range(self.n):
        #     delta_array[:] = 0
        #     delta_array[i] = self.delta
        #     new_position = position + delta_array
        #     new_value = self.get_value(new_position)
        #     # 负梯度为受力方向
        #     force_array[i] = (value-new_value) / self.delta
        # return force_array
        return self.PES.get_diff(position)

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
        self._update_normal()
        # 旋转力
        delta_f = self.f_vertical
        f_abs = np.linalg.norm(delta_f)
        if f_abs < self.min_value:
            return 0
        # 垂直向量判断
        if np.isnan(self.normal[0]):
            return 0
        # 这里是单位向量
        new_vector = self.vector * np.cos(self.delta_angle) + self.normal * np.sin(self.delta_angle)
        # 微旋转后的dimer受力
        new_f1 = self.force(self.position + new_vector * self.r)
        new_f2 = 2 * self.f_r - new_f1
        delta_new_f = (new_f1 - new_f2) - np.dot(new_f1 - new_f2, new_vector) * new_vector
        temp = f_abs * np.cos(2 * self.delta_angle) - np.dot(delta_new_f, delta_f) / f_abs
        angle = np.arctan((np.sin(2 * self.delta_angle) * f_abs) / temp) / 2
        return angle

    def get_rotate_angle_v2(self):
        """
        计算dimer的旋转角
        """
        self._update_f()
        delta_f = (self.f1 - self.f2) - np.dot(np.dot(self.f1 - self.f2, self.vector), self.vector)
        # 力大小
        f_abs = np.linalg.norm(delta_f)
        if f_abs < self.min_value:
            return 0
        # 垂直向量判断
        self._update_normal()
        if np.isnan(self.normal[0]):
            return 0
        # 这里是单位向量
        new_vector = self.vector * np.cos(self.delta_angle) + self.normal * np.sin(self.delta_angle)
        # 微旋转后的dimer受力
        new_f1 = self.force(self.position + new_vector * self.r)
        new_f2 = 2 * self.f_r - new_f1
        delta_new_f = (new_f1 - new_f2) - np.dot(np.dot(new_f1 - new_f2, new_vector), new_vector)
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
        new_vector = self.vector * np.cos(self.angle) + self.normal * np.sin(self.angle)
        self.vector[:] = new_vector / np.linalg.norm(new_vector)
        # 更新受力
        self._update_f()
        # 计算曲率
        self._update_c()

    def translate(self, method=0):
        if method == 0:
            return self._translate_v0()
        elif method == 1:
            return self._translate_v1()
        elif method == 2:
            return self._translate_v2()
        else:
            input('Find no translate method! ')

    def work_list(self, cal_method=0):
        self._update_all()
        # 旋转到曲率最小
        f_r_list = []
        f_parallel_list = []
        c_list = []
        rotate_times = []
        for i in range(120):
            pre_c = self.c
            for j in range(20):
                if self.whether_print:
                    print('rotated force: ', np.linalg.norm(self.f_vertical))
                # 垂直力大小
                if np.linalg.norm(self.vertical_force(self.f1 - self.f2, self.vector)) < self.min_vertical_force:
                    if self.c > pre_c and self.c > 0:  # 如果曲率上升，旋转pi/2 + angle, 重要
                        self.rotate(np.pi / 2)
                        continue
                    break
                rotate_angle = self.get_rotate_angle()
                self.rotate(rotate_angle)
            self.translate(cal_method)
            # 画出端点位置
            plt.plot(self.position[0], self.position[1], 'ro')
            x1 = self.position + self.vector * self.r * 3
            plt.plot(x1[0], x1[1], 'bo')  # 画出点1位置
            plt.show()
            rotate_times.append(j)

            f_r_list.append(np.linalg.norm(self.f_r))
            f_parallel_list.append(np.linalg.norm(np.dot(self.vector, self.f_r) * self.vector, ord=1))
            c_list.append(self.c)
            self.position_list.append(self.position.copy())
            if (np.abs(self.f_r) < 0.1).all():  # 所有方向都相反
                if self.c < 0.0:
                    break
        if self.whether_print:
            print('time', rotate_times)
            # print(self.position/np.pi*180)
        return np.array(self.position_list), rotate_times, f_r_list, f_parallel_list, c_list

    def _translate_v0(self):
        """
        移动dimer, 梯度下降法
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
        self.v = f_to_saddle
        self.position += (self.v * self.timer)
        # 计算新点相关数据
        self._update_f()

    def _translate_v1(self):
        """
        移动dimer, 加速梯度下降法？
        """
        self.angle = 0
        # dimer合力
        f_r = self.f_r
        # dimer平行力
        f_parallel = np.dot(self.vector, f_r) * self.vector
        # dimer指向鞍点力
        if self.c < 0:
            self.translate_situation[0] += 1
            f_to_saddle = f_r - f_parallel * 2  # 平行力反向
        else:
            self.translate_situation[1] += 1
            f_to_saddle = - f_parallel
        # f_to_saddle = f_r - f_parallel * 2  这样会增加旋转，减少移动
        delta_v = f_to_saddle
        # 速度调整
        if np.dot(self.v, f_to_saddle) < 0:
            self.v = delta_v
        else:
            delta_v_abs = np.dot(delta_v, delta_v)
            if delta_v_abs < self.min_value:
                self.v[:] = 0
            else:
                self.v = delta_v * (1 + np.dot(delta_v, self.v) / np.dot(delta_v, delta_v))
        self.position += (self.v * self.timer)
        # 计算新点相关数据
        self._update_f()

    def _translate_v2(self):
        """
        移动dimer, 线性探测法
        """
        self.angle = 0
        # dimer合力
        f_r = self.f_r
        # dimer平行力
        f_parallel = np.dot(self.vector, f_r) * self.vector
        # dimer指向鞍点力
        timer_m = 1  # self.timer的倍数
        if self.c < 0:
            f_to_saddle = f_r - f_parallel * 2  # 平行力反向
            self.v = f_to_saddle
            f_r_next = self.force(self.position + f_to_saddle * self.timer)
            if np.linalg.norm(f_r_next) > np.linalg.norm(self.f_r):
                self.translate_situation[0] += 1
                # 后面是极值，需要跳出
                timer_m = self.__get_m_translate_jump(f_to_saddle)
            elif (np.abs(self.f_r) < 0.16).all():
                self.translate_situation[1] += 1
                # 鞍点附近, 一维线性搜索，步长调整
                m = 1
                force_list = []
                force_list.append(self.f_r.copy())
                force_list.append(f_r_next)
                while m <= 10:
                    if (np.abs(force_list[-1]) < 0.1).all():
                        break
                    m += 2
                    f_r_next = self.force(self.position + f_to_saddle * self.timer * m)
                    force_list.append(f_r_next.copy())
                    if np.linalg.norm(force_list[-1]) > np.linalg.norm(force_list[-2]):
                        force_list.pop()
                        m -= 2
                        break
                self.f_r = force_list[-1]
                timer_m = m
            else:
                self.translate_situation[2] += 1
                timer_m = self.__get_m_translate_jump(f_to_saddle)
            if timer_m == 1:  # 可以减少一次梯度计算
                self.f_r = f_r_next

        else:
            self.translate_situation[3] += 1
            f_to_saddle = - f_parallel
            self.v = f_to_saddle
            # 一维搜索跳出该区域
            timer_m = self.__get_m_translate_jump(f_to_saddle)

        self.position += (self.v * self.timer * timer_m)
        # 计算新点相关数据
        self._update_f()
        return

    def __get_m_translate_jump(self, f_to_saddle, delta_m=2):
        """
        跳出极值区域，返回步进时间的倍数
        """
        value_list = [self.get_value(self.position),
                      self.get_value(self.position + f_to_saddle * self.timer)]
        m = 1
        while m < 12:
            value_1 = self.get_value(self.position + f_to_saddle * self.timer * (m+delta_m))
            if (value_list[-1] - value_list[-2]) * (value_1 - value_list[-1]) <= 0:
                break
            value_list.append(value_1)
            m += delta_m
        return m


    def _update_c(self):
        """
        计算曲率
        """
        self.c = -np.dot(self.f1 - self.f2, self.vector) / self.r / 2
        if self.whether_print:
            print('curvature: ', self.c)

    def _update_normal(self):
        """
        计算法向量
        """
        normal = self.vertical_force(self.f1 - self.f2, self.vector)
        normal_abs = np.linalg.norm(normal)
        if normal_abs < self.min_value:
            # 如果垂直力为0，中断计算垂直向量
            self.normal[:] = np.float('nan')
            # self.normal[:] = 0
        else:
            self.normal[:] = normal / normal_abs

        # 旋转力
        self.f_vertical = self.f1 - self.f2 - np.dot(np.dot(self.f1 - self.f2, self.vector), self.vector)
        return

    def _update_f(self):
        """
        更新受力，利用像点是否移动进行效率优化
        """
        if (self.position == self.position_pre).all():  # 没有移动
            self.f1[:] = self.force(self.position + self.vector * self.r)
            self.f2[:] = 2 * self.f_r - self.f1
        else:
            self.f1[:] = self.force(self.position + self.vector * self.r)
            if (self.f_r == self.f_r_pre).all():  # 没有更新f_r
                self.f_r[:] = self.force(self.position)
            self.f_r_pre[:] = self.f_r[:]
            self.f2[:] = 2 * self.f_r - self.f1
            self.position_pre[:] = self.position[:]
        return

    def _update_all(self):
        self._update_f()  # 更新受力
        self._update_normal()
        self._update_c()


class DimerRo(Dimer):
    """
    对旋转行为进行优化，计算\phi_1进行旋转，并优化收敛条件
    """

    def get_rotate_angle(self):
        """
        计算dimer的旋转角
        """
        self._update_normal()
        # 旋转力
        delta_f = self.f_vertical
        f_abs = np.linalg.norm(delta_f)
        if f_abs < self.min_value:
            return 0
        # 垂直向量判断
        if np.isnan(self.normal[0]):
            return 0
        theta_f = delta_f / f_abs
        # 计算\phi_1
        c_new = -np.dot(self.f1 - self.f_r, self.vector) / self.r
        partial_c = 2 * np.dot(self.f1 - self.f_r, theta_f) / self.r
        angle_1 = 0.5 * np.arctan(partial_c / np.abs(c_new) / 2)

        # 这里是单位向量
        new_vector = self.vector * np.cos(angle_1) + self.normal * np.sin(angle_1)
        # 微旋转后的dimer受力
        new_f1 = self.force(self.position + new_vector * self.r)
        new_f2 = 2 * self.f_r - new_f1
        delta_new_f = (new_f1 - new_f2) - np.dot(new_f1 - new_f2, new_vector) * new_vector
        temp = f_abs * np.cos(2 * angle_1) - np.dot(delta_new_f, delta_f) / f_abs
        angle = np.arctan((np.sin(2 * angle_1) * f_abs) / temp) / 2
        # c_angle = -np.dot(new_f1 - self.f_r, new_vector) / self.r  # 旋转后曲率
        return angle

    def work(self):
        self._update_all()
        # 旋转到曲率最小
        times = []
        for i in range(1200):
            pre_c = self.c
            for j in range(20):
                rotate_angle = self.get_rotate_angle()
                self.rotate(rotate_angle)
                if pre_c < self.c and self.c > 0:
                    self.rotate(np.pi / 2)
                    print('rotate_angle', rotate_angle * 180 / np.pi,
                          'pre_c: %f, c: %f' % (pre_c, self.c))
                # 旋转角度小于某个值
                if np.abs(rotate_angle) < np.pi / 180 * 1:
                    break
            self.translate_v3()
            # 画出端点位置
            x1 = self.position + self.vector * self.r * 3
            plt.plot(x1[0], x1[1], 'bo')  # 画出点1位置
            times.append(j)
            if self.whether_print:
                print('parallel force', np.dot(self.vector, self.f_r) * self.vector, '\n')
            self.position_list.append(self.position.copy())
            if (np.abs(self.f_r) < 0.1).all():  # 所有方向都相反
                if self.c < 0.0:
                    break
        if self.whether_print:
            print('time', times)
        return np.array(self.position_list), times

    def work2(self):
        self._update_all()
        # 旋转到曲率最小
        times = []
        for i in range(1200):
            for j in range(20):
                rotate_angle = self.get_rotate_angle()
                pre_c = self.c
                self.rotate(rotate_angle)
                if pre_c < self.c and self.c > 0:
                    self.rotate(np.pi / 2)
                    print('rotate_angle', rotate_angle * 180 / np.pi,
                          'pre_c: %f, c: %f' % (pre_c, self.c))

                # 垂直力大小
                if np.linalg.norm(self.vertical_force(self.f1 - self.f2, self.vector)) < self.min_vertical_force:
                    break
            self._translate_v1()
            print('end\n')
            plt.plot(self.position[0], self.position[1], 'ro')
            x1 = self.position + self.vector * self.r
            plt.plot(x1[0], x1[1], 'ko')  # 画出点1位置
            plt.show()
            times.append(j)
            if self.whether_print:
                print('parallel force', np.dot(self.vector, self.f_r) * self.vector, '\n')
            self.position_list.append(self.position.copy())
            if (np.abs(self.f_r) < 0.1).all():  # 所有方向都相反
                if self.c < 0.0:
                    break
            # print(self.position/np.pi*180)
        # self.PES.show_point_2d(np.array(self.position_list))
        # self.PES.show_surface_2d(-5, 5)
        # plt.title('It rotates %d times and run %d times' % (sum(times), len(times)))
        # self.PES.show()
        return np.array(self.position_list), times


class DimerQs(Dimer):
    """
    使用BFGS方法更新步长，在旋转时更新Bk
    """

    def rotate(self, angle):
        """
        旋转dimer
        """
        self.angle = angle
        # 垂直向量
        if np.isnan(self.normal[0]):
            return
        # 准备更新Bk
        position_pre = (self.position + self.vector * self.r).copy()
        force_pre = self.f1.copy()
        # 旋转
        new_vector = self.vector * np.cos(self.angle) + self.normal * np.sin(self.angle)
        self.vector[:] = new_vector / np.linalg.norm(new_vector)
        # 更新受力
        self._update_f()
        # 计算曲率
        self._update_c()
        # 更新bk
        if np.abs(angle) > np.pi / 2.5:  # 旋转太大，self.bk不更新
            return
        position_now = self.position + self.vector * self.r
        self._update_bk(position_pre, force_pre, position_now, self.f1.copy())

    def translate(self, method=0):
        if method == 0:
            return self._translate_v0()
        elif method == 1:
            return self._translate_v1()
        elif method == 2:
            return self._translate_v2()
        elif method == 3:
            return self._translate_v3()
        else:
            input('Find no translate method! ')

    def _translate_v3(self):
        """
        移动dimer, BFGS法
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
        delta_v = f_to_saddle

        position_pre = self.position.copy()
        force_pre = self.f_r.copy()
        translate_length = np.linalg.norm(self.bk.dot(self.f_r.reshape((-1, 1))).flatten())
        f_bk = -self.bk.dot(self.f_r).flatten()  # 拟牛顿指向力
        if translate_length > self.deltax_max:
            translate_length = self.deltax_max
        if self.timer < translate_length and \
                self.translate_situation[1] >= 1:
            self.position += translate_length * delta_v
            self.translate_situation[0] += 1
        else:
            self.translate_situation[1] += 1
            if np.dot(self.v, delta_v) > 0:
                self.v = delta_v * (1 + np.dot(delta_v, self.v) / np.dot(delta_v, delta_v))
            else:
                self.v = delta_v
            self.position += self.v * self.timer

        # 计算新点相关数据
        self._update_f()
        self._update_bk(position_pre, force_pre, self.position.copy(), self.f_r.copy())

    def _translate_v31(self):
        """
        移动dimer, 未优化BFGS法
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
        delta_v = f_to_saddle

        position_pre = self.position.copy()
        force_pre = self.f_r.copy()
        translate_length = np.linalg.norm(self.bk.dot(self.f_r.reshape((-1, 1))).flatten())
        f_bk = -self.bk.dot(self.f_r).flatten()  # 拟牛顿指向力
        if self.translate_situation[1] > 1:
            self.position += translate_length * delta_v * 0.5
            self.translate_situation[0] += 1
        else:
            self.translate_situation[1] += 1
            self.v = delta_v
            self.position += self.v * self.timer

        # 计算新点相关数据
        self._update_f()
        self._update_bk(position_pre, force_pre, self.position.copy(), self.f_r.copy())

    def work(self, cal_method=0):
        self._update_all()
        # 旋转到曲率最小
        rotate_times = []
        for i in range(120):
            pre_c = self.c
            for j in range(0, 20):
                if self.whether_print:
                    print('rotated force: ', np.linalg.norm(self.f_vertical))
                # 垂直力大小
                if np.linalg.norm(self.vertical_force(self.f1 - self.f2, self.vector)) < self.min_vertical_force:
                    if self.c > pre_c and self.c > 0:  # 如果曲率上升，旋转pi/2 + angle, 重要
                        self.rotate(np.pi / 2)
                        continue
                    break
                rotate_angle = self.get_rotate_angle()
                self.rotate(rotate_angle)
            self.translate(cal_method)
            # 画出端点位置
            plt.plot(self.position[0], self.position[1], 'ro')
            x1 = self.position + self.vector * self.r * 3
            plt.plot(x1[0], x1[1], 'bo')  # 画出点1位置
            plt.show()
            rotate_times.append(j)
            # print('The difference between B and H', np.sum(np.abs(self.bk - self.PES.get_hess(self.position))))
            print('Bk: ', self.bk)
            self.position_list.append(self.position.copy())
            if (np.abs(self.f_r) < 0.1).all():  # 所有方向都相反
                if self.c < 0.0:
                    break
        if self.whether_print:
            print('time', rotate_times)
            # print(self.position/np.pi*180)
        return np.array(self.position_list), rotate_times

    def _update_bk(self, position_previous, force_previous, position_now, force_now):
        """
        BFGS更新Hessian矩阵
        """
        x0 = position_previous
        x1 = position_now
        f0 = force_previous
        f1 = force_now
        if (np.abs(x1 - x0) < self.min_value).all():
            return
        # BFGS校正
        delta_x = (x1 - x0).reshape((-1, 1))
        delta_f = (f1 - f0).reshape((-1, 1))
        # if delta_g.T.dot(delta_x) > 0:  # 如果该方向梯度增加，则更新拟Hessian矩阵B
        self.bk = (np.eye(self.n) - delta_x.dot(delta_f.T) /
                   delta_f.T.dot(delta_x)).dot(self.bk).dot(np.eye(self.n) -
                                                            delta_f.dot(delta_x.T) / delta_f.T.dot(delta_x)) + \
                  delta_x.dot(delta_x.T) / delta_f.T.dot(delta_x)


def atest_1():
    # 随机选取位置测试
    np.random.seed(10)
    for i in range(1, 5):
        plt.figure(i)
        ini_position = (np.random.rand(2) - 0.5)
        # ini_position = np.array([0.042, 0.4])
        angle = np.pi / 180 * 3
        ini_vector = [np.cos(angle), np.sin(angle)]
        PES = SimpleSurface()
        d = DimerQs(PES, 2, ini_position, ini_vector, whether_print=True)
        d.PES.show_surface_2d(-0.5, 0.5)
        # d.PES.show_surface_2d(-5, 5)
        position_d, times_d = d.work(2)  # 得到dimer运行轨迹和每一次的旋转数
        d.PES.show_point_2d(position_d)

        plt.title('Dimer rotates %d times and run %d times \n '
                  % (sum(times_d), len(times_d)) + str(d.translate_situation))
        # plt.title('Dimer rotates %d times and run %d times \n '
        #           % (sum(times_d), len(times_d)))
        print('\n')

        x1 = d.position + d.vector * d.r
        plt.plot(x1[0], x1[1], 'ko')  # 画出点1位置


def atest_extreme1():
    # 第二个函数极端情况（极大极小）测试
    np.random.seed(10)

    ini_position = [-0.5, 0.0]
    angle = np.pi / 180 * 90
    ini_vector = [np.cos(angle), np.sin(angle)]
    PES = SimpleSurface()
    d = DimerQs(PES, 2, ini_position, ini_vector, whether_print=True)
    d.PES.show_surface_2d(-0.5, 0.5)
    # d.PES.show_surface_2d(-5, 5)
    position_d, times_d = d.work(1)  # 得到dimer运行轨迹和每一次的旋转数
    d.PES.show_point_2d(position_d)

    plt.title('Dimer rotates %d times and run %d times \n '
              % (sum(times_d), len(times_d)))
    print('\n')

    x1 = d.position + d.vector * d.r
    plt.plot(x1[0], x1[1], 'ko')  # 画出点1位置


def atest_extreme():
    # 极端情况（极大极小）测试
    np.random.seed(7)
    ini_position1 = np.array([[-4.3, 1.6], [-np.pi / 2 * 1.1, -np.pi / 2]], 'f')
    for i in range(1, 3):
        plt.figure(2 * i)
        ini_position = ini_position1[i - 1]
        ini_vector = np.random.rand(2)
        PES = SimpleSurface()
        d = Dimer(PES, 2, ini_position, ini_vector, whether_print=True)
        position_d, times_d, f_r_list, f_parallel_list, c_list = d.work_list()  # 得到dimer运行轨迹和每一次的旋转数
        d.PES.show_point_2d(position_d)
        d.PES.show_surface_2d(-5, 5)
        plt.title('Dimer rotates %d times and run %d times \n '
                  % (sum(times_d), len(times_d)))
        x1 = d.position + d.vector * d.r
        plt.plot(x1[0], x1[1], 'ko')  # 画出点1位置

        plt.figure(2 * i + 1)
        plt.plot(c_list)
        # plt.figure(2 * i + 2)
        # plt.plot(f_parallel_list)
        # plt.figure(2 * i + 3)
        # plt.plot(f_r_list)


def data_to_excel(data_in):
    # prepare for data
    data = np.array(data_in)
    data_df = pd.DataFrame(data)

    # change the index and column name
    data_df.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    data_df.index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    # create and writer pd.DataFrame to excel
    writer = pd.ExcelWriter('Save_Excel.xlsx')
    data_df.to_excel(writer, 'page_2', float_format='%d')  # float_format 控制精度
    writer.save()  # 如果有同名文件，该操作会覆盖原文件

if __name__ == "__main__":
    # atest_extreme1()
    atest_1()
