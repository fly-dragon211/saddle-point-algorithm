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
    def __init__(self, PES, n=2, ini_position=None,
                 ini_vector=None, ini_velocity=None,
                 whether_print=False):
        # 是否打印中间数据
        self.whether_print = whether_print
        # 旋转力收敛值
        self.min_vertical_force = 1e-2
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
        self.vector[:] = ini_vector if ini_vector is not None else np.random.rand(n)-0.5
        self.position[:] = ini_position if ini_position is not None else np.random.rand(n)
        self.v[:] = ini_velocity if ini_velocity is not None else self.vector
        # 向量归一化
        # 这里用=/会报错
        self.vector = self.vector / np.linalg.norm(self.vector)
        # 曲率
        self.c = 0
        self.delta_angle = np.pi / 180
        self.timer = 0.06  # dimer步进时间
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
        new_vector = self.vector*np.cos(self.delta_angle) + self.normal*np.sin(self.delta_angle)
        # 微旋转后的dimer受力
        new_f1 = self.force(self.position+new_vector * self.r)
        new_f2 = 2 * self.f_r - new_f1
        delta_new_f = (new_f1-new_f2) - np.dot(new_f1-new_f2, new_vector) * new_vector
        temp = f_abs*np.cos(2*self.delta_angle)-np.dot(delta_new_f, delta_f) / f_abs
        angle = np.arctan((np.sin(2*self.delta_angle)*f_abs) / temp) / 2
        return angle

    def get_rotate_angle_v2(self):
        """
        计算dimer的旋转角
        """
        self._update_f()
        delta_f = (self.f1-self.f2) - np.dot(np.dot(self.f1-self.f2, self.vector), self.vector)
        # 力大小
        f_abs = np.linalg.norm(delta_f)
        if f_abs < self.min_value:
            return 0
        # 垂直向量判断
        self._update_normal()
        if np.isnan(self.normal[0]):
            return 0
        # 这里是单位向量
        new_vector = self.vector*np.cos(self.delta_angle) + self.normal*np.sin(self.delta_angle)
        # 微旋转后的dimer受力
        new_f1 = self.force(self.position+new_vector * self.r)
        new_f2 = 2 * self.f_r - new_f1
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
        self._update_f()
        # 计算曲率
        self._update_c()

    def translate_v1(self):
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
                self.v = delta_v * (1+np.dot(delta_v, self.v)/np.dot(delta_v, delta_v))
        self.position += (self.v * self.timer)
        # 计算新点相关数据
        self._update_f()

    def translate_v2(self):
        """
        移动dimer, BFGS法，不太对
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
        delta_v = f_to_saddle.reshape((-1, 1))

        position_pre = self.position.copy()
        force_pre = self.f_r.copy()
        self.position += np.linalg.norm(self.bk.dot(delta_v).flatten())*delta_v.flatten()*self.timer
        # 计算新点相关数据
        self._update_f()
        self._update_bk(position_pre, force_pre)

    def translate_v3(self):
        """
        移动dimer, 线性探测法
        """
        # plt.plot(self.position[0], self.position[1])
        self.angle = 0
        # dimer合力
        f_r = self.f_r
        # dimer平行力
        f_parallel = np.dot(self.vector, f_r) * self.vector
        # dimer指向鞍点力
        timer_alpha = 1  # self.timer的倍数
        if self.c < 0:
            f_to_saddle = f_r - f_parallel * 2  # 平行力反向
            if np.linalg.norm(self.force(self.position - f_to_saddle*0.01)) < np.linalg.norm(self.f_r):
                # 后面是极值，需要跳出
                value_list = [self.get_value(self.position),
                              self.get_value(self.position + f_to_saddle*self.timer)]
                m = 2
                while m < 20:
                    value_1 = self.get_value(self.position + f_to_saddle*(self.timer*m))
                    if (value_list[-1] - value_list[-2]) * (value_1 - value_list[-1]) <= 0:
                        break
                    value_list.append(value_1)
                    m += 2
                timer_alpha = m

            elif np.linalg.norm(self.f_r) < 0.5:
                # 鞍点附近, 一维线性搜索，步长调整
                m = 5
                while True:
                    f_r_next = self.force(self.position + f_to_saddle * self.timer * m)
                    if np.linalg.norm(self.f_r) > np.linalg.norm(f_r_next) or m <= 1:
                        self.f_r = f_r_next  # 目标点梯度已经计算过了
                        break
                    m -= 1
                timer_alpha = m
            else:
                timer_alpha = 1

        else:
            f_to_saddle = - f_parallel
            # 一维搜索取值较小的的点
            m = 11
            value_now = self.get_value(self.position)
            while True:
                if value_now > self.get_value(self.position + f_to_saddle * self.timer * m) or m <= 2:
                    break
                m -= 2
            timer_alpha = m

        self.v = f_to_saddle
        self.position += (self.v * self.timer * timer_alpha)
        # 计算新点相关数据
        self._update_f()

    def work(self):
        self._update_all()
        # 旋转到曲率最小
        times = []
        for i in range(1200):
            pre_c = self.c
            for j in range(20):
                rotate_angle = self.get_rotate_angle()
                self.rotate(rotate_angle)
                if self.whether_print:
                    print('rotated force: ', np.linalg.norm(self.f_vertical))
                # 垂直力大小
                if np.linalg.norm(self.vertical_force(self.f1 - self.f2, self.vector)) < self.min_vertical_force:
                    if self.c > pre_c and self.c > 0:  # 如果曲率上升，旋转pi/2 + angle, 重要
                        self.rotate(np.pi / 2)
                        continue
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
            # print(self.position/np.pi*180)
        # self.PES.show_point_2d(np.array(self.position_list))
        # self.PES.show_surface_2d(-5, 5)
        # plt.title('It rotates %d times and run %d times' % (sum(times), len(times)))
        # self.PES.show()
        return np.array(self.position_list), times

    def work_list(self):
        self._update_all()
        # 旋转到曲率最小
        times = []
        f_r_list = []
        f_parallel_list = []
        c_list = []

        for i in range(1200):
            pre_c = self.c
            for j in range(200):
                rotate_angle = self.get_rotate_angle()
                self.rotate(rotate_angle)
                if self.whether_print:
                    print('rotated force: ', np.linalg.norm(self.f_vertical))
                # 垂直力大小
                if np.linalg.norm(self.vertical_force(self.f1 - self.f2, self.vector)) < self.min_vertical_force:
                    if self.c > pre_c:  # 如果曲率上升，旋转pi/2 + angle, 重要
                        self.rotate(np.pi / 2)
                        continue
                    times.append(j)
                    break
                elif j == 199:
                    times.append(j)
            self.translate_v3()
            if self.whether_print:
                print('parallel force', np.dot(self.vector, self.f_r) * self.vector,
                      'verticle force', '\n')
                f_r_list.append(np.linalg.norm(self.f_r))
                f_parallel_list.append(np.linalg.norm(np.dot(self.vector, self.f_r) * self.vector, ord=1))
                c_list.append(self.c)

            self.position_list.append(self.position.copy())
            if (np.abs(self.f_r) < 0.1).all():  # 所有方向都相反
                if self.c < 0.0:
                    break
        if self.whether_print:
            print('time', times)
            # print(self.position/np.pi*180)
        # self.PES.show_point_2d(np.array(self.position_list))
        # self.PES.show_surface_2d(-5, 5)
        # plt.title('It rotates %d times and run %d times' % (sum(times), len(times)))
        # self.PES.show()
        return np.array(self.position_list), times, f_r_list, f_parallel_list, c_list

    def _update_c(self):
        """
        计算曲率
        """
        self.c = -np.dot(self.f1-self.f2, self.vector) / self.r / 2
        if self.whether_print:
            print('curvature: ', self.c)

    def _update_normal(self):
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

    def _update_bk(self, position_previous, force_previous):
        """
        BFGS更新Hessian矩阵
        """
        x0 = position_previous
        x1 = self.position
        f0 = force_previous
        f1 = self.f_r
        # BFGS校正
        delta_x = (x1 - x0).reshape((-1, 1))
        delta_f = (f1 - f0).reshape((-1, 1))
        # if delta_g.T.dot(delta_x) > 0:  # 如果该方向梯度增加，则更新拟Hessian矩阵B
        self.bk = (np.eye(self.n) - delta_x.dot(delta_f.T) /
              delta_f.T.dot(delta_x)).dot(self.bk).dot(np.eye(self.n) -
              delta_f.dot(delta_x.T) / delta_f.T.dot(delta_x)) + \
              delta_x.dot(delta_x.T) / delta_f.T.dot(delta_x)

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
        theta_f = delta_f/f_abs
        # 计算\phi_1
        c_new = -np.dot(self.f1 - self.f_r, self.vector) / self.r
        partial_c = 2*np.dot(self.f1 - self.f_r, theta_f) / self.r
        angle_1 = 0.5 * np.arctan(partial_c/np.abs(c_new)/2)

        # 这里是单位向量
        new_vector = self.vector*np.cos(angle_1) + self.normal*np.sin(angle_1)
        # 微旋转后的dimer受力
        new_f1 = self.force(self.position+new_vector * self.r)
        new_f2 = 2 * self.f_r - new_f1
        delta_new_f = (new_f1-new_f2) - np.dot(new_f1-new_f2, new_vector) * new_vector
        temp = f_abs*np.cos(2*angle_1)-np.dot(delta_new_f, delta_f) / f_abs
        angle = np.arctan((np.sin(2*angle_1)*f_abs) / temp) / 2
        # c_angle = -np.dot(new_f1 - self.f_r, new_vector) / self.r  # 旋转后曲率
        return angle

    def translate_v1(self):
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
                self.v = delta_v * (1+np.dot(delta_v, self.v)/np.dot(delta_v, delta_v))
        self.position += (self.v * self.timer)
        # 计算新点相关数据
        self._update_f()

    def work(self):
        self._update_all()
        # 旋转到曲率最小
        times = []
        for i in range(1200):
            for j in range(20):
                rotate_angle = self.get_rotate_angle()
                pre_c = self.c
                self.rotate(rotate_angle)
                print('rotate_angle', rotate_angle * 180 / np.pi,
                      'pre_c: %f, c: %f' % (pre_c, self.c))
                if pre_c < self.c and self.c > 0:
                    self.rotate(np.pi / 2)
                    print('rotate_angle', rotate_angle * 180 / np.pi,
                          'pre_c: %f, c: %f' % (pre_c, self.c))
                # 垂直力大小
                if np.abs(rotate_angle) < np.pi/180 * 1:
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
            # print(self.position/np.pi*180)
        # self.PES.show_point_2d(np.array(self.position_list))
        # self.PES.show_surface_2d(-5, 5)
        # plt.title('It rotates %d times and run %d times' % (sum(times), len(times)))
        # self.PES.show()
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
            self.translate_v1()
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


def atest_1():
    # 随机选取位置测试
    np.random.seed(10)
    for i in range(1, 10):
        plt.figure(i)
        ini_position = (np.random.rand(2) - 0.5)
        angle = np.pi / 180 * 3
        ini_vector = [np.cos(angle), np.sin(angle)]
        ini_vector = np.random.rand(2)
        PES = SimpleSurface()
        d = DimerRo(PES, 2, ini_position, ini_vector, whether_print=True)
        d.PES.show_surface_2d(-0.5, 0.5)
        position_d, times_d = d.work()  # 得到dimer运行轨迹和每一次的旋转数
        d.PES.show_point_2d(position_d)

        plt.title('Dimer rotates %d times and run %d times \n '
                  % (sum(times_d), len(times_d)))

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

        # plt.figure(2 * i + 1)
        # plt.plot(c_list)
        # plt.figure(2 * i + 2)
        # plt.plot(f_parallel_list)
        # plt.figure(2 * i + 3)
        # plt.plot(f_r_list)

if __name__ == "__main__":
    atest_1()

