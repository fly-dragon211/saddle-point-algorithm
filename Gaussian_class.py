#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Gaussian_class.py
# @Author: Fly_dragon
# @Date  : 2020/3/10
# @Desc  : 修改后结合高斯程序的dimer算法


import numpy as np
import pandas as pd
import os
import re
import time

from dimer_vertical import Dimer, DimerRo, DimerQs
from newton import Newton


class GaussianFile:
    """
    gaussian文件处理，必须为笛卡尔坐标，分子字母大写
    """
    def __init__(self):
        self.cal_num = [0, 0]  # the calculate number of energy and gradient
        self.elements = []
        # out文件，Force比较复杂代码在下面
        self.pattern_spe = re.compile(r'SCF Done:  E\(.+?\) =(.+?)A.U')
        # gjf文件
        self.pattern_coordinates = re.compile(r'-?\d\.\d{8}')
        self.pattern_elements = re.compile(r'([A-Z][a-z]?) *?-?\d\.\d{8}')
        self.file_head = ['%mem=500MB\n',
                          '%nprocshared=4\n',
                          '#p force pm6 scf=xqc\n',
                          '\n',
                          'title\n',
                          '\n']
        self.file_head_noforce = ['%mem=500MB\n',
                                   '%nprocshared=4\n',
                                   '#pm6\n',
                                   '\n',
                                   'title\n',
                                   '\n',]
        self.file_head_noforce1 = ['%mem=500MB\n', '%nprocshared=4\n', '# B3LYP/6-31G(d) sp\n',
                          '\n', 'title\n', '\n']
        self.file_head_hess = ['%mem=500MB\n',
                                   '%nprocshared=4\n',
                                   '#p freq pm6\n',
                                   '\n',
                                   'title\n',
                                   '\n']
        self.file_electron = '0 1\n'

        self.file_rear = ['\n1 3 1.0 4 1.0 5 1.0\n', '2\n', '3\n', '4\n', '5\n']

    def generate_input(self, file_path, coordinates, elements, cal_type=0):
        """
        生成输入文件
        cal_type: 0 计算势能，1 计算受力(梯度)，2 计算Hessian矩阵
        """
        with open(file_path, 'w+') as f:
            if cal_type == 1:
                for line in self.file_head:
                    f.writelines(line)
            elif cal_type == 0:
                for line in self.file_head_noforce:
                    f.writelines(line)
            elif cal_type == 2:
                for line in self.file_head_hess:
                    f.writelines(line)
            f.write(self.file_electron)
            for i in range(len(coordinates)):
                f.writelines(' '+elements[i]+'  '+'%.8f' % coordinates[i][0]+'  '
                             + '%.8f' % coordinates[i][1]+'  '+'%.8f' % coordinates[i][2]+'\n')
            # for line in self.file_rear:
            #     f.writelines(line)
            f.writelines('\n')
        return

    def gjf_read(self, file_path):
        """
        初始化读取
        :return: 元素和坐标列表
        """
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                text = f.read()
        else:
            print('文件不存在! :', file_path)
            return
        # 修改电荷数和自旋多重度
        p = re.compile(r'\n(-?\d \d\n)')
        electron = p.findall(text)
        if len(electron) > 1:
            input('有多个自旋多重度。')
        self.file_electron = electron[0]
        # 修改元素文件
        self.elements = re.findall(self.pattern_elements, text)
        coordinates = np.array(re.findall(self.pattern_coordinates, text), dtype=float)
        coordinates = coordinates.reshape((-1, 3))

        return coordinates

    def _get_scf(self, file_path):
        """
        从out文件中得到势能
        :param file_path:
        :return: SCF Done energy
        """
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                text = f.read()
        else:
            print('文件不存在! :', file_path)
            return
        energy = re.findall(self.pattern_spe, text)
        if len(energy) != 1:
            print('势能不唯一')
            return
        return float(energy[0])

    def get_force(self, positions):
        """
        得到受力(梯度)
        :param positions:  x * 1 矩阵
        :return:
        """
        coordinates = positions.reshape((-1, 3))
        def _store_forces(f):
            p = re.compile(r'-?\d\.\d{9}')
            while True:
                line = f.readline()
                if re.search(r'------------', line) or (not line):
                    break
            forces = []
            while True:
                line = f.readline()
                #         print(line)
                if re.search(r'------------', line) or (not line):  # 这里要注意负号也为'-'
                    break
                r = p.findall(line)
                forces.extend(r)
            return forces

        self.generate_input('cache.gjf', coordinates, self.elements, cal_type=1)
        print('run gaussian to get force')
        self.run_gaussian('cache.gjf')
        # 读取第一个forces
        with open('cache.out', 'r', encoding='utf-8') as f:
            forces = []
            while True:
                line = f.readline()
                if not line:
                    break
                if re.search(r'Forces', line):
                    forces.extend(_store_forces(f))
                    break
            self.cal_num[1] += 1
            return np.array(forces, 'f')

    def get_hess(self, positions):
        """
        得到hessian矩阵
        :param positions:  1 * x 矩阵
        :return:
        """
        coordinates = positions.reshape((-1, 3))

        def store_hessian(f):
            p = re.compile(r'-?\d\.\d{6}D[+-]\d\d')
            forces = []
            while True:
                line = f.readline()
                #         print(line)
                if re.search(r'Leave Link', line) or (not line):  # 这里要注意负号也为'-'
                    break
                r = p.findall(line)
                forces.extend(r)
            return forces

        def vector_to_hess(vector, n):
            # 把高斯中读取的vector转换成Hessian，注意有坑
            hessian = np.zeros((n, n), 'f')
            num_iterate = 0
            for i in range(0, n, 5):
                for L in range(i, n):
                    for R in range(i, i + 5):
                        if R > L or R >= n:
                            break
                        hessian[L, R] = vector[num_iterate]
                        num_iterate += 1
            return hessian

        self.generate_input('cache.gjf', coordinates, self.elements, cal_type=2)
        print('run gaussian to get hess')
        self.run_gaussian('cache.gjf')

        # 读取第一个Hessian
        n = positions.size
        with open('cache.out', 'r', encoding='utf-8') as f:
            vector_h = []
            while True:
                line = f.readline()
                #         print(line)
                if not line:
                    break
                if re.search(r'Forces', line):
                    vector_h.extend(store_hessian(f))
                    break
            for i in range(len(vector_h)):  # 把字符串里的D替代成E
                vector_h[i] = float(vector_h[i].replace('D', 'E'))
            # 对向量进行重组
            hessian = vector_to_hess(vector_h, n)
            hessian += hessian.T - np.diag(hessian.diagonal())  # 转换成对称矩阵

        return hessian

    @staticmethod
    def run_gaussian(input_file):
        """
        计算
        """
        os.system('g09 '+input_file)
        print('done')

    def get_value(self, positions):
        """
        得到势能
        :param positions: x * 1 矩阵
        :return:
        """
        coordinates = positions.reshape((-1, 3))
        self.generate_input('cache.gjf', coordinates, self.elements)
        print('run gaussian to get value')
        self.run_gaussian('cache.gjf')
        energy = self._get_scf('cache.out')
        self.cal_num[0] += 1
        return energy


class DimerGaussian(DimerQs):
    def __init__(self, PES, n, ini_position=None,
                 ini_vector=None, ini_velocity=None,
                 whether_print=False):
        super().__init__(PES, n, ini_position,
                 ini_vector, ini_velocity,
                 whether_print)
        self.min_vertical_force = 0.01  # 旋转力收敛值
        self.min_value = 1e-20
        self.timer = 0.02  # dimer步进时间
        self.deltax_max = 0.3
        self.n = n
        self.r = 0.005

    def get_value(self, position):
        """
        获取值
        """
        return self.PES.get_value(position)

    def force(self, position):
        """
        计算像点受力，一阶梯度
        """
        return self.PES.get_force(position)

    def work(self, result_path='', cal_method=0, translate_time_max=30):
        self._update_all()
        # 旋转到曲率最小
        times = []
        for i in range(translate_time_max):
            pre_c = self.c
            for j in range(0, 20):
                if self.whether_print:
                    print('rotated force: ', np.linalg.norm(self.f_vertical))
                self.store_information()  # 存储dimer信息
                # 垂直力大小
                if np.linalg.norm(self.vertical_force(self.f1 - self.f2, self.vector)) < self.min_vertical_force and j>0:
                    if self.c > pre_c and self.c > 0:  # 如果曲率上升，旋转pi/2 + angle, 重要
                        self.rotate(np.pi / 2)
                        continue
                    break
                rotate_angle = self.get_rotate_angle()
                self.rotate(rotate_angle)
            self.translate(cal_method)
            times.append(j)
            # 把移动后的分子存储起来
            self.PES.generate_input(result_path + 'test_dimer' + str(i) + '.gjf', self.position.reshape((-1, 3)),
                                    self.PES.elements)

            if (np.abs(self.f_r) < 0.1).all():  # 所有方向都相反
                if self.c < 0.0:
                    if self.whether_print:
                        print('step: ', i)
                    break
        if self.whether_print:
            print('time', times)
            print(self.position / np.pi * 180)
        return times

    def store_information(self):
        # 存储dimer信息到 information.txt
        with open('information_dimer.txt', 'a+') as f:
            f.write('rotated force: ' + str(np.linalg.norm(self.f_vertical))
                    + 'vector: ' + str(self.vector) + '\n')
            f.write('curvature: ' + str(self.c))


class DimerRoGaussian(DimerRo):
    def __init__(self, PES, n, ini_position=None,
                 ini_vector=None, ini_velocity=None,
                 whether_print=False):
        super().__init__(PES, n, ini_position,
                 ini_vector, ini_velocity,
                 whether_print)
        self.min_vertical_force = 0.02  # 旋转力收敛值
        self.min_value = 1e-20
        self.timer = 0.16  # dimer步进时间
        self.n = n
        self.r = 0.005

    def get_value(self, position):
        """
        获取值
        """
        return self.PES.get_value(position)

    def force(self, position):
        """
        计算像点受力，一阶梯度
        """
        return self.PES.get_force(position)

    def work(self, result_path=''):
        self._update_all()
        # 旋转到曲率最小
        times = []
        for i in range(200):
            pre_c = self.c
            for j in range(1, 20):
                rotate_angle = self.get_rotate_angle()
                self.rotate(rotate_angle)
                self.store_information(rotate_angle)  # 存储dimer信息
                if pre_c < self.c and self.c > 0:
                    self.rotate(np.pi / 2)
                    print('rotate_angle', rotate_angle * 180 / np.pi,
                          'pre_c: %f, c: %f' % (pre_c, self.c))
                # 旋转角度小于某个值
                if np.abs(rotate_angle) < np.pi/180 * 5:
                    break
            self.translate(3)
            times.append(j)
            # 把移动后的分子存储起来
            self.PES.generate_input(result_path + 'test_dimer' + str(i) + '.gjf', self.position.reshape((-1, 3)),
                                    self.PES.elements)

            if (np.abs(self.f_r) < 0.1).all():  # 所有方向都相反
                if self.c < 0.0:
                    if self.whether_print:
                        print('step: ', i)
                    break
        if self.whether_print:
            print('time', times)
            print(self.position / np.pi * 180)
        return times

    def store_information(self, rotate_angle):
        # 存储dimer信息到 information.txt
        with open('information_dimer.txt', 'a+') as f:
            f.write('rotated angle: ' + str(rotate_angle * 180 / np.pi)
                    + 'vector: ' + str(self.vector) + '\n')
            f.write('curvature: ' + str(self.c))


class NewtonGaussian(Newton):
    def __init__(self, PES, ini_position):
        super().__init__(PES.get_value, PES.get_force, ini_position)
        self.PES = PES

    def bfgs_newton(self, hess_fun='None'):
        # 用BFGS拟牛顿法求解无约束问题，用B代表B^-1
        # x0是初始点，fun，gfun和hess分别是目标函数值，梯度，海森矩阵的函数(可不写)
        fun = self.fun
        gfun = self.gfun
        x0 = self.x0
        n = len(x0)  # 变量个数
        maxk = 1e5  # 最大迭代次数
        epsilon = 1e-4
        delta = 0.6  # 参数很重要，越大则单步可以搜索的范围越远(太小时收敛过慢)
        sigma = 0.4
        k = 0
        # 初始B
        if type(hess_fun) == str:
            Bk = np.linalg.inv(np.eye(n))
        else:
            Bk = np.linalg.inv(hess_fun(x0))
        # 在牛顿方向dk做一维搜索
        while k < maxk:
            gk = gfun(x0) if k == 0 else gkx1  # 后面计算过gfun(x0)，减少一次计算
            fx0 = fun(x0)
            if np.linalg.norm(gk) <= epsilon:
                break
            dk = -np.dot(Bk, gk)
            m = 0
            while m < 20:
                if fun(x0 + sigma**m * dk) <= fx0 + delta * sigma**m * np.dot(gk, dk):
                    break
                m += 1
            alpha = sigma**m
            x1 = x0 + alpha * dk
            # BFGS校正
            gkx1 = gfun(x1)
            delta_x = (x1 - x0).reshape((-1, 1))
            delta_g = (gkx1 - gk).reshape((-1, 1))
            # if delta_g.T.dot(delta_x) > 0:  # 如果该方向梯度增加，则更新拟Hessian矩阵B
            Bk = (np.eye(n) - delta_x.dot(delta_g.T) /
                  delta_g.T.dot(delta_x)).dot(Bk).dot(np.eye(n) - delta_g.dot(delta_x.T)/delta_g.T.dot(delta_x)) +\
                delta_x.dot(delta_x.T) / delta_g.T.dot(delta_x)

            # 测试Dk与hess(x0)差别
            # print('The difference between B and H', np.sum(np.abs(Bk - hess(x0))))
            k += 1
            x0 = x1
            # 把移动后的分子存储起来
            self.PES.generate_input('test_bfgs' + str(k) + '.gjf', x0.reshape((-1, 3)),
                                    self.PES.elements)
            # 存储信息到 information.txt
            with open('information_bfgs.txt', 'a+') as f:
                f.write('iterate_num: ' + str(k) + 'force: ' + str(np.linalg.norm(gk)))

        return k


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")


def baker_test_dimer(baker_folder_path, result_folder_path, cal_method=0):
    translate_time_max = 40
    np.random.seed(2)
    mkdir(result_folder_path)
    cal_nums = []
    cal_time = []
    for i in range(1, 26):
        baker_path = baker_folder_path + str(i)
        result_path = result_folder_path + str(i)
        # 进入result/i文件夹，不存在则创建
        mkdir(result_path)
        os.chdir(result_path)
        g = GaussianFile()
        ini_position = g.gjf_read(baker_path + r'.gjf').reshape((1, -1))
        # 算法测试
        d = DimerGaussian(g, ini_position.size, ini_position=ini_position)
        cal_time_begin = time.time()
        times_d = d.work(cal_method=cal_method, translate_time_max=translate_time_max)
        if len(times_d) >= translate_time_max - 1:
            g.cal_num[1] = 1000

        cal_time.append(time.time()-cal_time_begin)
        cal_nums.append(g.cal_num.copy())
        cal_nums[-1].append(float(d.get_value(d.position)))  # 记录计算过渡态能量
        # 存储信息到 result.txt
        with open(result_folder_path+'run_result.txt', 'a+') as f:
            f.write(str(i) + ' ' * 2 + 'Dimer rotates %d times, run %d times. %f second. E_final %f\n '
                    % (sum(times_d), len(times_d), cal_time[-1], d.get_value(d.position)))
            g.cal_num[0] -= 1
            f.write('energy and gradient cal:' + str(g.cal_num) +
                    ' translate situation:' + str(d.translate_situation) + '\n'*2)

    with open(result_folder_path+'result.txt', 'a+') as f:
        f.write('# method: v' + str(cal_method) + '\n')
        f.write(str(cal_nums) + '\n')

    return cal_nums


def baker_test_Newton():
    np.random.seed(2)
    baker_folder_path = r'D:\graduate_project\transition_state\saddle-point-algorithm\baker_molcule' + '\\'
    result_folder_path = r'D:\graduate_project\transition_state\result_v3' + '\\'
    mkdir(result_folder_path)
    cal_nums = []
    cal_time = []
    for i in range(1, 4):
        baker_path = baker_folder_path + str(i)
        result_path = result_folder_path + str(i)
        # 进入result/i文件夹，不存在则创建
        mkdir(result_path)
        os.chdir(result_path)
        g = GaussianFile()
        ini_position = g.gjf_read(baker_path + r'.gjf').reshape((1, -1))
        # 算法测试
        # bfgs algorithm
        Ne_bfgs = NewtonGaussian(g, ini_position)
        cal_time_begin = time.time()
        k = Ne_bfgs.bfgs_newton(hess_fun=g.get_hess)

        cal_time.append(time.time()-cal_time_begin)
        cal_nums.append(g.cal_num.copy())
        # 存储信息到 result.txt
        with open(result_folder_path+'result.txt', 'a+') as f:
            f.write(str(i)+' '*2 + 'Newton run %d times. %f second\n '
                    % (k, cal_time[-1]))

    with open(result_folder_path+'result.txt', 'a+') as f:
        f.write(str(cal_nums))

    return cal_nums


def my_test():
    path = r'D:\graduate_project\transition_state\dimer_test_co2'
    os.chdir(path)
    g = GaussianFile()
    ini_position = g.gjf_read(r'co2.gjf').reshape((1, -1))
    np.random.seed(2)
    # bfgs algorithm
    Ne_bfgs = NewtonGaussian(g, ini_position)
    Ne_bfgs.bfgs_newton(hess_fun=g.get_hess)
    # d = DimerGaussian(g, ini_position.size, ini_position=ini_position)
    # d.work()


if __name__ == '__main__':
    matrix_result = []
    baker_folder = r'D:\graduate_project\transition_state\saddle-point-algorithm\baker_molcule' + '\\'
    result_folder = r'D:\graduate_project\transition_state\result_3' + '\\'
    for cal_method in range(1, 4):
        cal_nums = baker_test_dimer(baker_folder, result_folder, cal_method)
        cal_nums = np.array(cal_nums, dtype=np.float)
        matrix_result.append(cal_nums)

    # write the data to excel
    writer = pd.ExcelWriter(result_folder + 'temp.xlsx')
    for i in range(1):
        a_df = pd.DataFrame(matrix_result[i])
        a_df.to_excel(writer, str(i + 1), float_format='%f')
    writer.save()
