#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Gaussian_test.py
# @Author: Fly_dragon
# @Date  : 2020/3/10
# @Desc  : 修改后结合高斯程序的dimer算法



import numpy as np
import os
import re

from dimer_vertical import Dimer


class GaussianFile:
    """
    gaussian文件处理
    """
    def __init__(self):
        self.elements = []
        # out文件，Force比较复杂代码在下面
        self.pattern_spe = re.compile(r'SCF Done:  E\(.+?\) =(.+?)A.U')
        # gjf文件
        self.pattern_coordinates = re.compile(r'-?\d\.\d{8}')
        self.pattern_elements = re.compile(r'([A-Z][a-z]?) *?-?\d\.\d{8}')
        self.file_head = ['%mem=500MB\n',
                          '%nprocshared=4\n',
                          '#p force pm6\n',
                          '\n',
                          'title\n',
                          '\n',
                          '0 1\n']
        self.file_head_noforce = ['%mem=500MB\n',
                                   '%nprocshared=4\n',
                                   '#pm6\n',
                                   '\n',
                                   'title\n',
                                   '\n',
                                   '0 1\n']
        self.file_head_noforce1 = ['%mem=500MB\n', '%nprocshared=4\n', '# B3LYP/6-31G(d) sp\n',
                          '\n', 'title\n', '\n', '0 1\n']
        self.file_head_hess = ['%mem=500MB\n',
                                   '%nprocshared=4\n',
                                   '#p freq pm6\n',
                                   '\n',
                                   'title\n',
                                   '\n',
                                   '0 1\n']

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
        print('running gaussian: ', input_file)
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
        self.run_gaussian('cache.gjf')
        energy = self._get_scf('cache.out')
        return energy


class DimerGaussian(Dimer):
    def __init__(self, PES, n, ini_position=None,
                 ini_vector=None, ini_velocity=None,
                 whether_print=False):
        super().__init__(PES, n, ini_position,
                 ini_vector, ini_velocity,
                 whether_print)
        self.min_vertical_force = 0.05  # 旋转力收敛值
        self.min_value = 1e-20
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

    def work(self):
        self._update_all()
        # 旋转到曲率最小
        times = []
        for i in range(1200):
            pre_c = self.c
            for j in range(200):
                rotate_angle = self.get_rotate_angle()
                self.rotate(rotate_angle)
                if self.whether_print:
                    print('rotated force: ', self.f_rota)
                self.store_information()  # 存储dimer信息
                # 垂直力大小
                if np.linalg.norm(self.vertical_force(self.f1 - self.f2, self.vector)) < self.min_vertical_force:
                    if self.c > pre_c:  # 如果曲率上升，旋转pi/2 + angle, 重要
                        self.rotate(np.pi / 2)
                        continue
                    times.append(j)
                    break
                elif j == 199:
                    times.append(j)
            self.translate()
            # 把移动后的分子存储起来
            self.PES.generate_input('test' + str(i) + '.gjf', self.position.reshape((-1, 3)),
                                    self.PES.elements)

            if (np.abs(self.f_r) < 0.5).all():  # 所有方向都相反
                if self.c < 0.0:
                    if self.whether_print:
                        print('step: ', i)
                    break
        if self.whether_print:
            print('time', times)
            print(self.position / np.pi * 180)
        return

    def store_information(self):
        # 存储dimer信息到 information.txt
        with open('information.txt', 'a+') as f:
            f.write('rotated force: ' + str(self.f_rota)
                    + 'vector: ' + str(self.vector) + '\n')
            f.write('curvature: ' + str(self.c))


if __name__ == '__main__':
    path = r'D:\graduate_project\transition_state\dimer_test_co2'
    os.chdir(path)
    g = GaussianFile()
    ini_position = g.gjf_read(r'co2.gjf').reshape((1, -1))
    # hess = g.get_hess(ini_position)
    np.random.seed(2)
    d = DimerGaussian(g, ini_position.size, ini_position=ini_position)
    d.work()