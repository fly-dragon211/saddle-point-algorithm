#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : d_qN_method.py
# @Author: Fly_dragon
# @Date  : 2020/2/27
# @Desc  : Combine dimer and quasi-newton

import numpy as np
import matplotlib.pyplot as plt
import os
import re

from newton import Newton
from dimer_vertical import Dimer
from simplesurface import SimpleSurface
import Gaussian_class as Gc


def test_simple_surface():
    # the class of test surface
    PES = SimpleSurface()
    # dimer algorithm
    ini_position = (np.random.rand(2) - 0.5) * 10
    ini_vector = np.random.rand(2)
    d = Dimer(2, ini_position, ini_vector, whether_print=False)
    position_d, times_d = d.work()  # 得到dimer运行轨迹和每一次的旋转数
    # print('vector', d.vector)
    # quasi newton method
    qN = Newton(PES.get_value, PES.get_diff, [position_d[-1, 0], position_d[-1, 1]])
    position_qN, times_qN = qN.bfgs_newton(PES.get_hess)

    position = np.concatenate((position_d, position_qN), axis=0)
    PES.show_point_2d(position)
    plt.plot(position_d[-1, 0], position_d[-1, 1], 'ko')
    PES.show_surface_2d(-5, 5)
    plt.title('Dimer rotates %d times and run %d times \n '
              'Newton runs %d times' % (sum(times_d), len(times_d), times_qN))
    plt.show()


def test_gaussian():
    path = r'D:\graduate_project\transition_state\dimer_test_DA\qn-dimer'
    path_gjf = r'DA_reaction.gjf'
    os.chdir(path)
    # the class of test surface
    g = Gc.GaussianFile()
    # dimer algorithm
    ini_position = g.gjf_read(path_gjf).reshape((1, -1))
    d = Gc.DimerGaussian(g, ini_position.size, ini_position=ini_position)
    times_d = d.work()  # 得到dimer运行轨迹和每一次的旋转数
    # quasi newton method
    ini_position = g.gjf_read(r'test_dimer' + str(len(times_d)-1) + '.gjf').reshape((1, -1))
    qN = Gc.NewtonGaussian(g, ini_position)
    times_qN = qN.bfgs_newton(g.get_hess)

    print('Dimer rotates %d times and run %d times \n '
          'Newton runs %d times' % (sum(times_d), len(times_d), times_qN))


if __name__ == '__main__':
    path = r'D:\graduate_project\transition_state\dimer_test_DA\qn-dimer'
    path_gjf = r'DA_reaction.gjf'
    os.chdir(path)
    np.random.seed(2)
    g = Gc.GaussianFile()
    ini_position = g.gjf_read(r'test_dimer' + str(0) + '.gjf').reshape((1, -1))
    qN = Gc.NewtonGaussian(g, ini_position)
    times_qN = qN.bfgs_newton(g.get_hess)
    # for i in range(1, 7):
    #     plt.figure(i)
    #     test1()
