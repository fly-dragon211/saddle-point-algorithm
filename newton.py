#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : newton.py
# @Author: Fly_dragon
# @Date  : 2020/2/22
# @Desc  :
import numpy as np
import matplotlib.pyplot as plt
from simplesurface import SimpleSurface


class Newton:

    def __init__(self, fun, gfun, x0):
        # x0是初始点，fun，gfun和hess分别是目标函数值，梯度
        self.fun = fun
        self.gfun = gfun
        self.x0 = x0
        return

    def damp_newton(self, hess):
        # 用阻尼牛顿法求解无约束问题
        # x0是初始点，fun，gfun和hess分别是目标函数值，梯度，海森矩阵的函数
        fun = self.fun
        gfun = self.gfun
        x0 = self.x0
        maxk = 500  # 最大迭代次数
        sigma = 0.55  # 非线性搜索中的σ因子
        delta = 0.4  # 非线性搜索中的δ因子
        k = 0  # 初始化迭代次数
        epsilon = 1e-5  # 设定迭代终止得的阈值
        x_store = [x0.copy()]

        while k < maxk:
            gk = gfun(x0)  # 计算梯度
            Gk = hess(x0)  # 计算海森矩阵
            dk = -1.0 * np.linalg.solve(Gk, gk)  # 相当于dk=-1.0*(Gk^-1)gk
            if np.linalg.norm(dk) < epsilon:
                break
            m = 0  # 初始化非线性搜索中的次数
            mk = 0  # 用于存放非线性搜索得到的最小非负整数
            while m < 20:
                if fun(x0 + sigma ** m * dk) < fun(x0) + delta * sigma ** m * np.dot(gk, dk):
                    mk = m
                    break
                m += 1
            x0 += sigma ** mk * dk  # 更新x的值
            k += 1  # 进入下一次循环

            x_store.append(x0.copy())
        x_store = np.array(x_store)
        return x0, x_store, k

    def dfp_newton(self, hess='None'):
        # 用DFP拟牛顿法求解无约束问题
        # x0是初始点，fun，gfun和hess分别是目标函数值，梯度，海森矩阵的函数(可不写)
        fun = self.fun
        gfun = self.gfun
        x0 = self.x0
        n = len(x0)  # 变量个数
        maxk = 1e5  # 最大迭代次数
        epsilon = 1e-5
        delta = 0.6  # 参数很重要，越大则单步可以走的范围越远(太小时收敛过慢)
        sigma = 0.4
        k = 0
        x_store = [x0.copy()]
        # 初始D
        if type(hess) == str:
            Dk = np.linalg.inv(np.eye(n))
        else:
            Dk = np.linalg.inv(hess(x0))
        # 在牛顿方向dk做一维搜索
        while k < maxk:
            gk = gfun(x0)
            if np.linalg.norm(gk) <= epsilon:
                break
            dk = -np.dot(Dk, gk)
            m = 0
            while m < 20:
                if fun(x0 + sigma**m * dk) <= fun(x0) + delta * sigma**m * np.dot(gk, dk):
                    break
                m += 1
            alpha = sigma**m
            x1 = x0 + alpha * dk
            # DFP校正
            delta_x = (x1 - x0).reshape((-1, 1))
            delta_g = (gfun(x1) - gk).reshape((-1, 1))
            # if delta_g.T.dot(delta_x) > 0:  # 如果该方向梯度增加，则更新拟Hessian矩阵D
            Dk = Dk + delta_x.dot(delta_x.T) / delta_x.T.dot(delta_g) - \
                Dk.dot(delta_g).dot(delta_g.T).dot(Dk) / delta_g.T.dot(Dk).dot(delta_g)

            # # 测试Dk与hess(x0)差别
            # print('The difference between D and H', np.sum(np.abs(Dk - hess(x0))))
            k += 1
            x0 = x1
            x_store.append(x0.copy())

        x_store = np.array(x_store)
        return x0, x_store, k

    def bfgs_newton(self, hess='None'):
        # 用BFGS拟牛顿法求解无约束问题，用B代表B^-1
        # x0是初始点，fun，gfun和hess分别是目标函数值，梯度，海森矩阵的函数(可不写)
        fun = self.fun
        gfun = self.gfun
        x0 = self.x0
        n = len(x0)  # 变量个数
        maxk = 1e5  # 最大迭代次数
        epsilon = 1e-5
        delta = 0.55  # 参数很重要，越大则单步可以搜索的范围越远(太小时收敛过慢)
        sigma = 0.4
        k = 0
        x_store = [x0.copy()]
        # 初始B
        if type(hess) == str:
            Bk = np.linalg.inv(np.eye(n))
        else:
            Bk = np.linalg.inv(hess(x0))
        # 在牛顿方向dk做一维搜索
        while k < maxk:
            gk = gfun(x0)
            if np.linalg.norm(gk) <= epsilon:
                break
            dk = -np.dot(Bk, gk)
            m = 0
            while m < 20:
                if fun(x0 + sigma**m * dk) <= fun(x0) + delta * sigma**m * np.dot(gk, dk):
                    break
                m += 1
            alpha = sigma**m
            x1 = x0 + alpha * dk
            # BFGS校正
            delta_x = (x1 - x0).reshape((-1, 1))
            delta_g = (gfun(x1) - gk).reshape((-1, 1))
            # if delta_g.T.dot(delta_x) > 0:  # 如果该方向梯度增加，则更新拟Hessian矩阵B
            Bk = (np.eye(n) - delta_x.dot(delta_g.T) /
                  delta_g.T.dot(delta_x)).dot(Bk).dot(np.eye(n) - delta_g.dot(delta_x.T)/delta_g.T.dot(delta_x)) +\
                delta_x.dot(delta_x.T) / delta_g.T.dot(delta_x)

            # # 测试Dk与hess(x0)差别
            # print('The difference between B and H', np.sum(np.abs(Bk - hess(x0))))
            k += 1
            x0 = x1
            x_store.append(x0.copy())

        x_store = np.array(x_store)
        return x_store, k


if __name__ == '__main__':
    PES = SimpleSurface()
    np.random.seed(3)
    for i in range(6):
        x0 = (np.pi/2, -np.pi/2) + (np.random.rand(2)-0.5)
        Qnewton = Newton(PES.get_value, PES.get_diff, x0)
        plt.figure(i+1)
        x_store, k = Qnewton.bfgs_newton(PES.get_hess)
        PES.show_surface_2d(-5, 5)
        PES.show_point_2d(x_store)
        print('第%d次迭代次数%d' % (i+1, k))
        plt.title('It iterates %d in No.%d time' % (k, i+1))