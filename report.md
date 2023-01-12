# 计算机图形学 - Info-NeRF Jittor

陈新	计研三一	2022210877

## 原理

**NeRF** 通过神经网络，输入位置和方向，输出该点的颜色和体密度
$$
(x, d) \rarr (c, \sigma)
$$
因此能够利用 Ray marching、体渲染的方式进行渲染。



神经网络共10层，输入位置与方向

第一层为输入维度到256维的线性层，之后8层256维的线性层。其中第四层重新添加输入信息以强化记忆；第八层额外添加特征层以学习神经辐射场参数。

最后降维到3，即RGB信息

![nerf](pictures/nerf.png)



损失函数由两部分组成（无$L_{KL}$）
$$
L = L_{RGB} +\lambda L_{entropy} \\
{L}_{entropy} = \frac{1}{|\mathcal{R}_s|+|\mathcal{R}_u|}\sum_{r\in \mathcal{R}_s \cup\mathcal{R}_u}M(r)\odot H(r)
$$
因为只有少部分光线击中了采样点，剩下的部分只起到噪音的作用。最小化光线的信息熵有助于减小重建过程中的噪音。



## 代码结构

`main.py` 为程序主入口

`create_nerf` 创建了一个 `model.NeRF` 网络

`render` 为体渲染计算函数



## 运行方式

训练：

```
python main.py --config configs/infonerf/synthetic/lego.txt
```

渲染测试集：

```
python main.py --config configs/infonerf/synthetic/lego.txt --testskip 1 --render_test --render_only
```



## 实验

### 环境

Windows 11，RTX2080Ti

参数见 `configs/infonerf/synthetic/lego.txt`



### 训练结果

| Iteration | PSNR    | PSNR_redefine      |
| --------- | ------- | ------------------ |
| 8000      | 18.4665 | 18.743877410888672 |
|           |         |                    |
|           |         |                    |

