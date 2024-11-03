初始化 Net = Mnist_CNN()
创建客户端群组
对于每一轮通讯t：
    随机选中一定数量的客户端
    向客户端发送全局模型Net
    对于每个客户端i：
        令model=Net
        分类器迭代M轮：
            加入扰动
            训练model total_epoch轮
            将model加入模型列表model_ls
        OSP计算model_ls中各分类器采用概率prob
        用prob加权计算，得到总模型
        上传总模型到服务器端
    收集所有客户端的模型
    Net模型参数=客户端模型参数平均值
    计算Net准确率
```