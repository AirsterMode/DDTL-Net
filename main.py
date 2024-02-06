import numpy as np



import torch
import torch.nn as nn
import torch.optim as optim
import get_cls_map
import time

from SSFTTnet import SSFTTnet
from AirsterNet import AirsterNet
from xiaorong_fft import FFT
from xiaorong_refine import Refine
from manba import Manba

from utilsss.general_util import setup_seed
from utilsss.mertics import acc_reports
from data_loader.load_method import create_data_loader_SSFTTnet, create_data_loader

# 随机点数
# setup_seed(2023)



def train(train_loader, epochs, num_classes, in_channels, model_name, patch_size):

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Train Environment!-->>>>>> "+ str(device))
    

    # 网络原来的
    if model_name == 'DDTL':
        net = AirsterNet(num_classes=num_classes, in_channels=in_channels, patch_size=patch_size).to(device)

        
    net.to(device)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 开始训练
    total_loss = 0
    for epoch in range(epochs):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            outputs = net(data)
            # 计算损失函数
            loss = criterion(outputs, target)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))
    print('Finished Training')

    return net, device

def test(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test




def train_work(
    dataset_name,
    model_name,
    to3dData,
    in_channels,
    test_ratio,
    patch_size = 15,
    pca_components = 30,
    batch_size = 32,
    epoch_num = 100,
    Iter=10):
    
    # 是否转换为3d数据，有些网络要进行3d卷积，因此需要3d数据！~
    # if model_name == 'SSFTTnet':
    to3dData = True


    if to3dData:
        train_loader, test_loader, all_data_loader, y_all, num_classes = create_data_loader_SSFTTnet(
            dataset_name=dataset_name,
            test_ratio=test_ratio,
            patch_size=patch_size,
            pca_components = pca_components,
            batch_size=batch_size)
    else:
        train_loader, test_loader, all_data_loader, y_all, num_classes = create_data_loader(
            dataset_name=dataset_name,
            test_ratio=test_ratio,
            patch_size=patch_size,
            pca_components = pca_components,
            batch_size=batch_size)

    
    arr = []
    oa_sum = []
    aa_sum = []
    kappa_sum = []
    for itera in range(1, Iter + 1):
        tic1 = time.perf_counter()
        net, device = train(train_loader, epochs=epoch_num, num_classes=num_classes, in_channels=in_channels, model_name=model_name, patch_size=patch_size)
        # 只保存模型参数
        torch.save(net, 'cls_params/' + model_name + '_' + dataset_name + '_' + 'epoch' + str(epoch_num) + '_Iter' + str(itera) + '.pth')
        toc1 = time.perf_counter()
        
        tic2 = time.perf_counter()
        y_pred_test, y_test = test(device, net, test_loader)
        toc2 = time.perf_counter()
        
        # 评价指标
        classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test, dataset_name)
        classification = str(classification)
        Training_Time = toc1 - tic1
        Test_time = toc2 - tic2

        # 修改部分，后期会把训练和测试分开，测试会选择多个训练后的模型进行测试，用来计算方差
        arr.append(np.array(each_acc))
        oa_sum.append(np.array(oa))
        aa_sum.append(np.array(aa))
        kappa_sum.append(np.array(kappa))

    # 变为numpy矩阵后分析
    arrNp = np.array(arr)
    # mean均值
    arrMean = np.mean(arrNp, axis=0)
    # std方差
    arrStd = np.std(arrNp, axis=0)

    # mean均值
    aa_mean = np.mean(aa_sum)
    print(aa_mean)
    # std方差
    aa_std = np.std(aa_sum)
    # mean均值
    oa_mean = np.mean(oa_sum)
    print(oa_mean)
    # std方差
    oa_std = np.std(oa_sum)
    # mean均值
    kappa_mean = np.mean(kappa_sum)
    # std方差
    kappa_std = np.std(kappa_sum)
    
    file_name = 'cls_result/classification_report_' + model_name + '_' + dataset_name + '_' + 'epoch' + str(epoch_num) + '_Iter'+ str(Iter) + '_patch' + str(patch_size) + '.txt'
    
    with open(file_name, 'w') as x_file:
        x_file.write('{} The Last one Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} The Last one Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} The Last one Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} The Last one Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} The Last one Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} The Last one Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{} The Average of AA'.format(aa_mean))
        x_file.write('\n')
        x_file.write('{} The std of AA'.format(aa_std))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{} The Average of OA'.format(oa_mean))
        x_file.write('\n')
        x_file.write('{} The std of OA'.format(oa_std))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{} The Average of Kappa'.format(kappa_mean))
        x_file.write('\n')
        x_file.write('{} The std of Kappa'.format(kappa_std))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{} The Average of each_acc'.format(arrMean))
        x_file.write('\n')
        x_file.write('{} The std of each_acc'.format(arrStd))
        x_file.write('\n')
        x_file.write(str(oa_sum))
        
        
    get_cls_map.get_cls_map(net, device, all_data_loader, y_all, dataset_name, model_name)
    
    return oa_mean,aa_mean


def test_work(
    dataset_name,
    model_name,
    to3dData,
    in_channels,
    test_ratio,
    patch_size = 15,
    pca_components = 30,
    batch_size = 32,
    epoch_num = 100,
    Iter=10):
    
    # 是否转换为3d数据，有些网络要进行3d卷积，因此需要3d数据！~
    # if model_name == 'SSFTTnet':
    to3dData = True
    
    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    


    if to3dData:
        train_loader, test_loader, all_data_loader, y_all, num_classes = create_data_loader_SSFTTnet(
            dataset_name=dataset_name,
            test_ratio=test_ratio,
            patch_size=patch_size,
            pca_components = pca_components,
            batch_size=batch_size)
    else:
        train_loader, test_loader, all_data_loader, y_all, num_classes = create_data_loader(
            dataset_name=dataset_name,
            test_ratio=test_ratio,
            patch_size=patch_size,
            pca_components = pca_components,
            batch_size=batch_size)
        
    
    # 网络原来的
    if model_name == 'DDTL':
        net = AirsterNet(num_classes=num_classes).to(device)

        
    arr = []
    oa_sum = []
    aa_sum = []
    kappa_sum = []
    for itera in range(1, Iter + 1):

        # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 只保存模型参数

        net = torch.load('cls_params/' + model_name + '_' + dataset_name + '_' + 'epoch' + str(epoch_num) + '_Iter' + str(itera) + '.pth')
        # toc1 = time.perf_counter()
        
        tic2 = time.perf_counter()
        y_pred_test, y_test = test(device, net, test_loader)
        toc2 = time.perf_counter()
        
        # 评价指标
        classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test, dataset_name)
        classification = str(classification)

        Test_time = toc2 - tic2

        # 修改部分，后期会把训练和测试分开，测试会选择多个训练后的模型进行测试，用来计算方差
        arr.append(np.array(each_acc))
        oa_sum.append(np.array(oa))
        aa_sum.append(np.array(aa))
        kappa_sum.append(np.array(kappa))

    # 变为numpy矩阵后分析
    arrNp = np.array(arr)
    # mean均值
    arrMean = np.mean(arrNp, axis=0)
    # std方差
    arrStd = np.std(arrNp, axis=0)

    # mean均值
    aa_mean = np.mean(aa_sum)
    # std方差
    aa_std = np.std(aa_sum)
    # mean均值
    oa_mean = np.mean(oa_sum)
    # std方差
    oa_std = np.std(oa_sum)
    # mean均值
    kappa_mean = np.mean(kappa_sum)
    # std方差
    kappa_std = np.std(kappa_sum)
    
    file_name = 'cls_result/classification_report_' + model_name + '_' + dataset_name + '_' + 'epoch' + str(epoch_num) + '_Iter'+ str(Iter) + '_patch' + str(patch_size) + '.txt'
    
    with open(file_name, 'w') as x_file:
        x_file.write('{} The Last one Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} The Last one Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} The Last one Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} The Last one Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} The Last one Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{} The Average of AA'.format(aa_mean))
        x_file.write('\n')
        x_file.write('{} The std of AA'.format(aa_std))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{} The Average of OA'.format(oa_mean))
        x_file.write('\n')
        x_file.write('{} The std of OA'.format(oa_std))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{} The Average of Kappa'.format(kappa_mean))
        x_file.write('\n')
        x_file.write('{} The std of Kappa'.format(kappa_std))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{} The Average of each_acc'.format(arrMean))
        x_file.write('\n')
        x_file.write('{} The std of each_acc'.format(arrStd))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{} Every OA'.format(oa_sum))
        
    get_cls_map.get_cls_map(net, device, all_data_loader, y_all, dataset_name, model_name)
    

def commandWork(dataset_name, model_name, patch_size):
    to3dData = False

    # 用于测试样本的比例
    if dataset_name == 'IN':
        test_ratio = 0.945
    if dataset_name == 'PU':
        test_ratio = 0.99
    if dataset_name == 'SA':
        test_ratio = 0.995 
    if dataset_name == 'PC':
        test_ratio = 0.99  
    if dataset_name == 'WHU_HongHu':
        test_ratio = 0.99  
    if dataset_name == 'H2013':
        test_ratio = 0.97
    if dataset_name == 'KSC':
        test_ratio = 0.97
        

    # 使用 PCA 降维，得到主成分的数量,如果为0，则不进行pca
    pca_components = 30
    in_channels = 30
    # 每批次训练的size
    batch_size = 512
    # 训练的epoch
    epoch_num = 100
    # 训练的Iter
    Iter = 10
    
    
    # Train Test
    task = 'Train'
    
    if task == 'Train':
        while True:
            oa_mean,aa_mean = train_work(dataset_name, model_name, to3dData, in_channels, test_ratio, patch_size, pca_components, batch_size,
                       epoch_num, Iter)
            if oa_mean>0 and aa_mean>0:
                break
        
        
    if task == 'Test':
        test_work(dataset_name, model_name, to3dData, in_channels, test_ratio, patch_size, pca_components, batch_size,
                   epoch_num, Iter)
    
if __name__ == '__main__':

    # patch
#   5 7 9 11 13 15 17 19 21
    patch_sizes = [15]
    
    # dataset_names = ["WHU_HongHu",'PU','SA','PC','H2013','KSC']
    dataset_names = ['IN']

    model_names = ['DDTL']
    
    for patch_size in patch_sizes:
        for dataset_name in dataset_names:
            for model_name in model_names:
                commandWork(dataset_name, model_name, patch_size)





