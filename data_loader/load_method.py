import torch
import scipy.io as sio
import numpy as np
from utilsss.general_util import applyPCA, createImageCubes, splitTrainTestSet
from data_loader.dataset import TrainDS, TestDS

def loadData(dataset_name):
    # 读入数据
    if dataset_name == "IN":
        data = sio.loadmat('./data/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat('./data/Indian_pines_gt.mat')['indian_pines_gt']
        num_classes = 16
    if dataset_name == "IN_DIFF":
        data = np.load('./data/IP_OUT.npy')
        labels = np.load('./data/IP_label_OUT.npy')
        num_classes = 16
    if dataset_name == 'PU':
        data = sio.loadmat('./data/PaviaU.mat')['paviaU']
        labels = sio.loadmat('./data/PaviaU_gt.mat')['paviaU_gt']
        num_classes = 9
    if dataset_name == 'SA':
        data = sio.loadmat('./data/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat('./data/Salinas_gt.mat')['salinas_gt']
        num_classes = 16
    if dataset_name == 'H2013':
        data = sio.loadmat('./data/HSI.mat')['HSI']
        labels = sio.loadmat('./data/gt.mat')['gt']
        num_classes = 15
    if dataset_name == 'WHU_HongHu':
        data = sio.loadmat('./data/WHU_Hi_HongHu.mat')['WHU_Hi_HongHu']
        labels = sio.loadmat('./data/WHU_Hi_HongHu_gt.mat')['WHU_Hi_HongHu_gt']
        num_classes = 22
    if dataset_name == 'PC':
        data = sio.loadmat('./data/Pavia.mat')['pavia']
        labels = sio.loadmat('./data/Pavia_gt.mat')['pavia_gt']
        num_classes = 9
    if dataset_name == 'KSC':
        data = sio.loadmat('./data/KSC.mat')['KSC']
        labels = sio.loadmat('./data/KSC_gt.mat')['KSC_gt']
        num_classes = 13
    
    return data, labels, num_classes



def create_data_loader(dataset_name, test_ratio, patch_size, pca_components, batch_size):
    # 读入数据
    X, y, num_classes = loadData(dataset_name=dataset_name)
    H,W,C = X.shape
    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    if pca_components == 0:
        X_pca = X
        pca_components = C
    else:
        X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y_all, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    # X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    # Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    # Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    # 为了适应 pytorch 结构，数据要做 transpose
    X = X_pca.transpose(0, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 3, 1, 2)
    Xtest = Xtest.transpose(0, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)
    
    # 保存中间数据集
    # np.save('IN_X.npy', X)
    # np.save('IN_X_label.npy', y_all)
    # np.save('IN_XTrain.npy', Xtrain)
    # np.save('IN_XTrain_label.npy', ytrain)
    

    # 创建train_loader和 test_loader
    X = TestDS(X, y_all)
    print(max(ytrain))
    print(min(ytrain))
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                              )

    return train_loader, test_loader, all_data_loader, y, num_classes


def create_data_loader_SSFTTnet(dataset_name, test_ratio, patch_size, pca_components, batch_size):
    # 读入数据
    X, y, num_classes = loadData(dataset_name)
    H,W,C = X.shape
    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    # 如果pca_components = 0 那么不进行pca处理
    if pca_components == 0:
        X_pca = X
        pca_components = C
    else:
        X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y_all, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    # 为了适应 pytorch 结构，数据要做 transpose
    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # 创建train_loader和 test_loader
    X = TestDS(X, y_all)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                              )

    return train_loader, test_loader, all_data_loader, y, num_classes
