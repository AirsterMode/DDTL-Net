import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv


def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc



# 这里需要更改！！！！！！，改为根据不同数据集进行适配
def acc_reports(y_test, y_pred_test, dataset_name):

    if dataset_name == 'IN':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    if dataset_name == 'PU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees'
            , 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    if dataset_name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow'
            , 'Fallow_smooth', 'Stubble', 'Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds','Lettuce_romaine_4wk'
            ,'Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk','Vinyard_untrained','Vinyard_vertical_trellis']
    if dataset_name == 'H2013':
        target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees'
            , 'Soil', 'Water', 'Residential','Commercial','Road','Highway','Railway'
            ,'Parking Lot 1','Parking Lot 2','Tennis Court','Running Track']
    if dataset_name == 'PC':
        target_names = ['Water', 'Trees', 'Asphalt', 'Self-Blocking Bricks'
            , 'Bitumen', 'Tiles', 'Shadows',
                        'Meadows', 'BareSoil']
    if dataset_name == 'WHU_HongHu':
        target_names = ['RedRoof', 'Road', 'BareSoil', 'Cotton'
            , 'CottonFireWood', 'Rape', 'ChineseCabbage','Pakchoi','Cabbage','TuberMustard','BrassicaParachinensis1'
            ,'BrassicaChinensis2','SmallBrassicaChinensis','LactucaSativa','Celtuce',
                       'FilmCoveredLettuce','RomaineLettuce','Carrot','WhiteRadish',
                       'GarlicSprout','BroadBean','Tree']
    if dataset_name == 'KSC':
        target_names = ['Scrub', 'WillowSwamp', 'CabbagePalmHammock', 'CabbagePalm'
            , 'SlashPine', 'Oak', 'HardwoodSwamp','GraminoidMarsh','SpartinaMarsh','CattailMarsh','SaltMarsh'
            ,'MudFlats','Wate']
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100