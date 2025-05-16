from Controllers import BaseController
from Networks import TransRBF

config = {
    'mode': 'Train',
    'network': 'TransRBF',
    'name': 'Final-excludeMM',
    # All
    # 'dataset': {
    #     'training_list_path': 'G:\\cardiac_data\\training_pair.txt',
    #     'testing_list_path': 'G:\\cardiac_data\\testing_pair.txt',
    #     'validation_list_path': 'G:\\cardiac_data\\validation_pair.txt',
    #     'pair_dir': 'G:\\cardiac_data\\2Dwithoutcenter1/',
    #     'resolution_path': 'G:\\cardiac_data\\resolution.txt'
    # },
    # ~MM
    'dataset': {
        # # MM数据路径
        # 'training_list_path': 'G:\\dataset2D_MnMs\\dataset2D_MnMs\\training_pair.txt',
        # 'testing_list_path': 'G:\\dataset2D_MnMs\\dataset2D_MnMs\\testing_pair.txt',
        # 'validation_list_path': 'G:\\dataset2D_MnMs\\dataset2D_MnMs\\validation_pair.txt',
        # 'pair_dir': 'G:\\dataset2D_MnMs\\dataset2D_MnMs\\data\\',
        # 'resolution_path': 'G:\\dataset2D_MnMs\\dataset2D_MnMs\\resolution.txt',

        # ACDC+数据路径
        'training_list_path': 'Data/dataset/training_pair.txt', 
        'testing_list_path': 'Data/dataset/testing_pair.txt',
        'validation_list_path': 'Data/dataset/validation_pair.txt',
        'pair_dir': 'Data/dataset/data',
        'resolution_path': 'Data/dataset/resolution.txt'
    },
    
    'Train': {
        'batch_size': 16,
        'model_save_dir':
            'model',
        'lr': 5e-4,
        'max_epoch': 400,
        'save_checkpoint_step': 20,
        'v_step': 20,
        'earlystop': {
            'min_delta': 0.00001,
            'patience': 1000
        },
    },
    'Test': {
        'epoch': 'best',
        'model_save_path':
            'model/TransRBF/Final-excludeMM-LCC---[9, 9]--1100000-diff7-20241210135807',
        'excel_save_path':
            'model',
        'verbose': 2,
    },
    'ICE': {
        'epoch': 'best',
        'model_save_path':
            'model',
        'excel_save_path':
            'model',
        'verbose': 2,
    },
    'SpeedTest': {
        'epoch': 'best',
        'model_save_path':
            'D:\Code\RegistrationPakageForNerualLearning\modelForRadialPaper\DalcaDiff\\0-MSE-50-11111.111111111111-20211108191421',
        'device': 'cpu'
    },
    'Hyperopt': {
        'n_trials': 30,
        'earlystop': {
            'min_delta': 0.00001,
            'patience': 500
        },
        'max_epoch': 800,
        'lr': 1e-4
    },


# 这段代码的总体作用是为 TransRBF 网络设置超参数，
# 以便网络在构建和训练时能够按照这些参数进行操作。它定义了控制器类、神经网络类及其参数，允许模型根据任务和数据的具体要求进行灵活配置。
    'TransRBF': {
        'controller': BaseController,
        'network': TransRBF,
        'params': {
            'c_list': [1.5, 2, 2.5],
            'i_size': [128, 128],
            'factor_list': [110000, 0],
            'int_steps': 7,
            # WLCC
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                # 'alpha': 0.02,
                'win': [9, 9]
            }
        }
    },

}
