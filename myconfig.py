import torch
import warnings

class DefaultConfig(object):
    env = 'default'
    model = 'AlexNet'

    train_data_root = 'T:\\dataset\\DOGCAT\\train'
    test_data_root = 'T:\\dataset\\DOGCAT\\test1'
    load_model_path = 'checkpoints/model.pth'

    batch_size = 128
    use_gpu = True
    num_workers = 4
    print_freq = 20 #每20个batch打印一次

    debug_file = 'T:\\dataset\\DOGCAT\\debug'
    result_file = 'result.csv'

    max_epoch = 20
    lr = 0.1
    lr_decay = 0.95 #当验证集loss增加，lr = lr*lr_decay
    weight_decay = 1e-4 #L2惩罚项的系数

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut {}".format(k))
            setattr(self, k, v)
        opt.device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')

        print('user config:')
        # module.__class__.__dict__能输出该类的类参数
        # module.__dict__能输出self参数
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))

opt = DefaultConfig()
