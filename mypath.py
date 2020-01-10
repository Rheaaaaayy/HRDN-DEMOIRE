class Path(object):
    @staticmethod
    def train_dir():
        #training data path
        train_path = '/Path/to/UCF-101'
        #validating data path
        val_path = "/seconddisk/sayhi/dataset/TIPbenchmark/train/trainData"

        #save model into save_path
        save_path = "/seconddisk/sayhi/dataset/TIPbenchmark/test/testData"

        return train_path,val_path,save_path

    @staticmethod
    def test_dir():
        #testing data path
        test_path = "/seconddisk/sayhi/dataset/TIPbenchmark/test/testData"

        #save result into result_dir
        result_dir = "/home/publicuser/workspaces/demoire/all_results/"

        return test_path,result_dir


    @staticmethod
    def model_dir():
        return "checkpoints/TIP_regular_1e-2/HRnet_epoch84_1221_14_43_05.pth"
