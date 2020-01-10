class Path(object):
    @staticmethod
    def train_dir():
        #training data path
        train_path = '/Path/to/TrainData'
        #validating data path
        val_path = "/Path/to/ValidationData"

        #save model into save_path
        save_path = "/checkpoints"

        return train_path,val_path,save_path

    @staticmethod
    def test_dir():
        #testing data path
        in_dir = "/Path/to/TestData"

        #save result into result_dir
        out_dir = "/Path/to/OutDemoireImage"

        return in_dir, out_dir


    @staticmethod
    def model_dir():
        return "/Path/to/PretrainedModel"
