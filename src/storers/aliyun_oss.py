import os
import oss2
import zipfile
import os
from itertools import islice

class OSSStorer:
    '''阿里云oss对象存储'''
    def __init__(
        self, 
        access_key_id : str='LTAI5tPsMSE5G3srWxB8j3yw', 
        access_key_secret : str ='z5jPdkfNq4WPtV4c7YaAJwH5Sj45gT', 
        endpoint :str = 'https://oss-cn-beijing.aliyuncs.com', 
        data_bucket : str = 'deepset',
        model_bucket : str = 'pretrained-model'
        ):
        super().__init__()
        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.data_bucket = oss2.Bucket(self.auth, endpoint, data_bucket)
        self.model_bucket = oss2.Bucket(self.auth, endpoint, model_bucket)

    def download(self, filename : str, localfile : str):
        self.data_bucket.get_object_to_file(key=filename, filename=localfile)


    def get_all_data(self):
        all_data = []
        for obj in oss2.ObjectIterator(self.data_bucket):
            data = obj.key.split('.')[0]
            all_data.append(data)
        return all_data


    def download_dataset(self, dataset:str, localpath: str):
        """下载数据集
        - dataset: 数据集名称
        - localpath: 下载到本地的路径
        """
        file = dataset + '.zip'
        file_path = localpath + file
        dataset_path = localpath + dataset
        if not os.path.exists(dataset_path):
            self.data_bucket.get_object_to_file(key=file, filename=file_path)
            with zipfile.ZipFile(file=file_path, mode='r') as zf:
                zf.extractall(path=localpath)
        if os.path.exists(file_path):
            os.remove(path=file_path)

    def download_model(self, model:str, localpath: str):
        """下载预训练模型
        - model: 模型名称
        - localpath: 下载到本地的路径
        """
        file = model + '.zip'
        file_path = localpath + file
        model_path = localpath + model
        if not os.path.exists(model_path):
            self.model_bucket.get_object_to_file(key=file, filename=file_path)
            with zipfile.ZipFile(file=file_path, mode='r') as zf:
                zf.extractall(path=localpath)
        if os.path.exists(file_path):
            os.remove(path=file_path)
