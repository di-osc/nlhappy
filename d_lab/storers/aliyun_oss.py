import os
from pyexpat import model
import oss2
import zipfile
import os
from itertools import islice

default_access_key_id = 'LTAI5tPsMSE5G3srWxB8j3yw'
default_access_key_secret = 'z5jPdkfNq4WPtV4c7YaAJwH5Sj45gT'
default_endpoint = 'http://oss-cn-beijing.aliyuncs.com'
default_data_bucket = 'deepset'
default_model_bucket = 'pretrained-model'

class OSSStorer:
    '''阿里云oss对象存储'''
    def __init__(
        self, 
        access_key_id : str = default_access_key_id,
        access_key_secret : str = default_access_key_secret, 
        endpoint :str = default_endpoint, 
        data_bucket : str = default_data_bucket,
        model_bucket : str = default_model_bucket
        ):
        super().__init__()
        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.data_bucket = oss2.Bucket(self.auth, endpoint, data_bucket)
        self.model_bucket = oss2.Bucket(self.auth, endpoint, model_bucket)

    def download(self, filename : str, localfile : str):
        self.data_bucket.get_object_to_file(key=filename, filename=localfile)


    def get_all_data(self):
        """获取所有数据集名称"""
        all_data = []
        for obj in oss2.ObjectIterator(self.data_bucket):
            data = obj.key.split('.')[0]
            all_data.append(data)
        return all_data
    
    def get_all_model(self):
        """获取所有模型名称"""
        all_model = []
        for obj in oss2.ObjectIterator(self.model_bucket):
            model = obj.key.split('.')[0]
            all_model.append(model)
        return all_model


    def download_dataset(self, dataset:str, localpath: str='./data/'):
        """下载数据集
        - dataset: 数据集名称
        - localpath: 下载到本地的路径 默认为./data/
        """
        if not os.path.exists(localpath):
            os.makedirs(localpath)
        file = dataset + '.zip'
        file_path = localpath + file
        dataset_path = localpath + dataset
        if not os.path.exists(dataset_path):
            self.data_bucket.get_object_to_file(key=file, filename=file_path)
            with zipfile.ZipFile(file=file_path, mode='r') as zf:
                zf.extractall(path=localpath)
        if os.path.exists(file_path):
            os.remove(path=file_path)

    def download_model(self, model:str, localpath: str = './pretrained_models/'):
        """下载预训练模型
        - model: 模型名称
        - localpath: 下载到本地的路径 默认为./pretrained_models/
        """
        if not os.path.exists(localpath):
            os.makedirs(localpath)
        file = model + '.zip'
        file_path = localpath + file
        model_path = localpath + model
        if not os.path.exists(model_path):
            self.model_bucket.get_object_to_file(key=file, filename=file_path)
            with zipfile.ZipFile(file=file_path, mode='r') as zf:
                zf.extractall(path=localpath)
        if os.path.exists(file_path):
            os.remove(path=file_path)

    def upload_dataset(self, dataset:str, localpath: str = 'data/'):
        """上传数据集
        - dataset: 数据集名称
        - localpath: 数据集路径, 默认为data/
        """
        file = dataset + '.zip'
        file_path = localpath + file
        dataset_path = localpath + dataset
        z = zipfile.ZipFile(file=file_path, mode='w')
        
        for root, dirs, files in os.walk(dataset_path):
            for f in files:
                z.write(os.path.join(root, f))
        
        self.data_bucket.put_object_from_file(key=file, filename=file_path)
        if os.path.exists(file_path):
            os.remove(path=file_path)
        

