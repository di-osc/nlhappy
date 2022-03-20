import os
import oss2
import zipfile
from .utils import zip_all_files

default_access_key_id = 'LTAI5tPsMSE5G3srWxB8j3yw'
default_access_key_secret = 'z5jPdkfNq4WPtV4c7YaAJwH5Sj45gT'
default_endpoint = 'http://oss-cn-beijing.aliyuncs.com'
default_data_bucket = 'deepset'
default_model_bucket = 'pretrained-model'
default_asset_bucket = 'deepasset'

class OSSStorer:
    '''阿里云oss对象存储'''
    def __init__(
        self, 
        access_key_id : str = default_access_key_id,
        access_key_secret : str = default_access_key_secret, 
        endpoint :str = default_endpoint, 
        data_bucket : str = default_data_bucket,
        model_bucket : str = default_model_bucket,
        asset_bucket : str = default_asset_bucket
        ):
        super().__init__()
        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.data_bucket = oss2.Bucket(self.auth, endpoint, data_bucket)
        self.model_bucket = oss2.Bucket(self.auth, endpoint, model_bucket)
        self.assets_bucket = oss2.Bucket(self.auth, endpoint, asset_bucket)

    def download(
        self, 
        filename : str, 
        localfile : str
        ):
        self.data_bucket.get_object_to_file(key=filename, filename=localfile)

    def list_all_assets(self):
        """获取数据名称"""
        all_asset = []
        for obj in oss2.ObjectIterator(self.assets_bucket):
            asset = obj.key.split('.')[0]
            all_asset.append(asset)
        return all_asset


    def list_all_datasets(self):
        """获取所有数据集名称"""
        all_data = []
        for obj in oss2.ObjectIterator(self.data_bucket):
            data = obj.key.split('.')[0]
            all_data.append(data)
        return all_data
    
    def list_all_plms(self):
        """获取所有预模型名称"""
        all_model = []
        for obj in oss2.ObjectIterator(self.model_bucket):
            model = obj.key.split('.')[0]
            all_model.append(model)
        return all_model


    def download_dataset(
        self, 
        dataset:str, 
        localpath: str='./datasets/'):
        """下载数据集
        - dataset: 数据集名称
        - localpath: 下载到本地的路径 默认为./datasets/
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

    def download_plm(
        self, 
        model:str, 
        localpath: str = './plms/'):
        """下载预训练模型
        - model: 模型名称
        - localpath: 下载到本地的路径 默认为./plms/
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


    def download_asset(
        self, 
        asset:str, 
        localpath: str = './assets/'):
        """下载assets
        - asset: 资产名称
        - localpath: 下载到本地的路径 默认为./assets/
        """
        if not os.path.exists(localpath):
            os.makedirs(localpath)
        file = asset + '.zip'
        file_path = localpath + file
        asset_path = localpath + asset
        if not os.path.exists(asset_path):
            self.assets_bucket.get_object_to_file(key=file, filename=file_path)
            with zipfile.ZipFile(file=file_path, mode='r') as zf:
                zf.extractall(path=localpath)
        if os.path.exists(file_path):
            os.remove(path=file_path)
        

    def upload_dataset(
        self, 
        dataset:str, 
        localpath: str = 'datasets/'):
        """上传数据集
        - dataset: 数据集名称
        - localpath: 数据集路径, 默认为datasets/
        """
        file = dataset + '.zip'
        file_path = localpath + file
        dataset_path = localpath + dataset
        with zipfile.ZipFile(file=file_path, mode='w') as z:
            zip_all_files(dataset_path, z, pre_dir=dataset)
        self.data_bucket.put_object_from_file(key=file, filename=file_path)
        if os.path.exists(file_path):
            os.remove(path=file_path)


    def upload_pretrained(
        self, 
        model, 
        localpath: str = 'plms/'):
        """上传预训练模型
        - model: 模型名称
        - localpath: 预训练模型路径, 默认为plms/
        """
        file = model + '.zip'
        file_path = localpath + file
        model_path = localpath + model
        # 注意如果不用with 语法, 如果没有关闭zip文件则解压会报错
        with zipfile.ZipFile(file=file_path, mode='w') as z:
            zip_all_files(model_path, z, model)
        self.model_bucket.put_object_from_file(key=file, filename=file_path)
        if os.path.exists(file_path):
            os.remove(path=file_path)


    def upload_asset(
        self,
        asset,
        localpath: str = './assets/'
    ):
        """上传原始数据
        - asset: 数据名称
        - localpath: 数据的路径, 默认为./assets/
        """
        file = asset + '.zip'
        file_path = localpath + file
        asset_path = localpath + asset
        with zipfile.ZipFile(file=file_path, mode='w') as z:
            zip_all_files(asset_path, z, asset)
        self.assets_bucket.put_object_from_file(key=file, filename=file_path)
        if os.path.exists(file_path):
            os.remove(path=file_path)


    
if __name__ == '__main__':
    oss = OSSStorer()
    assets = oss.list_all_assets()
    datasets = oss.list_all_datasets()
    print(assets)
    print(datasets)