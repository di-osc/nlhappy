import oss2

class OSSStorer:
    '''阿里云oss对象存储'''
    def __init__(
        self, 
        access_key_id : str='LTAI5tPsMSE5G3srWxB8j3yw', 
        access_key_secret : str ='z5jPdkfNq4WPtV4c7YaAJwH5Sj45gT', 
        endpoint :str = 'https://oss-cn-beijing.aliyuncs.com', 
        bucket_name : str = 'deepset'
        ):
        super().__init__()
        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(self.auth, endpoint, bucket_name)

    def download(self, filename : str, localfile : str):
        self.bucket.get_object_to_file(key=filename, filename=localfile)