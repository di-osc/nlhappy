from bypy import ByPy

class BaiduPanStorer:
    '''
    BaiduPanStorer is a class to store files on BaiduPan.
    '''
    def __init__(self):
        self.bypy = ByPy()

    def store(self, file_path, remote_path):
        '''
        Store a file on BaiduPan.
        '''
        self.bypy.upload(local_pah = filepath, remotepath = remote_path)

    def download(self, remote_file, local_path):
        '''
        Download a file from BaiduPan.
        '''
        self.bypy.downfile(remotefile=remote_file, localpath=local_path)
        