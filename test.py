import ossaudiodev
from nlhappy.utils.storer import OSSStorer

oss = OSSStorer()

print(oss.list_all_assets())