import sys
import os
from utility.CustomConfigParser import CustomConfigParser

CONFIG_FILES = 'Configuration.ini'

conf = CustomConfigParser()
config_path = os.path.realpath(os.path.join(__file__, '..','..', CONFIG_FILES))
conf.read(config_path)

reload(sys)
sys.setdefaultencoding('utf-8')

def get_conf():
    return conf