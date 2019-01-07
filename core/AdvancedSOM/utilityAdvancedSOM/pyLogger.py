#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: cp1252 -*-   #per non aver problemi con le vocali accentate.

import  sys, os
import  traceback


#CONFIGURAZIONE DEL LOGGING#############################################################################################################################################################
import datetime

LOG_ON_FILE     =   False
FILE_WITH_PID   =   False
LEVEL           =  'DEBUG'
REDIRECT_STDOUT =   False
REDIRECT_STDERR =   False
LOGGER_NAME     =   'py4j' #stampato nel file
MAX_MBYTES      =   10
BACKUP_COUNT    =   5
RUN_ON_SPARK    =   False
##########################################################################################################################################################################################



#Application path (path dell'eseguibile):
if getattr(sys, 'frozen', False):
    applicationPath = os.path.dirname(sys.executable)
elif __file__:
    applicationPath = os.path.dirname(__file__)

#
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
# logging.root.setLevel(level=LEVEL)
#
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
if LOG_ON_FILE:
    import logging as loggingFile
    from    logging.handlers import RotatingFileHandler

    #Filename:
    if FILE_WITH_PID:
        filename =  os.path.join(applicationPath,  'Log',  'Log.log'+str(os.getpid()))
    else:
        filename =  os.path.join(applicationPath,  'Log',  'Log.log')
    #Directory:
    if  not  os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    #File Handler
    # handler = logging.FileHandler(filename)
    handler =  RotatingFileHandler(filename, backupCount=2, maxBytes=1e6*MAX_MBYTES)
    handler.setLevel(level=LEVEL)
    formatter = loggingFile.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    loggingFile.root.addHandler(handler)





# logger      =  getlogger(LOGGER_NAME)
sys_stdout  =  sys.stdout
sys_stderr  =  sys.stderr



def pylogger(level, logger, *args):
    A = ''
    for a in args:
        if not isinstance(a, unicode)   and   not isinstance(a, str):
            # print type(a)
            a = str(a)
        A += '  '+a
    if level==0:
        logger.debug(A)
    elif level==1:
        logger.info(A)
    elif level==2:
        logger.warn(A)
    elif level==3:
        logger.error(A)
    elif level==4:
        logger.critical(A)





#logging della eventuale eccezione e del suo stack
def saveError(logger, *args):
    """
    Logging della eventuale eccezione + del suo stack
    """
    pylogger(3, logger, *args)
    str_log   =  traceback.format_exc()#Descrizione eccezione
    str_log  +=  '  STACK:'
    stack_log  =  traceback.extract_stack()#Stack eccezione
    for i in range(len(stack_log)):
        str_log += '\n  ' + str(stack_log[i])
    #Save
    pylogger(3, logger, str_log)






class PLog:
    def __init__(self, level=0, name='Plog'):
        import logging
        self.level = level
        str_time = datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')
        name = name + str_time
        self.logger= self.getlogger(name, self.level)

    def __call__(self, *args):
        pylogger(self.level, self.logger, *args)

    def debug(self, *args):
        pylogger(0, self.logger, *args)
        #handler.doRollover()

    def info(self, *args):
        pylogger(1, self.logger, *args)

    def warn(self, *args):
        pylogger(2, self.logger, *args)

    def error(self, *args):
        pylogger(3, self.logger, *args)

    def critical(self, *args):
        pylogger(4, self.logger, *args)

    def saveError(self, *args):
        saveError(self.logger, *args)

    def SaveError(self, *args):
        saveError(self.logger, *args)

    def setLevel(self, level):
        self.logger.setLevel(level)

    def write(self, *args):
        pylogger(self.level, self.logger, *args)

    def setName(self, name=None):
        if name:
            self.logger.name = name

    def flush(self):
        sys_stdout.flush()

    def getlogger(self, name='', level=None):
        import logging
        import sys

        if not level:
            level = logging.DEBUG

        logger = logging.getLogger(name)
        logger.setLevel(level)
        if RUN_ON_SPARK:
            ch = logging.StreamHandler(sys.stderr)
        else:
            ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger


#Python logger:
pylog   = PLog()


if REDIRECT_STDOUT:
    #Sends anything written to stdout to an object PLog instead.
    sys.stdout = PLog()

if REDIRECT_STDERR:
    #Sends anything written to stderr to an object PLog instead.
    sys.stderr = PLog()





######################################################################################################################################################################################
######################################################################################################################################################################################






def test_base():
    print "Test to standard out"
    raise Exception('Test to standard error')






def test_logger():
    from datetime import date
    import numpy
    pylogger(33, u'23éè@', 23.)
    pylog("AVANZAMENTO: ", 3 / float(10), "%")
    pylog(23, u'@à°', test_logger)
    pylog(24, u'@à°', date(2015, 1, 23), numpy.arange(10))
    pylog.error(24, u'@à°', date(2015, 1, 23), numpy.arange(10))
    raw_input()
    logger.debug(123)
    raw_input()

    print "Livello di log di default:"
    pylog.debug('debug message')
    pylog.info('info message')
    pylog.warn('warn message')
    pylog.error('error message')


    print "logger.setLevel: WARN"
    pylog.setLevel('WARN')
    pylog.debug('debug message')
    pylog.info('info message')
    pylog.warn('warn message')
    pylog.error('error message')
    pylog.critical('critical message')


def test_error():
    try:
        pp
    except:
        saveError('asdddadxas')
        pylog.saveError("dddddddd")
        print "-----------END----------------"



def test_size():
    import time
    for k in range(1000):
        pylog(str(k) + 'X' * 10000)
        time.sleep(0.01)



def main():
    # test_size()
    # test_base()
    test_logger()
    # test_error()
    print "-----------END----------------"



if __name__ == '__main__':
    main()











