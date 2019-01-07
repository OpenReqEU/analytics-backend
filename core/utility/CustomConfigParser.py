#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: cp1252 -*-


from ConfigParser import SafeConfigParser

class CustomConfigParser(SafeConfigParser):
    def __init__(self,*args,**kargs):
        SafeConfigParser.__init__(self,*args,**kargs)
        self.optionxform = str

    def ConfigSectionMap(self,section):
        """
        Function to manage the section of a configuration parser
        :param section: section of the configuration parser
        :return: a dictionary with the options/values of the section
        """
        dict1 = {}
        options = self.options(section)
        for option in options:
            try:
                dict1[option] = self.get(section, option)
            except:
                print("exception on %s!" % option)
                dict1[option] = None
        return dict1.values()


def creteDictFromConfigParamList( configParamList = None):

    return dict([eval(configParam) for configParam in configParamList])