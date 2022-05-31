# coding: utf-8
'''
Created on Mar 29, 2018
@author: guoweiyu
'''

import os
import time


class logger(object):
    def __init__(self, log_dir='.'):
        log_dir = os.path.join(log_dir, 'info')
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        self.filename = os.path.join(log_dir, 'training.log')

        with open(self.filename, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def logging(self, message):
        with open(self.filename, "a") as log_file:
            #now = time.strftime("%c")
            #log_file.write('(%s) ' % now)
            log_file.write(message+'\n')
