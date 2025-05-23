# coding: utf-8

"""
tools to measure elapsed time

测量耗时的工具
"""

import time

class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        return self.diff

    def clear(self):
        self.start_time = 0.
        self.diff = 0.
