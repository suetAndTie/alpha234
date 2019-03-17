"""
Utils for multiprocessing
"""

def executor_init(l):
    '''
    Used to initialize lock for all processes
    '''
    global lock
    lock = l
