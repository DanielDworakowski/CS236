import gc
import re
import os
import time
import torch
import inspect
import logging.handlers
from pathlib import Path
#
#
def setLogName(name):
    global __logdir
    ld = os.environ.get('logdir', None)
    if ld is None:
        __logdir = name
        os.environ['logdir'] = name
__logdict = {}
__logdir = None
setLogName(time.strftime('%d-%m-%Y-%H-%M-%S'))
#
#
class PIDFileHandler(logging.handlers.WatchedFileHandler):

    def __init__(self, filename, mode='a', encoding=None, delay=0):
        filename = self._append_pid_to_filename(filename)
        super(PIDFileHandler, self).__init__(filename, mode, encoding, delay)

    def _append_pid_to_filename(self, filename):
        pid = os.getpid()
        path, extension = os.path.splitext(filename)
        return '{0}-{1}{2}'.format(path, pid, extension)

#
#
def getLogger():
    global __logdir
    global __logdict
    pid = os.getpid()
    lg = __logdict.get(pid, None)
    if lg is None:
        if __logdir is None:
            __logdir = os.environ.get('logdir', None)
            assert __logdir is not None
        lg = logging.getLogger('gocr')
        lg.setLevel(logging.DEBUG)
        pth = 'log/%s/'%__logdir
        Path(pth).mkdir(parents=True, exist_ok=True)
        fh = PIDFileHandler('%s/gocr.log'%pth)
        lg.addHandler(fh)
        __logdict[pid] = lg
    return lg
#
# Color terminal (https://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python).
class Colours:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
#
# Error information.
def lineInfo(levelOffset=0):
    callerframerecord = inspect.stack()[2+levelOffset]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    file = info.filename
    file = file[file.rfind('/') + 1:]
    return '%s::%s:%d' % (colourString(file, Colours.OKGREEN), colourString(info.function, Colours.UNDERLINE), info.lineno)
#
# Line information.
def getLineInfo(leveloffset=0):
    level = 2 + leveloffset
    callerframerecord = inspect.stack()[level]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    file = info.filename
    file = file[file.rfind('/') + 1:]
    return '%s: %d' % (file, info.lineno)
#
# Colours a string.
def colourString(msg, ctype):
    return ctype + msg + Colours.ENDC
#
# Print something in color.
def printColour(msg, ctype):
    print(colourString(msg, ctype))
#
# Print information.
def strInfo(*umsg, searchTerm='strInfo', offset=0, colour=Colours.OKGREEN):
    msg = '%s:  ' % (lineInfo(offset))
    lst = ''
    for mstr in umsg:
        if isinstance(mstr, torch.Tensor):
            vname = varname(mstr, offset, searchTerm)
            lst += '[' + str(vname) +  ']\n'
        lst += str(mstr) + ' '
    msg = colourString(msg, colour) + lst
    return msg
#
#
def printInfo(*umsg):
    msg = strInfo(*umsg, searchTerm='printInfo', offset=1)
    print(msg)
#
#
def logInfo(*umsg):
    msg = strInfo(*umsg, searchTerm='printInfo', offset=1)
    lg = getLogger()
    lg.info(msg)

def printDebug(*umsg):
    msg = strInfo(*umsg, Colours.UNDERLINE)
    lg = getLogger()
    lg.debug(msg)
#
# Print error information.
def printFrame():
    print(lineInfo(), Colours.WARNING)
#
# Print an error.
def strError(*errstr):
    msg = '%s:  ' % (lineInfo(1))
    lst = ''
    for mstr in errstr:
        lst += str(mstr) + ' '
    msg = colourString(msg, Colours.FAIL) + lst
    return msg
#
#
def printError(*errstr):
    msg = strError(*errstr)
    print(msg)
#
#
def logError(*umsg):
    msg = strError(*umsg)
    lg = getLogger()
    lg.error(msg)
#
# Print a warning.
def strWarn(*warnstr):
    msg = '%s:  ' % (lineInfo(1))
    lst = ''
    for mstr in warnstr:
        lst += str(mstr) + ' '
    msg = colourString(msg, Colours.WARNING) + lst
    return msg
#
#
def printWarn(*warnstr):
    msg = strWarn(*warnstr)
    print(msg)
#
#
def logWarn(*warnstr):
    msg = strWarn(*warnstr)
    lg = getLogger()
    lg.warn(msg)
#
# Get name of variable passed to the function
def varname(p, leveloffset=0, ss='printTensor'):
    level = 2 + leveloffset
    frame = inspect.stack()[level][0]
    for line in inspect.getframeinfo(frame).code_context:
        m = re.search(r'\b%s\s*\(\s*(.*)\s*\)'%ss, line)
        if m:
            return m.group(1)
#
#
def printList(is_different, dlist):
    ret = ''
    if is_different:
        ret = dlist
    else:
        ret = [str(dlist[0])]
    return ret
#
#
def getDevice(t):
    ret = None
    if isinstance(t, torch.Tensor):
        ret = t.device
    else:
        ret = type(t)
    return ret
#
# Get the s
def tensorListInfo(tensor_list, vname, usrmsg, leveloffset):
    assert isinstance(tensor_list, list)
    str_ret = ''
    dtypes = [tensor_list[0].dtype]
    devices = [tensor_list[0].device]
    shapes = [tensor_list[0].shape]
    dtype_different = False
    devices_different = False
    shapes_different = False
    for t_idx in range(1, len(tensor_list)):
        t = tensor_list[t_idx]
        dtypes.append(t.dtype)
        devices.append(getDevice(t))
        shapes.append(t.shape)
        dtype_different |= (t.dtype != dtypes[0])
        devices_different |= (t.device != devices[0])
        shapes_different |= (t.shape != shapes[0])
    dtypes = printList(dtype_different or devices_different, dtypes)
    devices = printList(dtype_different or devices_different, devices)
    shapes = str(printList(shapes_different, shapes))

    devices_dtypes = ' '.join(map(str, *zip(dtypes, devices)))
    msg = colourString(colourString(getLineInfo(leveloffset + 1), Colours.UNDERLINE), Colours.OKBLUE) + ': [' + str(vname) +  '] ' + '<list> len: %d'%len(tensor_list) + ' (' + colourString(devices_dtypes, Colours.WARNING) + ') -- '  + colourString('%s'%shapes, Colours.OKGREEN) + ' </list>' + usrmsg
    return msg
#
# Print information about a tensor.
def strTensor(tensor, usrmsg='', leveloffset=0):
    vname = varname(tensor, 1)
    if isinstance(tensor, list):
        msg = tensorListInfo(tensor, vname, usrmsg, leveloffset)
    elif isinstance(tensor, torch.Tensor):
        msg = colourString(colourString(getLineInfo(leveloffset), Colours.UNDERLINE), Colours.OKBLUE) + ': [' + str(vname) +  '] (' + colourString(str(tensor.dtype) + ' ' + str(getDevice(tensor)), Colours.WARNING) + ') -- '  + colourString('%s'%str(tensor.shape), Colours.OKGREEN) + ' ' + usrmsg
    else:
        msg = colourString(colourString(getLineInfo(leveloffset), Colours.UNDERLINE), Colours.OKBLUE) + ': [' + str(vname) +  '] (' + colourString(str(tensor.dtype) + ' ' + str(getDevice(tensor)), Colours.WARNING) + ') -- '  + colourString('%s'%str(tensor.shape), Colours.OKGREEN) + ' ' + usrmsg
    return msg
#
#
def printTensor(tensor, usrmsg='', leveloffset=1):
    msg = strTensor(tensor, usrmsg, leveloffset)
    print(msg)
#
# Print debugging information.
def dprint(usrmsg, leveloffset=0):
    msg = colourString(colourString(getLineInfo(leveloffset), Colours.UNDERLINE), Colours.OKBLUE) + ': ' + str(usrmsg)
    print(msg)

def hasNAN(t):
    msg = colourString(colourString(getLineInfo(), Colours.UNDERLINE), Colours.OKBLUE) + ': ' + colourString(str('Tensor has %s NaNs'%str((t != t).sum().item())), Colours.FAIL)
    print(msg)

def torch_mem():
    dprint('Torch report: Allocated: %.2f MBytes Cached: %.2f' % (torch.cuda.memory_allocated() / (1024 ** 2), torch.cuda.memory_cached() / (1024 ** 2)), 1)

## MEM utils ##
def mem_report():
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported
    https://gist.github.com/Stonesjtu/368ddf5d9eb56669269ecdf9b0d21cbe'''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        print('Storage on %s' %(mem_type))
        print('-'*LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print('%s\t\t%s\t\t%.2f' % (
                element_type,
                size,
                mem) )
        print('-'*LEN)
        print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
        print('Torch report: %.2f MBytes' % (torch.cuda.memory_allocated() / (1024 ** 2)))
        print('-'*LEN)

    LEN = 65
    print('='*LEN)
    gc.collect()
    objects = gc.get_objects()
    print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, 'GPU')
    _mem_report(host_tensors, 'CPU')
    print('='*LEN)
