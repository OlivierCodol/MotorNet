import scipy.io
import numpy as np
def plotOpenSim(results, savePath):
    temp = results
    j = results['joint position'].numpy().transpose(2, 1, 0).reshape((4, -1), order='F').transpose()
    m = results['muscle state'].numpy().transpose(2, 3, 1, 0).reshape((5, 6, -1), order='F').transpose(2, 1, 0)
    mdict = {'muscle': m[:,:,0], 'joint': j[:,0:2]}
    scipy.io.savemat(savePath + '.mat', mdict)