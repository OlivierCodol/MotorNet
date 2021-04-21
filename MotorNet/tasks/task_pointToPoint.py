import numpy as np


def generateDelayTime(delayMin, delayMax, delayMode):
    if delayMode == 'random':
        delayTime = np.random.uniform(delayMin, delayMax)
    elif delayMode == 'noDelayInput':
        delayTime = 0

    return delayTime


