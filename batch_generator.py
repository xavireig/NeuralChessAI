import numpy as np
import keras
import h5py
import threading

class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe."""
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g

@threadsafe_generator
def batch_generator(data_path, batch_size, steps, variant, stage):
    i = 1
    while True: 
        # yield returns a generator
        yield load_data(data_path, i-1, batch_size, variant, stage) 

        if i < steps:
            i += 1
        else:
            i = 1

def load_data(data_path, i, batch_size, variant, stage):
    
    with h5py.File(data_path, "r") as h5f:
        # window of samples slicing the dataset file
        X = h5f['board_'+stage][batch_size*9*i:batch_size*9*(i+1)]
        y = h5f['moved_'+variant+'_'+stage][batch_size*i:batch_size*(i+1)]

    # reshape the data for proper input
    X = np.array(X).reshape(int(len(X)/9),9,8,12)
    y = np.array(y)

    return (X, y)
