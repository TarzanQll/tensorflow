from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

def readMnist(path):
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets(path, one_hot=True)