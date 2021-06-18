#   Author: Emil Møller Hansen


class BaseTrainer():

    def __init__(self, conf):
        self.conf = conf

    # Function called from evaluate.py, should evaluate the model on a given dataset
    def evaluate(self):
        pass

    # Function called from predict.py, should make the model
    # produce predictions on a given data set
    def predict(self):
        pass

    # this function is called from the train.py script, so it should train the model,
    # and possibly evaluate on a test set afterwards
    def run(self):
        pass
