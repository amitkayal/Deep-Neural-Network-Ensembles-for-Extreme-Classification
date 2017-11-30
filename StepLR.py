class StepLR():

    def get_rate(self, epoch):
        learning_rate = 0.01
        if epoch > 2.0:
            learning_rate = 0.001
        if epoch > 3.0:
            learning_rate = 0.0001
        return learning_rate