import sys

import models

BATCH_SIZE = 64

model = models.get_model(sys.argv[1])
model.train(int(sys.argv[2]), BATCH_SIZE)

