import sys

import models

model = models.get_model(sys.argv[1])
model.train(int(sys.argv[2]))
