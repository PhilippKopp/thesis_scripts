#!/usr/bin/env python3.5

import os
import math
import sys
import numpy as np
import cnn_db_loader




experiment_folder='21'

cnn_db_loader.NUMBER_ALPHAS = 0
cnn_db_loader.NUMBER_IMAGES = 1
cnn_db_loader.NUMBER_XYZ = 0

Experint_BASE = '/user/HS204/m09113/my_project_folder/cnn_experiments/'
experiment_dir = Experint_BASE+experiment_folder
db_dir = experiment_dir+'/db_input/'


db_loader = cnn_db_loader.lazy_dummy(db_dir)
db_loader.show_isomaps()
exit(0)