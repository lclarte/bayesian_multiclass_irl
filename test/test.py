import unittest
import sys
sys.path.append('..')

import numpy as np

from irl.dirichletprocess import *
from irl.niw import *
from irl.bp import *
from irl.environnement import *
from irl.sampling import *
from irl.inference import *