import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

df = pd.read_csv("Data/globalterrorismdb_0718dist.csv", encoding = "ISO-8859-1", low_memory = False)
df.head()