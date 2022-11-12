import pandas as pd
import numpy as np
from tqdm import tqdm




def inspect_category_wise_data(label, n=5):
    samples = train[train[label] == 1].sample(n)
    sentiment = mapping[label]
    
    print(f"{n} samples from {sentiment} sentiment: \n")
    for text in samples["text"]:
        print(text, end='\n\n')