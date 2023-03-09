import pandas as pd
import numpy as np
import cv2
import os
import copy

def split_csv(labels,images,split):
    digits = 5
    CLASS_NAMES = [
            'basophil', 'eosinophil', 'erythroblast', 'immature granulocytes', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet', 'No Finding'
                    ]
    csv_list = []
    for i in range(len(labels)):
        fname = split+ "_" + "0"*(digits-len(str(i)))+str(i)+".png"
        img = images[i]
        cv2.imwrite(os.path.join(f"images\\",fname),img)
        csv_list.append([fname,CLASS_NAMES[labels[i]]])

    df = pd.DataFrame(csv_list,columns=["Image Index","Finding Labels"])
    df.to_csv(f'{split}.csv')


data = np.load("bloodmnist.npz")
split = "val"
labels = [int(i) for i in data[f"{split}_labels"]]
images = copy.deepcopy(data[f"{split}_images"])
split_csv(labels,images,split)