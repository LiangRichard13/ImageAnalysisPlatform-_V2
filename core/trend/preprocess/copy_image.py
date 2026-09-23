# TODO:
# yolo v5 量化成 int4 / int8
# tensorRT 、PPQ

import os
import shutil
import pandas as pd

path = '/mnt/share/wry/2023-10-12(17.36)/'
dst = '/mnt/share/wry/silicon/'
# chmod 777 dst + f in folders
folders = ["1", "2", "3", "4"]
# folders = ["1", "4"]

# df = pd.read_excel("data_with_blank.xlsx")
# # store list folder into local folder
# for f in folders:
#     for i in df["lane" + f]:
#         if pd.isna(i):
#             continue
#         else:
#             shutil.copy(path + i, dst + f + "/" + i)
#             print(i + ' is copied to folder No.' + f + '.')


# rename images in folder with time!
df = pd.read_excel('/mnt/share/wry/data_insertion.xlsx')
for f in folders:
    for i in range(len(df["lane" + f])):
        shutil.copyfile(path + df["lane" + f][i], dst + f + "/images/" + df["time"][i] + ".jpg")
        # shutil.copyfile(path + df["lane" + f][i], dst + f + df["time"][i] + ".jpg")
        print(df["lane" + f][i] + ' is copied to folder No.' + f + '.')

        # if i < len(df["lane" + f]) and df["lane" + f][i] == df["lane" + f][i + 1]:
        #     shutil.copy(
        #         dst + f + "/" + df["lane" + f][i],
        #         dst + f + "/" + df["time"][i] + ".jpg",
        #     )
        # else:
        #     os.rename(
        #         dst + f + "/" + df["lane" + f][i],
        #         dst + f + "/" + df["time"][i] + ".jpg",
        #     )
