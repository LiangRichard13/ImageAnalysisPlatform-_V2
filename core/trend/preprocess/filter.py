import os
from datetime import datetime
import pandas as pd


path = '/mnt/share/wry/2023-10-12(17.36)/'
file_lists = os.listdir(path)
folders = ["1", "2", "3", "4"]
df = pd.DataFrame({"time": [], "lane1": [], "lane2": [], "lane3": [], "lane4": []})
count = 0
# pop a image
for filename in file_lists:
    # print(filename)
    _, _, time = filename.split("_")
    # folder = int(folder)
    # folders_other = folders.copy().remove(folder)
    h, m, s, _, _ = time.split(".")
    subtime = h + "." + m + "." + s
    time_format = datetime.strptime(subtime, "%H.%M.%S").strftime("%H:%M:%S")

    # find all images with the same time stamp (s)
    images = []
    for f in file_lists:
        if subtime in f:
            images.append(f)

    # # if not all lanes containing a image
    # if len(images) < 4:
    #     # pop all iamges
    #     for image in images:
    #         file_lists.remove(image)
    # else:

    # times.append(time_format)
    # df['time'][0] = time_format
    df.loc[len(df)] = [time_format, "", "", "", ""]
    folders_temp = folders.copy()
    for image in images:
        fold, _, _ = image.split("_")
        if fold in folders_temp:
            # swich case to put image into folder
            if fold == "1":
                # folder1.append(image)
                df["lane1"][count] = image
            elif fold == "2":
                # folder2.append(image)
                df["lane2"][count] = image
            elif fold == "3":
                # folder3.append(image)
                df["lane3"][count] = image
            else:
                # folder4.append(image)
                df["lane4"][count] = image
            folders_temp.remove(fold)
        file_lists.remove(image)
    count += 1

    # print(images_other)
# Sort the timelist in ascending order
# sorted_dates = sorted(times)
df = df.sort_values(by="time")
df.to_excel("/mnt/share/wry/data_with_blank.xlsx", index=False)

# print(type(folder), time_format)
df = pd.read_excel("/mnt/share/wry/data_with_blank.xlsx")
# 使用上一行的数据填充空值
df_filled = df.fillna(method="ffill")
# df_filled["lane2"][0] = df_filled["lane2"][1]
df_filled.to_excel("/mnt/share/wry/data_insertion.xlsx", index=False)
