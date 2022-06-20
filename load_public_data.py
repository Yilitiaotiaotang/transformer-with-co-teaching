import csv
import os
import torch
import torch.utils.data as Data
from tqdm import tqdm
import noise

"""this code is used to convert the public data which is downloaded from 
https://drive.google.com/file/d/19uH0_z1MBLtmMLh8L4BlNA0w-XAFKipM/view
to 'Data.pt'.
"""


def average_list(d_list):
    sum = [0.0 for _ in range(len(d_list[0]))]
    for j in range(len(d_list[0])):
        for i in range(len(d_list)):
            sum[j] += d_list[i][j]
        sum[j] /= len(d_list)
    return sum


def merge_timestamp(data, time_stamp):
    intervel = (time_stamp[len(time_stamp)-1] - time_stamp[0]) / 2000
    cur_range = time_stamp[0] + intervel
    temp_list = []
    new_data = []
    for i in range(len(time_stamp)):
        if time_stamp[i] > cur_range:
            if len(temp_list) != 0:
                new_data.append(average_list(temp_list))
            else:
                new_data.append(data[i])
            temp_list = []
            cur_range = cur_range + intervel
        temp_list.append(data[i])
    if len(temp_list) != 0:
        new_data.append(average_list(temp_list))
    if len(new_data) < 2000:
        new_data.append(data[len(time_stamp)-1])
        print("!!!!")
    return new_data[:2000]


def load_data(root):
    root = root + '\\'
    file_list = os.listdir(root)
    label = []
    data = []
    aclist = ['bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']  # 'bed' means 'lie down' in the paper
    #minsize = 15813
    for file in tqdm(file_list): #tqdm(file_list)
        if file.startswith("annotation"):
            continue
        with open(root + file, encoding='utf-8') as f:
            reader = csv.reader(f)
            record = []
            time_stamp = []
            for r in reader:
                record.append([float(str_d) for str_d in r[1:91]])
                time_stamp.append(float(r[0]))
            record = merge_timestamp(record, time_stamp)
            float_data = torch.tensor(record, dtype=torch.float32, requires_grad=False)
            data.append(float_data.unsqueeze(0))
            for j in range(len(aclist)):
                if file.find(aclist[j]) != -1:
                    label.append(j)
                    break
    data = torch.cat(data, dim=0)
    label = torch.tensor(label)
    # #####################################################
    # noise_type = 'pairflip' # 'pairflip' 'symmetric'
    # label, actual_noise_rate = noise.noisify(train_labels=label, noise_type=noise_type, noise_rate=0.45, nb_classes=7)
    # #####################################################
    # label = torch.from_numpy(label)
    data = Data.TensorDataset(data, label)
    ##################################分割data###################
    length = len(data)
    train_size = int(0.8 * length)
    validate_size = length - train_size
    train_set, validate_set = torch.utils.data.random_split(data,[train_size,validate_size])
    torch.save(validate_set, "validate_data.pt")

    ###################################加入noise_pairflip_0.45##################
    noise_type1 = 'pairflip' # 'pairflip' 'symmetric'
    train_clean_label1 = train_set.dataset.tensors[1]
    train_noise_label1, actual_noise_rate = noise.noisify(train_labels=train_clean_label1, noise_type=noise_type1, noise_rate=0.45, nb_classes=7)
    train_clean_data1 = train_set.dataset.tensors[0]
    train_noise_label1 = torch.from_numpy(train_noise_label1)
    train_data1 = Data.TensorDataset(train_clean_data1, train_noise_label1)
    torch.save(train_data1, "train_data_with_noise_pairflip_0.45.pt")
    ###################################加入noise_symmetric_0.20##################
    noise_type2 = 'symmetric' # 'pairflip' 'symmetric'
    train_clean_label2 = train_set.dataset.tensors[1]
    train_noise_label2, actual_noise_rate = noise.noisify(train_labels=train_clean_label2, noise_type=noise_type2, noise_rate=0.20, nb_classes=7)
    train_clean_data2 = train_set.dataset.tensors[0]
    train_noise_label2 = torch.from_numpy(train_noise_label2)
    train_data2 = Data.TensorDataset(train_clean_data2, train_noise_label2)
    torch.save(train_data2, "train_data_with_noise_symmetric_0.20.pt")
    ###################################加入noise_symmetric_0.50##################
    noise_type3 = 'symmetric' # 'pairflip' 'symmetric'
    train_clean_label3 = train_set.dataset.tensors[1]
    train_noise_label3, actual_noise_rate = noise.noisify(train_labels=train_clean_label3, noise_type=noise_type3, noise_rate=0.50, nb_classes=7)
    train_clean_data3 = train_set.dataset.tensors[0]
    train_noise_label3 = torch.from_numpy(train_noise_label3)
    train_data3 = Data.TensorDataset(train_clean_data3, train_noise_label3)
    torch.save(train_data3, "train_data_with_noise_symmetric_0.50.pt")     
    #############################################################    

    # torch.save(data, "Data_with_noise_pairflip_0.45.pt")
    # return data


if __name__ == '__main__':
    load_data(r"C:\Users\HP\Desktop\wifi-THAT\Dataset\Data")

