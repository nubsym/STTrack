import csv
import os

import lmdb
from lib.train.dataset import Got10k


env = lmdb.open('/home/luohui/SYM/tracking/code/temporal/data/got10k_lmdb/got10k.lmdb', map_size=int(1e9))
txn = env.begin(write=True)

dataset_root='/home/luohui/SYM/tracking/data/got-10k/train'
dataset = Got10k(dataset_root, split='train_full')

# with open(os.path.join(dataset_root, 'list.txt')) as f:
#     dir_list = list(csv.reader(f))
# dir_list = [dir_name[0] for dir_name in dir_list]
# seq_ids = list(range(0, len(dir_list)))
# sequence_list = [dir_list[i] for i in seq_ids]

for seq in dataset:
    seq_id = seq.__class__.__name__ + '_' + seq.sequence_name

    for frame_id, (img, anno) in enumerate(seq):
        # 将图像和标注转换为字符串
        img_data = img.tobytes()
        anno_data = anno.tostring()

        # 生成键
        key = f'{seq_id}_{frame_id:08}'.encode()

        # 将图像和标注存储到LMDB数据库中
        txn.put(key, img_data, overwrite=False)
        txn.put(key + b'_anno', anno_data, overwrite=False)

# 提交写入事务并关闭数据库
txn.commit()
env.close()