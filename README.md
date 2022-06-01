# transformer-with-co-teaching
①load_public_data.py对csi数据(.csv格式)进行处理，将data和label组合成了"Data.pt"文件，以供后续validation载入。
②在load_public_data.py文件中加入noise.py模块，将label打乱后再与data组合成"Data_noise.pt"文件，以供后续train载入。
transformer-csi可以查看HARTrans、TransCNN和TransformerM的表现性能，只是THAT论文中的实现过程，没有co-teaching实现。
transformer-co加入了co-teaching的实现。
