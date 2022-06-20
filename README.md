# transformer-with-co-teaching
1) load_public_data.py对csi数据(.csv格式)进行处理，先将data和label组合，接着切割成train_data(0.8比例)和validation_data(0.2比例)，并给trian_data分别加入三种噪声（pairflip_0.45, symmetric_0.20, symmetric_0.50），validation_data不做处理。
2) transformer-csi可以查看HARTrans、TransCNN和TransformerM的表现性能，只是THAT论文中的实现过程，没有co-teaching实现。
3) transformer-co加入了co-teaching的实现。
