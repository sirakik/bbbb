# ※※※工事中※※※　
もう少し編集する必要あり

# ナニコレ  
自分用のst-gcn実装  
忠実な実装ではないです．  

※NTU-RGB+Dのみ対応

# 実行

## データセット準備
``dataset_utils/gen_data_ntu_rgb_d.py``の中身編集
```
origin_path = '../ntu-rgb+d-skeletons60'
output_dir_path = 'Data/NTU-RGB+D60'
ignore_sample_path = 'Tools/Gen_dataset/ignore_sample_ntu60.txt'
```
を自分の環境に合わせる．

```
python3 dataset_utils/gen_data_ntu_rgb_d.py
```

``config/xsub.yaml, config/xview``内のデータセットのパスを合わせる

## train
- cross subject  
```
python3 train.py --config config/xsub.yaml
```
- cross view
```
python3 train.py --config config/xview.yaml
```

## test
- cross subject  
```
python3 test.py --config config/xsub.yaml
```
- cross view
```
python3 test.py --config config/xview.yaml
```


# 再現度
###NTU-RGB+D  

|      | x-sub[\%] | x-view[%] |
| :--- | :---: | :---: |
| 論文　| 81.5  | 88.3  |
|作者実装(古いVer.)| xx.x | xx.x |
| 白木実装　| xx.x  | xx.x  |


# 参考
- [paper](https://arxiv.org/pdf/1801.07455.pdf): ST-GCN(arXiv)  
- [github](https://github.com/yysijie/st-gcn): 現行実装  
- [古code](): 作者による古い実装(この頃が一番わかりやすい)  
- [sta-gcn](https://github.com/machine-perception-robotics-group/SpatialTemporalAttentionGCN): Spatial Temporal Attention GCN 
  - [paper](https://openaccess.thecvf.com/content/ACCV2020/html/Shiraki_Spatial_Temporal_Attention_Graph_Convolutional_Networks_with_Mechanics-Stream_for_Skeleton-based_ACCV_2020_paper.html)
  \[k.shiraki, ACCV2020\]
  
# memo
(実装方法に古臭さを感じる)