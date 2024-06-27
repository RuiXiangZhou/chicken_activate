# 土雞活動力偵測模型
## chicken activate

此模型能夠偵測影片中的雞隻活動力  
模型可視化展示如下圖:  
![m6](https://hackmd.io/_uploads/HkC_W9l26.jpg)

![m7](https://hackmd.io/_uploads/BkxA_Z9l3a.jpg)

電腦需求: 需配有Nvidia 顯卡，並已安裝cuda、anaconda  
輸入: 影片  
輸出: 活動力數值  
使用步驟:  
0. 使用anaconda建立一個虛擬環境  
1. 根據安裝的cuda版本下載pytorch，網址:https://pytorch.org/get-started/locally/  
2. 於連結下載檔案並解壓縮獲得一資料夾  
3. 將anaconda終端所在位置移動至檔案資料夾  
4. 執行pip install -r requirement.txt
5. 下載辨識模型[**chicken_model**](https://github.com/RuiXiangZhou/chicken_activate/releases/download/model/best_chi.pt)並置於chicken_activate資料夾中
6. 將欲辨識的影片[**example_video**](https://github.com/RuiXiangZhou/chicken_activate/releases/download/model/Generic_DAHUA-001-20230616-154321-1686901401534-7.mp4)放在"sourse"資料夾中  
7. 執行python rtsp_yolo7_1.py  
8. 辨識結果的數值將存在"result"資料夾，存檔檔名為當時時間的文字文件檔案  


## Acknowledgements
https://github.com/WongKinYiu/yolov7/
