![image](https://github.com/tircra3/model/raw/main/model_img/TIRClogo.png)
# TIRC Model
## 程式環境：python3.7.13 <br>
## step1：<br>
開啟cmd後cd至model_UI資料夾，安裝`requirements.txt` <br>
`pip install -r requirements.txt` <br>
## step2：<br>
執行model_ui_local.py <br>
`python model_ui_local.py`
## step3：<br>
使用UI介面 <br>
![image](https://github.com/tircra3/model/raw/main/model_img/UI.png)

### 1. Click "Nii Package Location" <br>
Step1:選擇您的nii檔案存放的資料夾 <br>
格式請設定為: <br>
`(「choose this file」 / 「PTID」 / * / 「FOLDER」 / 「EXAMDATE」 / * / *.nii ))` <br>
Step2:選擇您的xlsx檔案存放位置 <br>
Step3:等待儲存結果 <br>

### 2. Choose Testing Model <br>
選擇測試的模型: <br>
Step1:選擇您的pickle檔案 <br>
Step2:選擇您的npy檔案 <br>
Step3:等待預測結果 <br>


#### 選擇測試的模型: <br>
1. `BrainAge` <br>
預測結果可獲得MRI影像之大腦年齡。 <br>
2. `NIFD_2ndclass` <br>
預測結果可獲得NIFD次分類，可分為BV、SV、PNFA、L_SD等四類。 <br>
3. `ADNI&NIFD_classification` <br>
預測結果可獲得ADNI與NIFD等兩分類結果。 <br>

4. `CN_MCI_other` <br>
預測結果可獲得CN、MCI與剩下其餘失智階段(other)等三種分類結果 <br>

失智分數(CDR-SB)根據分數大小可分為六種階段如下： <br>
![image](https://github.com/tircra3/model/raw/main/model_img/CDR-SB.png)

5. `CDRSB_6class` <br>
預測結果可獲得CDR-SB六階段分類結果 <br>
6. `CDRSB_6reg` <br>
預測結果可獲得CDR-SB六階段分類結果 <br>
7. `3111class` <br>
將前三類分為一組，剩餘三類各一組，預測結果可獲得CDR-SB的四種分類結果。 <br>
8. `2112class` <br>
將前兩類分為一組，三四類各分一組，後兩類分為一組，預測結果可獲得CDR-SB的四種分類結果。 <br>
