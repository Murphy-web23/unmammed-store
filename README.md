# 無人商店 AI 自助結帳系統

## 專題簡介

這是一個結合 Tkinter、OpenCV、MediaPipe Face Landmark 的無人商店自助結帳系統。系統以「能穩定 demo」為優先，正式辨識流程之外，也保留測試模式，避免現場攝影機、模型或光線不穩造成展示中斷。

系統可以：

- 掃描人臉辨識會員
- 根據會員等級套用折扣
- 非會員可以選擇不加入並以原價結帳
- 非會員也可以現場加入會員
- 加入會員時會倒數拍攝三張照片
- 使用 OpenCV 比對商品參考照片
- 加入購物車
- 計算折扣後總金額
- 折扣後金額會自動四捨五入

## 使用技術

- Python：主要程式語言
- Tkinter：GUI 介面
- OpenCV：攝影機、拍照、影像處理
- MediaPipe Face Landmark：會員人臉特徵點辨識
- OpenCV：攝影機、拍照、影像處理與商品參考照片比對
- CSV：會員資料與商品資料儲存
- Decimal：折扣金額四捨五入

## 系統功能

### 會員辨識功能

- 既有會員掃臉後顯示會員等級與折扣
- 非會員掃臉後顯示查無會員
- 非會員可以選擇是否加入會員
- 非會員也可以繼續掃商品並以原價結帳

### 會員申請功能

- 輸入姓名
- 選擇會員等級
- 倒數拍三張照片
- 自動建立 `src/face/general_member_XXX/` 資料夾
- 自動寫入 `src/unmanned_store/data/members.csv`
- 加入成功後請重新掃描會員，再套用會員折扣

### 商品結帳功能

- 掃描商品
- 對應 `products.csv` 的商品名稱與價格
- 加入購物車
- 根據會員折扣計算總金額
- 折扣後金額四捨五入

### 測試模式

- 測試非會員
- 測試學生會員
- 測試一般會員
- 測試加入商品
- 檢查資料庫
- 測試 Landmark 單次辨識
- 測試 OpenCV 單次辨識
- 重置現場會員

## 會員折扣規則

| 會員身份 | discount | 顯示折扣 |
|---|---:|---|
| 非會員 | 1.0 | 原價 |
| 一般會員 | 0.9 | 9 折 |
| 學生會員 | 0.85 | 85 折 |
| VIP 會員 | 0.8 | 8 折 |

折扣後總金額使用 `Decimal` 與 `ROUND_HALF_UP` 四捨五入成整數，避免 Python `round()` 在 `.5` 時不是一般四捨五入的狀況。

## demo 角色設計

### A 同學

- 不在會員資料庫
- 掃臉後顯示非會員
- 選擇不加入會員
- 以原價結帳

### B 同學

- 事先建立為學生會員
- 掃臉後顯示學生會員
- 結帳套用 85 折

### C 同學

- 一開始不在會員資料庫
- 掃臉後顯示非會員
- 選擇加入會員
- 現場拍三張照片
- 建立一般會員
- 再次掃臉後套用 9 折

## 資料夾結構

```text
src/unmanned_store/
├── main.py
├── data/
│   ├── members.csv
│   └── products.csv
├── utils/
│   ├── member_manager.py
│   ├── member_register.py
│   ├── face_recognizer.py
│   ├── product_manager.py
│   ├── item_detector.py
│   └── checkout.py

src/face/
└── student_member/

src/item/
├── asahi/
├── owala/
├── pen_blue/
├── pen_tiffany/
├── umbrella_green/
└── umbrella_tiffany/
```

## 執行方式

安裝套件：

```bash
pip install -r requirements.txt
```

或手動安裝：

```bash
pip install opencv-python mediapipe pillow pandas
```

執行主程式：

```bash
python src/unmanned_store/main.py
```

## demo 前準備

### 會員資料

- A 同學不能放進 `members.csv` 和 `src/face/`
- B 同學照片要放入 `src/face/student_member/`
- `members.csv` 要有 B 同學學生會員資料
- C 同學 demo 前不能存在 `members.csv` 和 `src/face/`
- demo 前可按「重置現場會員」，清掉 C 同學現場加入後留下的資料

B 學生會員資料範例：

```csv
member_id,name,level,discount,face_folder
M001,學生會員示範者,學生會員,0.85,src/face/student_member
```

### 商品資料

- 商品參考照片要依分類放在 `src/item/`
- `products.csv` 的 `class_name` 要跟 `src/item/` 底下的資料夾名稱完全一致
- demo 商品建議每類準備 2 到 4 張即可

商品資料範例：

```csv
product_id,name,price,class_name
P001,Asahi 啤酒,49,asahi
P002,Owala 水壺,899,owala
P003,藍色原子筆,15,pen_blue
P004,Tiffany 色原子筆,15,pen_tiffany
P005,綠色雨傘,199,umbrella_green
P006,Tiffany 色雨傘,199,umbrella_tiffany
```

如果資料夾名稱是 `pen_blue`，`products.csv` 的 `class_name` 就要寫 `pen_blue`。

## 正式 demo 順序

### 流程 1：A 同學，非會員原價結帳

1. A 同學按「掃描會員」
2. 系統掃臉
3. 查無會員資料
4. 選擇不加入會員
5. GUI 顯示非會員、折扣原價
6. 掃描商品並加入購物車
7. 按「掃描完成 / 結帳」
8. 結帳結果顯示原價總金額與折扣後金額相同

### 流程 2：B 同學，學生會員 85 折結帳

1. B 同學按「掃描會員」
2. 系統辨識成功
3. GUI 顯示學生會員、85 折
4. 掃描商品並加入購物車
5. 按「掃描完成 / 結帳」
6. 結帳結果顯示 85 折與四捨五入後金額

### 流程 3：C 同學，現場加入一般會員 9 折結帳

1. C 同學按「掃描會員」
2. 查無會員資料
3. 選擇加入會員
4. 輸入姓名，會員等級選擇「一般會員」
5. 倒數拍三張照片
6. 寫入 `members.csv`
7. 重新按「掃描會員」
8. GUI 顯示一般會員、9 折
9. 掃描商品並加入購物車
10. 按「掃描完成 / 結帳」
11. 結帳結果顯示 9 折與四捨五入後金額

## 注意事項與備用方案

- MediaPipe Face Landmark 可能受光線、角度、鏡頭距離影響
- OpenCV 商品比對可能受商品角度、光線與背景影響
- demo 當天如果辨識不穩，可以使用測試模式展示完整流程
- 「測試非會員」可展示 A 同學原價結帳
- 「測試學生會員」可展示 B 同學 85 折結帳
- 「測試一般會員」可展示 C 同學 9 折結帳
- 「測試加入商品」可在 OpenCV 或攝影機失敗時直接加入商品，仍可完成結帳流程
- 「檢查資料庫」可確認 CSV、face 資料夾、商品參考照片與商品 class_name

## 商品辨識調整建議

- 商品參考照片放在 `src/item/<class_name>/`，`products.csv` 的 `class_name` 必須和資料夾名稱一致。
- 商品辨識參數集中在 `src/unmanned_store/utils/item_detector.py` 上方：
  - `OPENCV_MATCH_THRESHOLD`：辨識通過門檻，調高會比較嚴格，調低會比較容易誤判。
  - `OPENCV_MIN_MARGIN`：第一名和第二名的最低差距，調高可避免相似商品誤判。
  - `COLOR_WEIGHT`、`EDGE_WEIGHT`、`APPEARANCE_WEIGHT`、`ORB_WEIGHT`：分別控制顏色、輪廓、整體外觀、局部細節的比重。
- 每個商品建議至少 8 到 15 張照片，包含正面、側面、斜角、遠近、不同光線；背景盡量單純，商品要佔畫面大部分。
- 顏色相近或外型相近的商品，例如兩支筆、兩把傘，最需要補不同角度照片；如果照片太少，系統會顯示「商品辨識不確定」並要求重掃。

## 隱私提醒

- 實際組員的人臉照片不建議 commit 到 GitHub
- 商品照片可以放在 `src/item/`，若涉及私人或授權素材，請確認再 commit
- demo 用的 `src/face/` 資料夾可以只存在本機
- 本專案 `.gitignore` 已忽略 `src/face/*` 與 `model/`，只保留 `src/face/.gitkeep`
