# 無人商店 AI 自助結帳系統

## 專題簡介

這是一個結合 Tkinter、OpenCV、DeepFace、YOLO 的無人商店自助結帳系統。系統以「能穩定 demo」為優先，正式辨識流程之外，也保留測試模式，避免現場攝影機、模型或光線不穩造成展示中斷。

系統可以：

- 掃描人臉辨識會員
- 根據會員等級套用折扣
- 非會員可以選擇不加入並以原價結帳
- 非會員也可以現場加入會員
- 加入會員時會倒數拍攝三張照片
- 使用 YOLO 掃描商品
- 加入購物車
- 計算折扣後總金額
- 折扣後金額會自動四捨五入

## 使用技術

- Python：主要程式語言
- Tkinter：GUI 介面
- OpenCV：攝影機、拍照、影像處理
- DeepFace：會員人臉辨識
- YOLO / Ultralytics：商品辨識
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
- 測試 DeepFace 單次辨識
- 測試 YOLO 單次辨識
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

model/
└── best.pt
```

## 執行方式

安裝套件：

```bash
pip install -r requirements.txt
```

或手動安裝：

```bash
pip install opencv-python deepface ultralytics pillow pandas
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

- `best.pt` 要放在 `model/best.pt`
- 如果使用 Ultralytics classify 訓練路徑，也可以放在 `model/runs/classify/train/weights/best.pt`
- `products.csv` 的 `class_name` 要跟 YOLO 模型輸出完全一致
- demo 商品建議準備 2 到 4 個即可

商品資料範例：

```csv
product_id,name,price,class_name
P001,可樂,30,coke
P002,餅乾,25,cookie
P003,泡麵,45,noodle
P004,水,20,water
```

如果 YOLO 回傳 `cola`，`products.csv` 裡就要寫 `cola`，不能寫 `coke`。

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

- DeepFace 可能受光線、角度、鏡頭距離影響
- YOLO 可能受商品角度與光線影響
- demo 當天如果辨識不穩，可以使用測試模式展示完整流程
- 「測試非會員」可展示 A 同學原價結帳
- 「測試學生會員」可展示 B 同學 85 折結帳
- 「測試一般會員」可展示 C 同學 9 折結帳
- 「測試加入商品」可在 YOLO 或攝影機失敗時直接加入商品，仍可完成結帳流程
- 「檢查資料庫」可確認 CSV、face 資料夾、模型路徑與商品 class_name

## 隱私提醒

- 實際組員的人臉照片不建議 commit 到 GitHub
- `model/best.pt` 不建議直接 commit，除非確認檔案大小與授權沒問題
- demo 用的 `src/face/` 資料夾可以只存在本機
- 本專案 `.gitignore` 已忽略 `src/face/*` 與 `model/`，只保留 `src/face/.gitkeep`
