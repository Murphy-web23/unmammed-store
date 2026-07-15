# 無人商店 AI 自助結帳系統

這是三人小組的 Python / OpenCV / MediaPipe 專題，目標是建立一套可展示、可維護、可擴充的無人商店自助結帳系統。系統目前已完成會員辨識、商品辨識、購物車、折扣結帳、管理員維護，以及 YOLO 滿額贈品辨識展示流程；後續仍可擴充 YOLO 鈔票辨識，讓結帳流程更接近完整自助收銀情境。

本 README 已整合目前程式狀態與 `report.txt` 的技術整理。

## 專題目標

- 顧客可透過鏡頭掃描會員身份。
- 非會員可選擇現場加入會員，或直接以原價結帳。
- 商品可透過鏡頭掃描，系統以 OpenCV 多特徵比對加入購物車。
- 系統依會員等級套用折扣並計算結帳金額。
- 結帳後可依金額門檻啟動 YOLO 贈品辨識，展示茶與牛奶的信心值對比。
- 管理員可維護會員與商品資料，新增商品時可拍攝參考照片。
- 在辨識不確定時保留人工確認流程，降低 demo 中斷風險。

## 目前功能

### 會員流程

- 掃描會員臉部並辨識會員身份。
- 查無會員時，自動切換為非會員身份。
- 非會員可選擇加入會員。
- 加入會員時可輸入姓名與會員等級，並拍攝會員臉部照片。
- 會員建立完成後，重新掃描即可套用折扣。

### 商品流程

- 掃描商品時會開啟鏡頭畫面。
- 掃描畫面使用鏡像顯示，操作方向與使用者視角一致。
- 系統會先擷取商品主體 ROI，降低桌面或背景影響。
- 以 OpenCV 多特徵比對商品參考照片，取得最可能的 `class_name`。
- 若辨識結果不確定，會顯示候選商品並詢問是否仍加入購物車。
- 購物車內商品可單筆刪除。

### 結帳流程

- 顯示原價總金額與折扣後總金額。
- 支援非會員、一般會員、學生會員、VIP 會員折扣。
- 金額使用 `Decimal` 與 `ROUND_HALF_UP` 做一般四捨五入。
- 按下「掃描完成 / 結帳」後會先顯示結帳結果。
- 結帳完成提示關閉後，若折扣後金額達到滿額門檻，系統會詢問是否兌換贈品。
- 贈品流程結束後清空購物車。

### YOLO 滿額贈品流程

- 折扣後金額滿 `100` 元，可兌換茶。
- 折扣後金額滿 `150` 元，可兌換牛奶，並優先於滿 100 元贈品。
- 使用者選擇兌換後，系統會啟動 YOLO 模型 `drinks.pt` 辨識贈品。
- YOLO 視窗會即時顯示 Tea 與 Milk 的信心值與條狀對比。
- 辨識到正確贈品後，畫面仍會持續即時更新，不會停在第一個偵測到的角落畫面。
- 展示者可調整商品位置，確認大家都看到完整畫面後，按 `Enter` 或 `Space` 結束 YOLO 視窗。
- 按 `Esc` 可取消或關閉 YOLO 辨識視窗。

### 管理員功能

- 主畫面提供「管理員操作」入口。
- 預設 demo 密碼為 `1234`。
- 會員管理：新增、更新、刪除、重新整理。
- 商品管理：新增、更新、刪除、重新整理。
- 新增商品時會啟動攝影機，拍攝多張不同角度參考照片。
- 會員與商品表格支援水平捲動，方便查看完整欄位。

## 使用技術

| 類別 | 技術 |
|---|---|
| 程式語言 | Python 3 |
| 桌面介面 | Tkinter / ttk |
| 影像處理 | OpenCV |
| 人臉辨識 | MediaPipe Face Landmark + 臉部外觀特徵向量 |
| 商品辨識 | OpenCV 多特徵加權比對 |
| 贈品辨識 | Ultralytics YOLO + `drinks.pt` |
| 中文提示 | Pillow 繪製中文文字，避免 OpenCV 中文亂碼 |
| 資料儲存 | CSV |
| 檔案管理 | pathlib / dataclass |
| 版本控制 | Git、`.editorconfig`、`.gitattributes` |

## 會員辨識方法

會員辨識以 MediaPipe Face Landmark 為核心，搭配臉部外觀特徵向量進行比對。系統不只看單一臉部位置，而是使用 landmark 與 appearance 的組合距離，提升不同角度、表情與光線下的穩定度。

會員照片資料預設放在：

```text
src/face/
```

會員資料表：

```text
src/unmanned_store/data/members.csv
```

實際組員或測試者的人臉照片不建議 commit 到 GitHub。

## 商品辨識方法

商品辨識目前不使用 YOLO，而是使用 OpenCV 特徵工程。流程如下：

1. 鏡頭拍攝商品影像。
2. 進行 Dynamic Color Balancing，降低偏色光源影響。
3. 在 HSV 的 V 通道套用 CLAHE，改善亮度不均。
4. 強化 HSV 綠色區間，用於分辨綠色與蒂芬妮綠商品。
5. 透過前景遮罩與輪廓框選商品主體 ROI。
6. 萃取多種特徵並加權評分。
7. 彙整同一類商品的 Top-N 參考圖分數。
8. 若分數不足或結果不確定，提示使用者重新掃描或人工確認。

目前使用的商品特徵：

- HSV 顏色直方圖
- 綠色遮罩直方圖
- Canny 邊緣向量
- 形狀向量
- 整體外觀向量
- ORB 局部特徵
- 雨傘 / 水壺幾何後處理規則
- 蒂芬妮綠 HSV 像素佔比規則

形狀向量會協助辨識：

- 長形
- 方形 / 矩形
- 有稜角
- 有尖角
- 圓滑或不規則外型

## 商品辨識調參位置

主要參數集中在：

```text
src/unmanned_store/utils/item_detector.py
```

常用參數：

| 參數 | 用途 |
|---|---|
| `OPENCV_MATCH_THRESHOLD` | 最低通過門檻，調高會更嚴格 |
| `OPENCV_MIN_MARGIN` | 第一名與第二名最低差距，調高可降低相似商品誤判 |
| `COLOR_WEIGHT` | 顏色特徵權重 |
| `EDGE_WEIGHT` | 邊緣特徵權重 |
| `SHAPE_WEIGHT` | 形狀特徵權重 |
| `APPEARANCE_WEIGHT` | 整體外觀權重 |
| `ORB_WEIGHT` | 局部細節權重 |
| `TIFFANY_RATIO_*` | 蒂芬妮綠佔比門檻 |

建議每個商品至少準備 8 到 15 張參考照片，包含正面、側面、斜角、遠近與不同光線。外型或顏色相近的商品，例如兩把雨傘、兩支筆，參考照片越多越穩定。

## 目前商品資料

商品資料表：

```text
src/unmanned_store/data/products.csv
```

目前已建檔商品：

| product_id | 商品名稱 | 價格 | class_name |
|---|---:|---:|---|
| P001 | Asahi 啤酒 | 49 | `asahi` |
| P003 | 藍色原子筆 | 15 | `pen_blue` |
| P004 | Tiffany 色原子筆 | 15 | `pen_tiffany` |
| P005 | 綠色雨傘 | 199 | `umbrella_green` |
| P006 | Tiffany 色雨傘 | 199 | `umbrella_tiffany` |

商品照片資料夾：

```text
src/item/
├── asahi/
├── pen_blue/
├── pen_tiffany/
├── umbrella_green/
└── umbrella_tiffany/
```

新增商品時，`products.csv` 的 `class_name` 必須和 `src/item/<class_name>/` 資料夾名稱一致。

## 資料夾結構

```text
src/unmanned_store/
├── main.py
├── data/
│   ├── members.csv
│   └── products.csv
└── utils/
    ├── admin_panel.py
    ├── checkout.py
    ├── face_recognizer.py
    ├── gift_detector.py
    ├── item_detector.py
    ├── member_manager.py
    ├── member_register.py
    └── product_manager.py

src/unmanned_store/yolo_gift_test.py

model/
└── drinks.pt

tools/
└── yolo_objectdetection_sv_2.py

src/face/
└── .gitkeep

src/item/
├── asahi/
├── pen_blue/
├── pen_tiffany/
├── umbrella_green/
└── umbrella_tiffany/
```

## 安裝與執行

建議先建立並啟用虛擬環境。

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\unmanned_store\main.py
```

一般 Python 環境：

```bash
pip install -r requirements.txt
python src/unmanned_store/main.py
```

目前必要套件：

```text
opencv-python
pillow
pandas
mediapipe
ultralytics
supervision
```

`model/drinks.pt` 為 YOLO 贈品辨識模型，因 `.gitignore` 已忽略 `model/`，不會直接上傳到 GitHub。若在新環境執行，需自行將模型放回：

```text
model/drinks.pt
```

## Demo 流程

### 流程 1：非會員原價結帳

1. 顧客按「掃描會員」。
2. 系統查無會員資料。
3. 顧客選擇不加入會員。
4. 系統以非會員身份結帳。
5. 掃描商品並加入購物車。
6. 按「掃描完成 / 結帳」。
7. 顯示原價總金額。

### 流程 2：既有會員折扣結帳

1. 既有會員按「掃描會員」。
2. 系統辨識成功並顯示會員等級。
3. 掃描商品並加入購物車。
4. 按「掃描完成 / 結帳」。
5. 顯示折扣後總金額。

### 流程 3：現場加入會員

1. 顧客按「掃描會員」。
2. 系統查無會員資料。
3. 顧客選擇加入會員。
4. 輸入姓名並選擇會員等級。
5. 拍攝會員臉部照片。
6. 系統寫入 `members.csv`。
7. 重新掃描會員。
8. 掃描商品並結帳。

### 流程 4：管理員新增商品

1. 按「管理員操作」。
2. 輸入密碼 `1234`。
3. 切換到「商品管理」。
4. 輸入商品名稱、價格與 `class_name`。
5. 按「新增商品」並拍攝多張參考照片。
6. 系統建立商品資料與 `src/item/<class_name>/` 參考圖。

### 流程 5：結帳後兌換滿額贈品

1. 掃描所有商品並加入購物車。
2. 按「掃描完成 / 結帳」。
3. 系統先顯示結帳結果。
4. 若折扣後金額滿 `100` 元，系統詢問是否兌換茶。
5. 若折扣後金額滿 `150` 元，系統詢問是否兌換牛奶。
6. 使用者選擇兌換後，YOLO 視窗會顯示茶與牛奶的即時信心值對比。
7. 辨識到正確贈品後，畫面仍保持即時更新，展示者按 `Enter` 或 `Space` 後結束辨識。

## report.txt 重點整理

`report.txt` 記錄了本專題從 MVP 到目前版本的演進：

- 第一階段先完成可操作的 Tkinter 主流程。
- 第二階段建立 MediaPipe Face Landmark 會員辨識。
- 第三階段以 OpenCV 多特徵工程建立商品辨識。
- 第四階段加入 YOLO 滿額贈品辨識，作為結帳後展示流程。
- 針對光線與背景問題加入色彩平衡、HSV CLAHE 與前景 ROI。
- 針對相似商品加入不確定結果處理，降低 demo 失敗率。
- 針對雨傘與水壺加入 Canny 幾何規則。
- 針對綠色與蒂芬妮綠商品加入 HSV 像素佔比規則。
- 新增管理員維護功能，支援會員與商品資料長期管理。
- 建議後續可評估 MobileNetV3 Small embedding 或 YOLO 方案提升辨識能力。

## 後續規劃：YOLO 鈔票辨識

下一階段預計新增 YOLO 鈔票辨識功能，讓系統除了辨識商品，也能在付款階段辨識紙鈔面額。

規劃方向：

- 新增鈔票辨識模組，例如 `banknote_detector.py`。
- 建立鈔票資料集，類別可包含 `100`、`500`、`1000` 等面額。
- 使用 YOLO 模型偵測畫面中的鈔票位置與面額。
- 將辨識到的面額加總為已付款金額。
- 與購物車折扣後金額比對，顯示付款成功、付款不足或應找零金額。
- 保留人工確認流程，避免模型誤判直接影響結帳。

預計資料夾：

```text
model/
└── banknote_yolo.pt

src/banknote/
├── train/
├── valid/
└── data.yaml
```

注意：目前 `requirements.txt` 已因贈品辨識加入 `ultralytics` 與 `supervision`。若後續開發鈔票辨識，可沿用同一套 YOLO 推論環境，或依課程環境改用其他 YOLO 推論方式。

## 已知限制

- OpenCV 商品辨識仍會受光線、角度、背景與鏡頭品質影響。
- 顏色相近或外型相近的商品需要更多參考照片。
- 目前商品辨識偏向 demo 與小型商品庫，不適合直接用於大型真實商店。
- YOLO 贈品辨識模型目前針對茶與牛奶展示情境，需在模型可辨識的類別與光線條件下使用。
- 會員照片屬個資，不建議上傳真實人臉資料到公開 GitHub。
- YOLO 鈔票辨識仍屬後續規劃，尚未整合進目前主流程。

## 隱私與版本控制

- `.gitignore` 已忽略 `src/face/*`，只保留 `src/face/.gitkeep`。
- `.gitignore` 已忽略 `model/`，避免 `drinks.pt` 等大型模型檔誤上傳。
- `src/unmanned_store/temp/` 與拍攝暫存檔不會提交。
- `.editorconfig` 與 `.gitattributes` 用於統一文字檔編碼與換行，降低 clone 後中文亂碼風險。

## 專題結論

本專題目前已從「流程可跑」提升到「現場可展示、資料可維護、功能可擴充」的狀態。會員辨識使用 MediaPipe，商品辨識使用 OpenCV 多特徵加權與幾何規則，管理員功能可維護會員與商品資料；結帳後的 YOLO 贈品辨識則讓展示流程能呈現模型信心值對比。後續加入 YOLO 鈔票辨識後，系統會更接近完整的無人商店自助結帳流程。
