.show(): 直接開啟視窗顯示繪製後的圖片。
.save(filename="result.jpg"): 將繪製後的圖片儲存到磁碟。
results[0].boxes.xyxy: 取得左上角與右下角座標 [x1, y1, x2, y2]。
results[0].boxes.xywh: 取得中心點座標與寬高 [x, y, w, h]。
results[0].boxes.conf: 取得置信度 (Confidence score)。
results[0].boxes.cls: 取得類別 ID。
results[0].names: 取得該模型所有的類別名稱字典（例如 {0: 'person', 1: 'bicycle', ...}）。
results[0].orig_img: 取得原始未處理的 numpy 圖片。