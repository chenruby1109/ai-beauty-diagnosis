# AI Beauty Diagnosis System - Enhanced Version

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import base64
import io
from datetime import datetime
from PIL import Image
from scipy.spatial import distance
from jinja2 import Template

st.set_page_config(
page_title=“AI 智能醫美面診輔助系統”,
page_icon=“🏥”,
layout=“wide”,
initial_sidebar_state=“collapsed”,
)

# ─────────────────────────────────────────

# 產品知識庫

# ─────────────────────────────────────────

PRODUCT_DB = {
“肉毒毒素_抬頭紋”: {“category”: “肉毒毒素”, “brands”: [“奇蹟肉毒”, “天使肉毒”, “寶提拉肉毒”], “dose”: {“輕度”: “10-15U”, “中度”: “15-25U”, “重度”: “25-40U”}, “dose_mid”: {“輕度”: 12, “中度”: 20, “重度”: 32}, “layer”: “額肌（肌肉層）”, “method”: “多點注射，間距約 1.5cm”, “effect”: “3-7 天見效，維持 4-6 個月”},
“肉毒毒素_眉間紋”: {“category”: “肉毒毒素”, “brands”: [“奇蹟肉毒”, “天使肉毒”, “寶提拉肉毒”], “dose”: {“輕度”: “10-15U”, “中度”: “15-25U”, “重度”: “25-40U”}, “dose_mid”: {“輕度”: 12, “中度”: 20, “重度”: 32}, “layer”: “皺眉肌（肌肉層）”, “method”: “5 點標準注射法”, “effect”: “3-7 天改善，維持 4-6 個月”},
“肉毒毒素_魚尾紋”: {“category”: “肉毒毒素”, “brands”: [“奇蹟肉毒”, “天使肉毒”], “dose”: {“輕度”: “5-10U/側”, “中度”: “10-15U/側”, “重度”: “15-20U/側”}, “dose_mid”: {“輕度”: 7, “中度”: 12, “重度”: 17}, “layer”: “眼輪匝肌”, “method”: “眼外角扇形多點注射”, “effect”: “5-7 天平滑，維持 3-5 個月”},
“肉毒毒素_下頜緣”: {“category”: “肉毒毒素”, “brands”: [“奇蹟肉毒”, “天使肉毒”], “dose”: {“輕度”: “10-20U”, “中度”: “20-30U”, “重度”: “30-50U”}, “dose_mid”: {“輕度”: 15, “中度”: 25, “重度”: 40}, “layer”: “頸闊肌”, “method”: “沿頸闊肌條索線性注射”, “effect”: “輪廓提升，維持 4-6 個月”},
“肉毒毒素_國字臉”: {“category”: “肉毒毒素”, “brands”: [“奇蹟肉毒”, “天使肉毒”], “dose”: {“輕度”: “20-30U/側”, “中度”: “30-40U/側”, “重度”: “40-60U/側”}, “dose_mid”: {“輕度”: 25, “中度”: 35, “重度”: 50}, “layer”: “咬肌（深層）”, “method”: “咬肌中下 1/3 定點注射”, “effect”: “2-4 週縮小，維持 6-12 個月”},
“肉毒毒素_小腿”: {“category”: “肉毒毒素”, “brands”: [“奇蹟肉毒”, “天使肉毒”, “寶提拉肉毒”], “dose”: {“輕度”: “50-80U/側”, “中度”: “80-100U/側”, “重度”: “100-150U/側”}, “dose_mid”: {“輕度”: 65, “中度”: 90, “重度”: 125}, “layer”: “腓腸肌內側頭（深層肌肉）”, “method”: “多點格狀注射，每點 5-10U”, “effect”: “4-8 週小腿線條改善，維持 6-9 個月”},
“VOLUMA_蘋果肌”: {“category”: “玻尿酸”, “brands”: [“喬雅登 VOLUMA”], “dose”: {“輕度”: “0.5-1ml/側”, “中度”: “1-1.5ml/側”, “重度”: “1.5-2ml/側”}, “dose_mid”: {“輕度”: “0.75ml”, “中度”: “1.25ml”, “重度”: “1.75ml”}, “layer”: “骨膜上層或深層皮下脂肪”, “method”: “扇形注射 / 線性推注”, “effect”: “蘋果肌圓潤，維持 18-24 個月”},
“VOLUMA_下巴”: {“category”: “玻尿酸”, “brands”: [“喬雅登 VOLUMA”], “dose”: {“輕度”: “0.5-1ml”, “中度”: “1-1.5ml”, “重度”: “1.5-2ml”}, “dose_mid”: {“輕度”: “0.75ml”, “中度”: “1.25ml”, “重度”: “1.75ml”}, “layer”: “骨膜上層”, “method”: “單點或扇形注射”, “effect”: “下巴延長翹挺，維持 12-18 個月”},
“VOLUMA_法令紋”: {“category”: “玻尿酸”, “brands”: [“喬雅登 VOLUMA”], “dose”: {“輕度”: “0.5-1ml/側”, “中度”: “1-1.5ml/側”, “重度”: “1.5-2ml/側”}, “dose_mid”: {“輕度”: “0.75ml”, “中度”: “1.25ml”, “重度”: “1.75ml”}, “layer”: “深層皮下 / 骨膜上層”, “method”: “逆行線性 + 扇形”, “effect”: “法令紋減少 60-80%，維持 12-18 個月”},
“VOLUX_下頜輪廓”: {“category”: “玻尿酸”, “brands”: [“喬雅登 VOLUX”], “dose”: {“輕度”: “1-2ml”, “中度”: “2-3ml”, “重度”: “3-4ml”}, “dose_mid”: {“輕度”: “1.5ml”, “中度”: “2.5ml”, “重度”: “3.5ml”}, “layer”: “骨膜上層”, “method”: “線性注射，沿下頜骨緣推注”, “effect”: “下頜輪廓清晰，維持 18-24 個月”},
“VOLIFT_法令紋”: {“category”: “玻尿酸”, “brands”: [“喬雅登 VOLIFT（豐麗緹）”], “dose”: {“輕度”: “0.8-1ml/側”, “中度”: “1-1.5ml/側”, “重度”: “1.5-2ml/側”}, “dose_mid”: {“輕度”: “0.9ml”, “中度”: “1.25ml”, “重度”: “1.75ml”}, “layer”: “真皮深層至皮下層”, “method”: “逆行線性注射 + 蕨葉技術”, “effect”: “法令紋平滑，維持 12-15 個月”},
“VOLBELLA_淚溝”: {“category”: “玻尿酸”, “brands”: [“喬雅登 VOLBELLA（夢蓓菈）”], “dose”: {“輕度”: “0.3-0.5ml/側”, “中度”: “0.5-1ml/側”, “重度”: “1-1.5ml/側”}, “dose_mid”: {“輕度”: “0.4ml”, “中度”: “0.75ml”, “重度”: “1.25ml”}, “layer”: “眶隔前脂肪層 / 骨膜上層”, “method”: “微量多點 / 線性注射”, “effect”: “淚溝填補，維持 9-12 個月”},
“VOLITE_全臉保濕”: {“category”: “玻尿酸”, “brands”: [“喬雅登 VOLITE（芙潤）”], “dose”: {“輕度”: “1-2ml”, “中度”: “2-3ml”, “重度”: “3-4ml”}, “dose_mid”: {“輕度”: “1.5ml”, “中度”: “2.5ml”, “重度”: “3.5ml”}, “layer”: “真皮中層”, “method”: “多點均勻注射 / 水光槍輔助”, “effect”: “膚質細緻，維持 6-9 個月”},
“舒顏萃Sculptra”: {“category”: “膠原蛋白增生劑”, “brands”: [“舒顏萃 Sculptra”], “dose”: {“輕度”: “1瓶”, “中度”: “2瓶”, “重度”: “3-4瓶”}, “dose_mid”: {“輕度”: “1瓶”, “中度”: “2瓶”, “重度”: “3瓶”}, “layer”: “真皮深層至皮下層”, “method”: “扇形大範圍注射，按摩分散”, “effect”: “刺激膠原新生，2-3 月顯現，維持 18-24 個月”},
“鳳凰埋線”: {“category”: “埋線提拉”, “brands”: [“鳳凰埋線（大V線）”], “dose”: {“輕度”: “4-6根/側”, “中度”: “6-10根/側”, “重度”: “10-16根/側”}, “dose_mid”: {“輕度”: “5根/側”, “中度”: “8根/側”, “重度”: “13根/側”}, “layer”: “SMAS 筋膜層 / 深層皮下”, “method”: “逆行進針，錨定點固定，雙向倒鉤提拉”, “effect”: “即時提拉，維持 12-18 個月”},
“麗珠蘭PN1%”: {“category”: “PN 核酸修復”, “brands”: [“麗珠蘭 PN 1%”], “dose”: {“輕度”: “1ml”, “中度”: “1.5-2ml”, “重度”: “2-3ml”}, “dose_mid”: {“輕度”: “1ml”, “中度”: “1.75ml”, “重度”: “2.5ml”}, “layer”: “真皮淺中層”, “method”: “水光槍 / 多點注射”, “effect”: “膚色提亮，建議 3-4 療程”},
“皮秒雷射”: {“category”: “能量療程”, “brands”: [“皮秒雷射”], “dose”: {“輕度”: “1次”, “中度”: “3-5次”, “重度”: “5-8次”}, “dose_mid”: {“輕度”: “1次”, “中度”: “4次”, “重度”: “6次”}, “layer”: “表皮至真皮層”, “method”: “全臉掃描，依斑點加強”, “effect”: “膚色均勻，每 4-6 週一次”},
“音波拉提”: {“category”: “能量療程”, “brands”: [“音波拉提 (Ultherapy)”], “dose”: {“輕度”: “200-400發”, “中度”: “400-600發”, “重度”: “600-1000發”}, “dose_mid”: {“輕度”: “300發”, “中度”: “500發”, “重度”: “800發”}, “layer”: “SMAS 筋膜層 + 真皮層”, “method”: “線性掃描，分層施打”, “effect”: “3-6 月顯效，維持 12-18 個月”},
“電波拉提”: {“category”: “能量療程”, “brands”: [“電波拉提 (Thermage)”], “dose”: {“輕度”: “900發”, “中度”: “1200發”, “重度”: “1500發”}, “dose_mid”: {“輕度”: “900發”, “中度”: “1200發”, “重度”: “1500發”}, “layer”: “真皮深層至皮下層”, “method”: “全臉均勻掃描”, “effect”: “即時緊緻，維持 12-24 個月”},
}

PROBLEM_TO_PRODUCTS = {
“抬頭紋”:  [(“肉毒毒素_抬頭紋”, “首選”), (“皮秒雷射”, “備選”), (“麗珠蘭PN1%”, “備選”)],
“眉間紋”:  [(“肉毒毒素_眉間紋”, “首選”), (“皮秒雷射”, “備選”)],
“魚尾紋”:  [(“肉毒毒素_魚尾紋”, “首選”), (“電波拉提”, “備選”), (“VOLBELLA_淚溝”, “備選”)],
“淚溝”:    [(“VOLBELLA_淚溝”, “首選”), (“麗珠蘭PN1%”, “備選”)],
“法令紋”:  [(“VOLUMA_法令紋”, “首選”), (“VOLIFT_法令紋”, “首選”), (“舒顏萃Sculptra”, “備選”), (“鳳凰埋線”, “備選”)],
“蘋果肌”:  [(“VOLUMA_蘋果肌”, “首選”), (“舒顏萃Sculptra”, “備選”), (“鳳凰埋線”, “備選”)],
“下頜緣”:  [(“VOLUX_下頜輪廓”, “首選”), (“肉毒毒素_下頜緣”, “首選”), (“音波拉提”, “備選”)],
“下巴”:    [(“VOLUMA_下巴”, “首選”)],
“皮膚質地”: [(“VOLITE_全臉保濕”, “首選”), (“皮秒雷射”, “備選”), (“麗珠蘭PN1%”, “備選”)],
“對稱性”:  [(“肉毒毒素_眉間紋”, “首選”), (“VOLUMA_蘋果肌”, “備選”)],
}

PHYSIO_BEAUTY_TIPS = {
“上庭”: {
“好”: “上庭寬闊，早年運佳，智慧過人，父母緣深厚”,
“改善”: “若額頭有皺紋或過窄，可透過肉毒放鬆額肌改善抬頭紋，或以玻尿酸填充額頭弧度，讓上庭更飽滿圓潤，提升整體氣場。”,
“products”: [“肉毒毒素_抬頭紋”, “VOLITE_全臉保濕”]
},
“中庭”: {
“好”: “中庭均衡，中年事業運旺，適合創業或擔任管理職”,
“改善”: “若鼻樑較低或法令紋明顯，可透過玻尿酸墊高鼻樑、填充法令紋，讓中庭比例更完美，面相上增強事業運與威嚴感。”,
“products”: [“VOLUMA_法令紋”, “VOLIFT_法令紋”]
},
“下庭”: {
“好”: “下庭豐厚，晚年運佳，福氣深厚，子孫有緣”,
“改善”: “若下巴短或下頜緣不清晰，可透過玻尿酸墊下巴或埋線提拉，讓下庭比例更協調，面相上加強晚年財運與福氣。”,
“products”: [“VOLUMA_下巴”, “VOLUX_下頜輪廓”]
}
}

SCORE_THRESHOLD = 0.25

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ─────────────────────────────────────────

# 輔助函式

# ─────────────────────────────────────────

def pil_to_cv2(pil_img):
return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img):
return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def image_to_base64(img, fmt=“JPEG”):
buf = io.BytesIO()
img.save(buf, format=fmt)
return base64.b64encode(buf.getvalue()).decode(“utf-8”)

def resize_image(img, max_size=800):
w, h = img.size
if max(w, h) > max_size:
ratio = max_size / max(w, h)
img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
return img

def get_landmarks(img_bgr):
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
results = face_mesh.process(img_rgb)
if not results.multi_face_landmarks:
return None, None
h, w = img_bgr.shape[:2]
lm = results.multi_face_landmarks[0].landmark
landmarks = np.array([[l.x * w, l.y * h, l.z * w] for l in lm])
return landmarks, results

def estimate_yaw(landmarks):
nose = landmarks[1]
left_eye = landmarks[33]
right_eye = landmarks[263]
left_dist = abs(nose[0] - left_eye[0])
right_dist = abs(nose[0] - right_eye[0])
total = left_dist + right_dist
if total < 1e-6:
return 0.0
ratio = left_dist / total
yaw = (ratio - 0.5) * 180.0
return yaw

def classify_severity(value):
if value < 0.3:
return “輕度”
elif value < 0.7:
return “中度”
else:
return “重度”

def safe_depth_std(landmarks, indices):
pts = landmarks[indices, 2]
return float(np.std(pts))

def normalize(value, min_v, max_v):
if max_v - min_v < 1e-6:
return 0.0
return max(0.0, min(1.0, (value - min_v) / (max_v - min_v)))

# ─────────────────────────────────────────

# 臉部分析

# ─────────────────────────────────────────

def analyze_face(landmarks, img_bgr):
h, w = img_bgr.shape[:2]
results = {}

```
hairline_y = landmarks[10, 1]
glabella_y = landmarks[8, 1]
subnasale_y = landmarks[94, 1]
chin_y = landmarks[152, 1]
upper = abs(glabella_y - hairline_y)
middle = abs(subnasale_y - glabella_y)
lower = abs(chin_y - subnasale_y)
total_height = upper + middle + lower + 1e-6
upper_ratio = upper / total_height
middle_ratio = middle / total_height
lower_ratio = lower / total_height

results["三庭比例"] = {
    "upper_ratio": round(upper_ratio, 3),
    "middle_ratio": round(middle_ratio, 3),
    "lower_ratio": round(lower_ratio, 3),
    "dominant": "上庭" if upper_ratio > middle_ratio and upper_ratio > lower_ratio
                else "中庭" if middle_ratio >= upper_ratio and middle_ratio > lower_ratio
                else "下庭",
}

face_width = abs(landmarks[454, 0] - landmarks[234, 0]) + 1e-6
left_eye_w = abs(landmarks[133, 0] - landmarks[33, 0])
right_eye_w = abs(landmarks[263, 0] - landmarks[362, 0])
inter_eye = abs(landmarks[362, 0] - landmarks[133, 0])
eye_avg = (left_eye_w + right_eye_w) / 2.0 + 1e-6
five_eye_ratio = face_width / (5 * eye_avg)
inter_ratio = inter_eye / eye_avg

results["五眼比例"] = {
    "face_width": round(face_width, 1),
    "eye_avg_width": round(eye_avg, 1),
    "inter_eye_distance": round(inter_eye, 1),
    "five_eye_ratio": round(five_eye_ratio, 3),
    "inter_ratio": round(inter_ratio, 3),
}

nose_center_x = landmarks[1, 0]
sym_pairs = [(33, 263), (133, 362), (234, 454), (61, 291), (50, 280), (149, 378)]
asym_sum = 0.0
for li, ri in sym_pairs:
    l_dist = abs(landmarks[li, 0] - nose_center_x)
    r_dist = abs(landmarks[ri, 0] - nose_center_x)
    pair_max = max(l_dist, r_dist, 1e-6)
    asym_sum += abs(l_dist - r_dist) / pair_max
asym_score = asym_sum / len(sym_pairs)
asym_norm = normalize(asym_score, 0.0, 0.3)
results["對稱性"] = {"score": round(asym_norm, 3), "severity": classify_severity(asym_norm), "description": "臉部左右對稱性"}

tear_z_avg = np.mean(landmarks[[159, 145], 2])
cheek_z_avg = np.mean(landmarks[[50, 280], 2])
tear_depth = abs(tear_z_avg - cheek_z_avg)
tear_norm = normalize(tear_depth, 0.001, 0.015)
results["淚溝"] = {"score": round(tear_norm, 3), "severity": classify_severity(tear_norm), "description": "淚溝凹陷程度"}

nasal_z = np.mean(landmarks[[49, 279], 2])
mouth_corner_z = np.mean(landmarks[[61, 291], 2])
nasolabial_depth = abs(nasal_z - mouth_corner_z)
left_len = np.linalg.norm(landmarks[49, :2] - landmarks[61, :2])
right_len = np.linalg.norm(landmarks[279, :2] - landmarks[291, :2])
nasolabial_len_avg = (left_len + right_len) / 2.0
depth_norm = normalize(nasolabial_depth, 0.001, 0.015)
len_norm = normalize(nasolabial_len_avg, face_width * 0.1, face_width * 0.25)
nasolabial_score = 0.6 * depth_norm + 0.4 * len_norm
results["法令紋"] = {"score": round(nasolabial_score, 3), "severity": classify_severity(nasolabial_score), "description": "法令紋深度與長度"}

cheekbone_z = np.mean(landmarks[[117, 346], 2])
cheek_ref_z = np.mean(landmarks[[50, 280], 2])
apple_muscle = abs(cheekbone_z - cheek_ref_z)
apple_norm = normalize(apple_muscle, 0.001, 0.012)
apple_score = 1.0 - apple_norm
results["蘋果肌"] = {"score": round(apple_score, 3), "severity": classify_severity(apple_score), "description": "蘋果肌飽滿度"}

jaw_indices = [152, 172, 171, 170, 169, 136, 135, 134, 58, 172]
jaw_pts = landmarks[jaw_indices, :2]
jaw_curvature_std = np.std(jaw_pts[:, 1])
jaw_norm = normalize(jaw_curvature_std, 2.0, 25.0)
results["下頜緣"] = {"score": round(jaw_norm, 3), "severity": classify_severity(jaw_norm), "description": "下頜緣輪廓清晰度"}

forehead_indices = [55, 107, 66, 105, 65, 52, 53, 46, 124, 156, 70, 63]
forehead_std = safe_depth_std(landmarks, forehead_indices)
forehead_norm = normalize(forehead_std, 0.001, 0.008)
results["抬頭紋"] = {"score": round(forehead_norm, 3), "severity": classify_severity(forehead_norm), "description": "額頭橫紋深度"}

glabella_indices = [168, 6, 197, 195, 5, 4, 8, 9]
glabella_std = safe_depth_std(landmarks, glabella_indices)
glabella_norm = normalize(glabella_std, 0.001, 0.008)
results["眉間紋"] = {"score": round(glabella_norm, 3), "severity": classify_severity(glabella_norm), "description": "眉間川字紋深度"}

crow_feet_indices = [33, 133, 246, 161, 160, 159, 263, 362, 466, 388, 387, 386]
crow_std = safe_depth_std(landmarks, crow_feet_indices)
crow_norm = normalize(crow_std, 0.001, 0.01)
results["魚尾紋"] = {"score": round(crow_norm, 3), "severity": classify_severity(crow_norm), "description": "眼外角魚尾紋深度"}

chin_tip = landmarks[152]
chin_left = landmarks[172]
chin_right = landmarks[397]
chin_width = abs(chin_left[0] - chin_right[0])
chin_length = abs(chin_tip[1] - subnasale_y)
chin_ratio = chin_length / (chin_width + 1e-6)
chin_score = normalize(chin_ratio, 0.3, 1.0)
results["下巴"] = {"score": round(chin_score, 3), "severity": classify_severity(chin_score), "chin_ratio": round(chin_ratio, 3), "description": "下巴長寬比"}

skin_indices = [50, 280, 205, 425, 117, 346, 187, 411]
skin_z = landmarks[skin_indices, 2]
skin_cv = np.std(skin_z) / (np.mean(np.abs(skin_z)) + 1e-6)
skin_norm = normalize(skin_cv, 0.05, 0.6)
results["皮膚質地"] = {"score": round(skin_norm, 3), "severity": classify_severity(skin_norm), "description": "皮膚粗糙度"}

return results
```

# ─────────────────────────────────────────

# 小腿分析

# ─────────────────────────────────────────

def analyze_calf(img_bgr):
result = {“detected”: False, “calf_ratio”: 0, “severity”: “輕度”, “score”: 0, “description”: “”}
try:
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
results = pose.process(img_rgb)
if not results.pose_landmarks:
return result
h, w = img_bgr.shape[:2]
lm = results.pose_landmarks.landmark
left_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE]
left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
right_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
right_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]

```
    left_calf_len = abs(left_knee.y - left_ankle.y) * h
    right_calf_len = abs(right_knee.y - right_ankle.y) * h

    if left_calf_len < 10 or right_calf_len < 10:
        return result

    calf_width = abs(left_knee.x - right_knee.x) * w
    body_width = w * 0.6
    calf_ratio = calf_width / body_width if body_width > 0 else 0

    score = normalize(float(calf_ratio), 0.2, 0.6)
    severity = classify_severity(score)

    result = {
        "detected": True,
        "calf_ratio": round(float(calf_ratio), 3),
        "score": round(score, 3),
        "severity": severity,
        "description": f"小腿寬度比例：{calf_ratio:.2f}（比例越高代表肌肉越發達）",
        "left_knee": (int(left_knee.x * w), int(left_knee.y * h)),
        "left_ankle": (int(left_ankle.x * w), int(left_ankle.y * h)),
        "right_knee": (int(right_knee.x * w), int(right_knee.y * h)),
        "right_ankle": (int(right_ankle.x * w), int(right_ankle.y * h)),
    }
except Exception as e:
    result["description"] = f"偵測失敗：{e}"
return result
```

def draw_calf_annotations(img_bgr, calf_result):
img = img_bgr.copy()
if not calf_result.get(“detected”):
return img
COLOR = (0, 220, 255)
for key in [“left_knee”, “left_ankle”, “right_knee”, “right_ankle”]:
if key in calf_result:
cv2.circle(img, calf_result[key], 8, COLOR, -1)
if “left_knee” in calf_result and “left_ankle” in calf_result:
cv2.line(img, calf_result[“left_knee”], calf_result[“left_ankle”], COLOR, 2)
if “right_knee” in calf_result and “right_ankle” in calf_result:
cv2.line(img, calf_result[“right_knee”], calf_result[“right_ankle”], COLOR, 2)
sev = calf_result.get(“severity”, “”)
score = calf_result.get(“score”, 0)
cv2.putText(img, f”Calf: {sev} ({score:.2f})”, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
return img

# ─────────────────────────────────────────

# 背部分析

# ─────────────────────────────────────────

def analyze_back(img_front_bgr, img_side_bgr):
result = {“detected”: False, “back_width_ratio”: 0, “back_thickness_ratio”: 0, “severity”: “輕度”, “description”: “”}
try:
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
img_rgb = cv2.cvtColor(img_front_bgr, cv2.COLOR_BGR2RGB)
res_front = pose.process(img_rgb)

```
    if res_front.pose_landmarks:
        h, w = img_front_bgr.shape[:2]
        lm = res_front.pose_landmarks.landmark
        left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

        shoulder_width = abs(left_shoulder.x - right_shoulder.x) * w
        hip_width = abs(left_hip.x - right_hip.x) * w
        back_width_ratio = shoulder_width / (w * 0.8 + 1e-6)

        result["detected"] = True
        result["back_width_ratio"] = round(float(back_width_ratio), 3)
        result["shoulder_width_px"] = round(float(shoulder_width), 1)
        result["hip_width_px"] = round(float(hip_width), 1)
        result["shoulder_hip_ratio"] = round(float(shoulder_width / (hip_width + 1e-6)), 3)

    if img_side_bgr is not None:
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            img_rgb = cv2.cvtColor(img_side_bgr, cv2.COLOR_BGR2RGB)
            res_side = pose.process(img_rgb)
        if res_side.pose_landmarks:
            h, w = img_side_bgr.shape[:2]
            lm = res_side.pose_landmarks.landmark
            nose = lm[mp_pose.PoseLandmark.NOSE]
            left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            back_thickness = abs(nose.x - left_shoulder.x) * w
            result["back_thickness_ratio"] = round(float(back_thickness / (w + 1e-6)), 3)

    bwr = result.get("back_width_ratio", 0)
    score = normalize(bwr, 0.3, 0.7)
    result["score"] = round(score, 3)
    result["severity"] = classify_severity(score)
    result["description"] = f"背部寬度比例：{bwr:.2f}，肩臀比：{result.get('shoulder_hip_ratio', 0):.2f}"

except Exception as e:
    result["description"] = f"偵測失敗：{e}"
return result
```

# ─────────────────────────────────────────

# 治療建議生成

# ─────────────────────────────────────────

def generate_recommendations(analysis):
rec_list = []
for problem, data in analysis.items():
if problem in (“三庭比例”, “五眼比例”):
continue
if not isinstance(data, dict) or “score” not in data:
continue
score = data.get(“score”, 0)
severity = data.get(“severity”, “輕度”)
if score < SCORE_THRESHOLD:
continue
if problem not in PROBLEM_TO_PRODUCTS:
continue

```
    primary = []
    alternatives = []
    for pk, priority in PROBLEM_TO_PRODUCTS[problem]:
        if pk not in PRODUCT_DB:
            continue
        prod = PRODUCT_DB[pk]
        info = {
            "product_key": pk,
            "product_name": prod["brands"][0],
            "category": prod.get("category", ""),
            "dose_mid": prod["dose_mid"].get(severity, "依醫師評估") if isinstance(prod["dose_mid"], dict) else prod["dose_mid"],
            "layer": prod.get("layer", ""),
            "method": prod.get("method", ""),
            "effect": prod.get("effect", ""),
        }
        if priority == "首選":
            primary.append(info)
        else:
            alternatives.append(info)

    if primary or alternatives:
        rec_list.append({
            "problem": problem,
            "description": data.get("description", ""),
            "score": score,
            "severity": severity,
            "primary": primary,
            "alternatives": alternatives,
        })
return rec_list
```

# ─────────────────────────────────────────

# 面相學分析

# ─────────────────────────────────────────

def physiognomy_reading(analysis):
readings = []
three_zones = analysis.get(“三庭比例”, {})
five_eyes = analysis.get(“五眼比例”, {})

```
dominant = three_zones.get("dominant", "")
upper_r = three_zones.get("upper_ratio", 0.333)
middle_r = three_zones.get("middle_ratio", 0.333)
lower_r = three_zones.get("lower_ratio", 0.333)

for zone, ratio in [("上庭", upper_r), ("中庭", middle_r), ("下庭", lower_r)]:
    tip = PHYSIO_BEAUTY_TIPS[zone]
    is_dominant = (dominant == zone)
    readings.append({
        "aspect": f"{zone}分析",
        "zone": zone,
        "ratio": f"{ratio:.1%}",
        "icon": "⭐" if is_dominant else "📐",
        "reading": tip["好"] if is_dominant else f"{zone}比例偏{'高' if ratio > 0.4 else '低'}，建議調整比例",
        "beauty_tip": tip["改善"],
        "beauty_products": tip["products"],
    })

inter_r = five_eyes.get("inter_ratio", 1.0)
if inter_r < 0.85:
    readings.append({
        "aspect": "眼距偏窄", "zone": "", "ratio": f"{inter_r:.2f}",
        "icon": "⚡", "reading": "眼距較窄，個性敏銳反應快，容易急躁",
        "beauty_tip": "可透過眉形調整或淚溝填充，視覺上拉寬眼距，讓面部比例更協調。",
        "beauty_products": ["VOLBELLA_淚溝"],
    })
elif inter_r > 1.2:
    readings.append({
        "aspect": "眼距寬廣", "zone": "", "ratio": f"{inter_r:.2f}",
        "icon": "🌊", "reading": "眼距寬廣，心胸寬大，人緣佳，適合公關外交",
        "beauty_tip": "眼距已相當理想，可搭配眼周保養或皮秒改善眼周膚質。",
        "beauty_products": ["皮秒雷射"],
    })

if analysis.get("法令紋", {}).get("score", 0) > 0.5:
    readings.append({
        "aspect": "法令紋顯現", "zone": "", "ratio": "",
        "icon": "👑", "reading": "法令紋深象徵威嚴與領導力，主掌大局之相",
        "beauty_tip": "法令紋象徵威嚴，但視覺顯老。可透過玻尿酸填充或舒顏萃刺激膠原新生，保持威嚴命格同時外觀更年輕。",
        "beauty_products": ["VOLUMA_法令紋", "舒顏萃Sculptra"],
    })

return readings
```

# ─────────────────────────────────────────

# 繪製標註圖

# ─────────────────────────────────────────

def draw_annotations(img_bgr, landmarks, analysis):
img = img_bgr.copy()
h, w = img.shape[:2]

```
def pt(idx):
    return (int(landmarks[idx, 0]), int(landmarks[idx, 1]))

COLOR_LINE = (0, 220, 255)
COLOR_PT = (0, 100, 255)
COLOR_TEXT = (255, 255, 255)

lines_y = {"髮際": int(landmarks[10, 1]), "眉間": int(landmarks[8, 1]), "鼻下": int(landmarks[94, 1]), "下巴": int(landmarks[152, 1])}
for label, y in lines_y.items():
    cv2.line(img, (0, y), (w, y), COLOR_LINE, 1)
    cv2.putText(img, label, (5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

for x in [landmarks[234, 0], landmarks[33, 0], landmarks[133, 0], landmarks[362, 0], landmarks[263, 0], landmarks[454, 0]]:
    cv2.line(img, (int(x), 0), (int(x), h), (200, 200, 0), 1)

for idx in [10, 8, 94, 152, 33, 133, 362, 263, 234, 454, 159, 145, 49, 279, 61, 291, 117, 346, 1]:
    cv2.circle(img, pt(idx), 3, COLOR_PT, -1)

zones = analysis.get("三庭比例", {})
for y1, y2, text in [
    (lines_y["髮際"], lines_y["眉間"], f'上庭 {zones.get("upper_ratio", 0):.1%}'),
    (lines_y["眉間"], lines_y["鼻下"], f'中庭 {zones.get("middle_ratio", 0):.1%}'),
    (lines_y["鼻下"], lines_y["下巴"], f'下庭 {zones.get("lower_ratio", 0):.1%}'),
]:
    cv2.putText(img, text, (w - 130, (y1 + y2) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_LINE, 1, cv2.LINE_AA)

return img
```

# ─────────────────────────────────────────

# CSS 樣式

# ─────────────────────────────────────────

def apply_styles():
st.markdown(”””
<style>
.stApp { background: #0d1117; }
h1, h2, h3, h4 { color: #ffd700 !important; }
p, li, span, div { color: #e8e8e8; }
.stMarkdown p { color: #e8e8e8 !important; }
.stCaption { color: #a0b4c8 !important; }
label, .stSelectbox label, .stRadio label { color: #e8e8e8 !important; }
.stRadio > div > div > label { color: #e8e8e8 !important; font-weight: 600; }
.upload-label { font-size: 0.95rem; font-weight: 700; color: #ffd700; margin-bottom: 6px; display: block; }
.angle-ok   { color: #2ecc71; font-size: 0.88rem; font-weight: 600; }
.angle-fail { color: #e74c3c; font-size: 0.88rem; font-weight: 600; }
.badge-輕度 { background: #1a3a2a; color: #2ecc71; border: 1px solid #2ecc71; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
.badge-中度 { background: #3a2a0a; color: #ffcc00; border: 1px solid #ffcc00; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
.badge-重度 { background: #3a1a1a; color: #ff6060; border: 1px solid #ff6060; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
.primary-tag { background: #1a3a2a; color: #2ecc71; border: 1px solid #2ecc71; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; font-weight: 700; }
.alt-tag { background: #1a1a3a; color: #8888ff; border: 1px solid #8888ff; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; }
.stButton > button { background: linear-gradient(135deg, #e0c44a, #c0a030) !important; color: #000 !important; font-weight: 700 !important; border-radius: 8px !important; border: none !important; padding: 12px 30px !important; font-size: 1rem !important; }
.stButton > button:hover { opacity: 0.85 !important; }
div[data-testid=“stInfo”] { background: #0d2137 !important; color: #c0d8f0 !important; }
div[data-testid=“stInfo”] p { color: #c0d8f0 !important; }
div[data-testid=“stSuccess”] p { color: #c0f0c0 !important; }
div[data-testid=“stWarning”] p { color: #f0d080 !important; }
div[data-testid=“metric-container”] label { color: #a0b4c8 !important; }
div[data-testid=“metric-container”] [data-testid=“metric-value”] { color: #ffd700 !important; font-size: 1.4rem !important; }
div[data-testid=“metric-container”] [data-testid=“metric-delta”] { color: #90c090 !important; }
.stExpander details summary p { color: #ffd700 !important; font-weight: 600; }
.stProgress > div > div { background: linear-gradient(90deg, #2ecc71, #e0c44a, #e74c3c) !important; }
</style>
“””, unsafe_allow_html=True)

# ─────────────────────────────────────────

# 主程式

# ─────────────────────────────────────────

def main():
apply_styles()

```
st.title("🏥 AI 智能醫美面診輔助系統")
st.caption("上傳照片，AI 自動分析並生成個人化治療建議")

st.markdown("---")
mode = st.radio("**📋 選擇分析模式**", ["👤 臉部分析", "🦵 小腿肌肉分析", "🔙 背部分析"], horizontal=True)

# ═══════════════════════════════════════
# 模式一：臉部分析
# ═══════════════════════════════════════
if mode == "👤 臉部分析":
    st.markdown("---")
    st.subheader("📸 上傳臉部照片")
    st.info("📌 請上傳：正面、左側 45°、左側 90°、右側 45°、右側 90°（至少需要正面照片）")

    ANGLE_CONFIG = {
        "front":   {"expected": 0,   "tolerance": 15, "label": "① 正面（0°）"},
        "left45":  {"expected": -35, "tolerance": 18, "label": "② 左側 45°"},
        "left90":  {"expected": -70, "tolerance": 20, "label": "③ 左側 90°"},
        "right45": {"expected": 35,  "tolerance": 18, "label": "④ 右側 45°"},
        "right90": {"expected": 70,  "tolerance": 20, "label": "⑤ 右側 90°"},
    }

    col1, col2, col3, col4, col5 = st.columns(5)
    cols_map = {"front": col1, "left45": col2, "left90": col3, "right45": col4, "right90": col5}
    uploads = {}

    for key, col in cols_map.items():
        with col:
            cfg = ANGLE_CONFIG[key]
            st.markdown(f'<span class="upload-label">{cfg["label"]}</span>', unsafe_allow_html=True)
            f = st.file_uploader("", type=["jpg", "jpeg", "png"], key=key, label_visibility="collapsed")
            if f:
                uploads[key] = Image.open(f).convert("RGB")
                st.image(uploads[key], use_container_width=True)

    # 角度驗證
    angle_ok = {}
    if uploads:
        st.markdown("---")
        st.subheader("🔍 角度驗證")
        v_cols = st.columns(len(uploads))
        for i, (key, pil_img) in enumerate(uploads.items()):
            cfg = ANGLE_CONFIG[key]
            img_bgr = pil_to_cv2(resize_image(pil_img, 640))
            lm, _ = get_landmarks(img_bgr)
            with v_cols[i]:
                if lm is None:
                    st.markdown(f'<div class="angle-fail">❌ {cfg["label"]}<br>無法偵測人臉</div>', unsafe_allow_html=True)
                    angle_ok[key] = False
                else:
                    yaw = estimate_yaw(lm)
                    diff = abs(yaw - cfg["expected"])
                    ok = diff <= cfg["tolerance"]
                    angle_ok[key] = ok
                    icon = "✅" if ok else "⚠️"
                    cls = "angle-ok" if ok else "angle-fail"
                    st.markdown(f'<div class="{cls}">{icon} {cfg["label"]}<br>Yaw ≈ {yaw:.1f}°</div>', unsafe_allow_html=True)

    st.markdown("---")

    if "front" in uploads:
        if st.button("🚀 開始 AI 面診分析", use_container_width=True):
            with st.spinner("AI 分析中，請稍候..."):
                try:
                    front_bgr = pil_to_cv2(resize_image(uploads["front"], 800))
                    lm_front, _ = get_landmarks(front_bgr)

                    if lm_front is None:
                        st.error("❌ 無法偵測正面人臉，請上傳更清晰的照片。")
                        return

                    analysis = analyze_face(lm_front, front_bgr)
                    img_annotated = draw_annotations(front_bgr, lm_front, analysis)
                    recommendations = generate_recommendations(analysis)
                    physio = physiognomy_reading(analysis)

                    st.success("✅ 分析完成！")

                    # 標註圖
                    st.subheader("🎯 三庭五眼標註圖")
                    st.image(cv2_to_pil(img_annotated), use_container_width=True)

                    # 三庭比例
                    st.subheader("📐 三庭比例分析")
                    zones = analysis["三庭比例"]
                    zc1, zc2, zc3 = st.columns(3)
                    zc1.metric("上庭", f'{zones["upper_ratio"]:.1%}', delta=f'{(zones["upper_ratio"]-0.333)*100:+.1f}%（理想33.3%）')
                    zc2.metric("中庭", f'{zones["middle_ratio"]:.1%}', delta=f'{(zones["middle_ratio"]-0.333)*100:+.1f}%（理想33.3%）')
                    zc3.metric("下庭", f'{zones["lower_ratio"]:.1%}', delta=f'{(zones["lower_ratio"]-0.333)*100:+.1f}%（理想33.3%）')
                    st.info(f"**主導分區：{zones['dominant']}**（各區理想比例均為 33.3%）")

                    # 五眼比例
                    st.subheader("👁️ 五眼比例分析")
                    fe = analysis["五眼比例"]
                    fc1, fc2, fc3 = st.columns(3)
                    fc1.metric("五眼比例", f'{fe["five_eye_ratio"]:.2f}', delta=f'{fe["five_eye_ratio"]-1.0:+.2f}（理想=1.0）')
                    fc2.metric("眼距比", f'{fe["inter_ratio"]:.2f}', delta=f'{fe["inter_ratio"]-1.0:+.2f}（理想=1.0）')
                    fc3.metric("平均眼寬", f'{fe["eye_avg_width"]:.0f}px')

                    # 問題診斷
                    st.subheader("🔍 問題診斷")
                    scored = sorted([(k, v) for k, v in analysis.items() if isinstance(v, dict) and "score" in v], key=lambda x: x[1]["score"], reverse=True)
                    for name, data in scored:
                        sev = data["severity"]
                        score = data["score"]
                        desc = data.get("description", "")
                        color = "#2ecc71" if sev == "輕度" else "#ffcc00" if sev == "中度" else "#ff6060"
                        st.markdown(
                            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>"
                            f"<span style='color:#ffffff;font-weight:700;min-width:80px;font-size:0.95rem;'>{name}</span>"
                            f"<span class='badge-{sev}'>{sev}</span>"
                            f"<span style='color:#a0b4c8;font-size:0.82rem;flex:1;'>{desc}</span>"
                            f"<span style='color:{color};font-weight:600;font-size:0.88rem;'>{score:.2f}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        st.progress(score)

                    # 治療建議
                    st.subheader("💉 個人化治療建議")
                    st.markdown(
                        "<div style='color:#c0d8e8;font-size:0.88rem;margin-bottom:12px;'>"
                        "✅ <strong style='color:#2ecc71;'>首選</strong> = 最推薦療程 ｜ "
                        "🔄 <strong style='color:#8888ff;'>備選</strong> = 輔助或替代療程"
                        "</div>",
                        unsafe_allow_html=True
                    )

                    if recommendations:
                        for rec in recommendations:
                            label = f"🔸 {rec['problem']} — {rec['severity']}（{rec['score']:.2f}）"
                            with st.expander(label, expanded=rec["score"] > 0.5):
                                st.markdown(f"<p style='color:#a0c0d8;font-size:0.85rem;margin-bottom:10px;'>{rec['description']}</p>", unsafe_allow_html=True)

                                if rec["primary"]:
                                    st.markdown("<div style='color:#2ecc71;font-weight:700;margin-bottom:6px;'>✅ 首選治療</div>", unsafe_allow_html=True)
                                    for prod in rec["primary"]:
                                        st.markdown(
                                            f"<div style='background:#0a1f10;border-radius:8px;padding:14px;margin:6px 0;border-left:4px solid #2ecc71;'>"
                                            f"<div style='color:#ffd700;font-weight:700;font-size:1rem;margin-bottom:6px;'>"
                                            f"{prod['product_name']} <span class='primary-tag'>首選</span></div>"
                                            f"<div style='color:#c0e8c0;font-size:0.88rem;line-height:1.8;'>"
                                            f"💊 <b>類別：</b>{prod['category']}<br>"
                                            f"💉 <b>建議劑量：</b>{prod['dose_mid']}<br>"
                                            f"📍 <b>注射層次：</b>{prod['layer']}<br>"
                                            f"🎯 <b>注射方式：</b>{prod['method']}<br>"
                                            f"✨ <b>預估效果：</b>{prod['effect']}"
                                            f"</div></div>",
                                            unsafe_allow_html=True
                                        )

                                if rec["alternatives"]:
                                    st.markdown("<div style='color:#8888ff;font-weight:700;margin:10px 0 6px;'>🔄 備選治療</div>", unsafe_allow_html=True)
                                    alt_cols = st.columns(min(len(rec["alternatives"]), 3))
                                    for j, prod in enumerate(rec["alternatives"]):
                                        with alt_cols[j % len(alt_cols)]:
                                            st.markdown(
                                                f"<div style='background:#0a0a20;border-radius:8px;padding:10px;border-left:3px solid #8888ff;height:100%;'>"
                                                f"<div style='color:#c0c8ff;font-weight:600;margin-bottom:4px;'>{prod['product_name']} <span class='alt-tag'>備選</span></div>"
                                                f"<div style='color:#8090c0;font-size:0.82rem;line-height:1.7;'>"
                                                f"{prod['category']}<br>劑量：{prod['dose_mid']}<br>{prod['effect']}"
                                                f"</div></div>",
                                                unsafe_allow_html=True
                                            )
                    else:
                        st.success("✅ 未偵測到明顯問題，繼續維持良好保養即可！")

                    # 面相學
                    st.subheader("🔮 面相學分析與醫美改善建議")
                    st.markdown("<p style='color:#a0b4c8;font-size:0.85rem;'>以下結合面相解讀與對應醫美調整方向，供娛樂與參考。</p>", unsafe_allow_html=True)

                    for reading in physio:
                        zone_label = f"｜{reading['zone']} {reading['ratio']}" if reading.get("zone") else ""
                        prod_names = ", ".join([PRODUCT_DB[p]["brands"][0] for p in reading.get("beauty_products", []) if p in PRODUCT_DB])
                        st.markdown(
                            f"<div style='background:#1a1a2e;border-radius:10px;padding:16px;margin-bottom:14px;border-left:4px solid #e0c44a;'>"
                            f"<div style='margin-bottom:8px;'>"
                            f"<span style='font-size:1.4rem;'>{reading['icon']}</span>"
                            f"<span style='color:#ffd700;font-weight:700;font-size:1rem;margin-left:8px;'>{reading['aspect']}</span>"
                            f"<span style='color:#7090a0;font-size:0.82rem;margin-left:8px;'>{zone_label}</span>"
                            f"</div>"
                            f"<div style='background:#0f1520;border-radius:6px;padding:10px;margin-bottom:8px;'>"
                            f"<span style='color:#ffd700;font-size:0.85rem;font-weight:600;'>📖 面相解讀：</span>"
                            f"<span style='color:#c8d8e8;font-size:0.88rem;'>{reading['reading']}</span>"
                            f"</div>"
                            f"<div style='background:#0f2015;border-radius:6px;padding:10px;'>"
                            f"<span style='color:#2ecc71;font-size:0.85rem;font-weight:600;'>💉 醫美改善建議：</span>"
                            f"<span style='color:#a0d0a0;font-size:0.88rem;'>{reading['beauty_tip']}</span><br>"
                            f"<span style='color:#7090a0;font-size:0.82rem;'>🏷 建議產品：{prod_names}</span>"
                            f"</div></div>",
                            unsafe_allow_html=True
                        )

                    # 下載報告
                    st.subheader("📄 下載分析報告")
                    report_lines = [
                        "=" * 50,
                        "  AI 智能醫美面診報告",
                        f"  日期：{datetime.now().strftime('%Y年%m月%d日 %H:%M')}",
                        "=" * 50,
                        "",
                        "【三庭比例】",
                        f"  上庭：{zones['upper_ratio']:.1%}",
                        f"  中庭：{zones['middle_ratio']:.1%}",
                        f"  下庭：{zones['lower_ratio']:.1%}",
                        f"  主導：{zones['dominant']}",
                        "",
                        "【問題診斷】",
                    ]
                    for name, data in scored:
                        if isinstance(data, dict) and "score" in data:
                            report_lines.append(f"  {name}：{data['severity']}（{data['score']:.2f}）")

                    report_lines += ["", "【治療建議】"]
                    for rec in recommendations:
                        report_lines.append(f"\n  ▸ {rec['problem']}（{rec['severity']}）")
                        for p in rec["primary"]:
                            report_lines.append(f"    ✅ 首選：{p['product_name']} / {p['dose_mid']}")
                            report_lines.append(f"       效果：{p['effect']}")
                        for p in rec["alternatives"]:
                            report_lines.append(f"    🔄 備選：{p['product_name']}")

                    report_lines += ["", "【免責聲明】", "  本報告為 AI 輔助參考，不構成醫療建議，請諮詢合法執照醫師。"]

                    st.download_button(
                        "⬇️ 下載完整分析報告（TXT）",
                        data="\n".join(report_lines).encode("utf-8"),
                        file_name=f"face_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                    st.markdown("---")
                    st.warning("⚠️ 本系統為輔助參考工具，分析結果不構成醫療建議，實際治療請諮詢合法執照醫師。")

                except Exception as e:
                    st.error(f"❌ 分析錯誤：{e}")
                    st.exception(e)
    else:
        st.info("👆 請先上傳正面照片後開始分析")

# ═══════════════════════════════════════
# 模式二：小腿肌肉分析
# ═══════════════════════════════════════
elif mode == "🦵 小腿肌肉分析":
    st.markdown("---")
    st.subheader("🦵 小腿肌肉肥大分析")
    st.info("📌 拍攝要求：全身正面站立，確保小腿（膝蓋至腳踝）完整入鏡。光線充足效果最佳。")

    col1, col2 = st.columns(2)
    calf_normal = None
    calf_tiptoe = None

    with col1:
        st.markdown('<span class="upload-label">① 自然站姿（正面）</span>', unsafe_allow_html=True)
        f1 = st.file_uploader("", type=["jpg", "jpeg", "png"], key="calf_normal", label_visibility="collapsed")
        if f1:
            calf_normal = Image.open(f1).convert("RGB")
            st.image(calf_normal, use_container_width=True, caption="自然站姿")

    with col2:
        st.markdown('<span class="upload-label">② 墊腳尖（正面）—— 選填</span>', unsafe_allow_html=True)
        f2 = st.file_uploader("", type=["jpg", "jpeg", "png"], key="calf_tiptoe", label_visibility="collapsed")
        if f2:
            calf_tiptoe = Image.open(f2).convert("RGB")
            st.image(calf_tiptoe, use_container_width=True, caption="墊腳尖")

    if calf_normal:
        if st.button("🚀 開始小腿分析", use_container_width=True):
            with st.spinner("分析中..."):
                try:
                    img_bgr = pil_to_cv2(resize_image(calf_normal, 800))
                    result = analyze_calf(img_bgr)
                    img_ann = draw_calf_annotations(img_bgr, result)

                    st.success("✅ 分析完成！")
                    st.image(cv2_to_pil(img_ann), use_container_width=True, caption="小腿關鍵點標註")

                    if result["detected"]:
                        sev = result["severity"]
                        score = result["score"]

                        st.markdown(
                            f"<div style='background:#1a1a2e;border-radius:10px;padding:16px;border-left:4px solid #e0c44a;margin:12px 0;'>"
                            f"<div style='color:#ffd700;font-size:1.1rem;font-weight:700;margin-bottom:8px;'>"
                            f"小腿肌肉肥大程度：<span class='badge-{sev}'>{sev}</span></div>"
                            f"<div style='color:#c0d0e0;'>{result['description']}</div>"
                            f"<div style='color:#90a0b0;font-size:0.85rem;margin-top:4px;'>AI 評分：{score:.2f} / 1.00</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        st.progress(score)

                        # 肉毒建議
                        prod = PRODUCT_DB["肉毒毒素_小腿"]
                        dose = prod["dose_mid"].get(sev, "依醫師評估")
                        st.markdown("### 💉 小腿肉毒毒素治療建議")
                        st.markdown(
                            f"<div style='background:#0a1f10;border-radius:10px;padding:16px;border-left:4px solid #2ecc71;'>"
                            f"<div style='color:#ffd700;font-size:1.05rem;font-weight:700;margin-bottom:10px;'>✅ 腓腸肌肉毒毒素注射</div>"
                            f"<div style='color:#c0e8c0;font-size:0.9rem;line-height:2;'>"
                            f"💊 <b>建議品牌：</b>{', '.join(prod['brands'])}<br>"
                            f"💉 <b>建議劑量（{sev}）：</b><span style='color:#ffd700;font-weight:700;'>{dose}U</span><br>"
                            f"📍 <b>注射層次：</b>{prod['layer']}<br>"
                            f"🎯 <b>注射方式：</b>{prod['method']}<br>"
                            f"✨ <b>預估效果：</b>{prod['effect']}"
                            f"</div></div>",
                            unsafe_allow_html=True
                        )

                        st.markdown("### ⚠️ 治療注意事項")
                        for tip in [
                            "治療前 2 週停止高強度腿部訓練，避免肌肉過度充血影響藥效",
                            "注射後 4-6 小時內請勿按摩注射部位",
                            "注射後 24 小時內避免劇烈運動、三溫暖及飲酒",
                            "建議每 6-9 個月維持一次療程以保持效果",
                            "若有長期穿高跟鞋習慣，建議同時調整姿勢以配合治療",
                            "可搭配小腿拉伸伸展運動，加速肌肉放鬆效果更佳",
                        ]:
                            st.markdown(f"<div style='color:#c0d0e0;padding:3px 0;'>• {tip}</div>", unsafe_allow_html=True)

                        # 墊腳尖對比
                        if calf_tiptoe:
                            st.markdown("### 📊 墊腳尖對比分析")
                            img_tip_bgr = pil_to_cv2(resize_image(calf_tiptoe, 800))
                            tip_result = analyze_calf(img_tip_bgr)
                            img_tip_ann = draw_calf_annotations(img_tip_bgr, tip_result)

                            ca, cb = st.columns(2)
                            with ca:
                                st.image(cv2_to_pil(img_ann), caption="自然站姿", use_container_width=True)
                                if result["detected"]:
                                    st.metric("肌肉評分", f"{result['score']:.2f}")
                            with cb:
                                st.image(cv2_to_pil(img_tip_ann), caption="墊腳尖", use_container_width=True)
                                if tip_result["detected"]:
                                    delta = tip_result["score"] - result["score"]
                                    st.metric("肌肉評分", f"{tip_result['score']:.2f}", delta=f"{delta:+.2f}")

                            if result["detected"] and tip_result["detected"]:
                                diff = tip_result["score"] - result["score"]
                                if diff > 0.1:
                                    st.info("💡 墊腳尖時肌肉明顯收縮，腓腸肌活躍度高，建議治療劑量可略微增加。")
                                else:
                                    st.info("💡 兩姿勢評分差異不大，肌肉肥大主要為靜態增大型，標準劑量即可。")
                    else:
                        st.warning("⚠️ 無法偵測小腿關鍵點，請確認照片包含膝蓋至腳踝完整範圍，並保持光線充足。")

                except Exception as e:
                    st.error(f"❌ 分析錯誤：{e}")
                    st.exception(e)
    else:
        st.info("👆 請上傳自然站姿照片以開始分析")

# ═══════════════════════════════════════
# 模式三：背部分析
# ═══════════════════════════════════════
elif mode == "🔙 背部分析":
    st.markdown("---")
    st.subheader("🔙 背部肌肉與輪廓分析")
    st.info("📌 請上傳：**正背面**（背對鏡頭站立，完整上半身入鏡）及 **側背面**（側面站立，選填）")

    col1, col2 = st.columns(2)
    back_front = None
    back_side = None

    with col1:
        st.markdown('<span class="upload-label">① 正背面（背對鏡頭）</span>', unsafe_allow_html=True)
        bf = st.file_uploader("", type=["jpg", "jpeg", "png"], key="back_front", label_visibility="collapsed")
        if bf:
            back_front = Image.open(bf).convert("RGB")
            st.image(back_front, use_container_width=True, caption="正背面")

    with col2:
        st.markdown('<span class="upload-label">② 側背面（側面站立）—— 選填</span>', unsafe_allow_html=True)
        bs = st.file_uploader("", type=["jpg", "jpeg", "png"], key="back_side", label_visibility="collapsed")
        if bs:
            back_side = Image.open(bs).convert("RGB")
            st.image(back_side, use_container_width=True, caption="側背面")

    if back_front:
        if st.button("🚀 開始背部分析", use_container_width=True):
            with st.spinner("分析中..."):
                try:
                    front_bgr = pil_to_cv2(resize_image(back_front, 800))
                    side_bgr = pil_to_cv2(resize_image(back_side, 800)) if back_side else None
                    back_result = analyze_back(front_bgr, side_bgr)

                    st.success("✅ 分析完成！")

                    if back_result["detected"]:
                        sev = back_result["severity"]
                        st.markdown(
                            f"<div style='background:#1a1a2e;border-radius:10px;padding:16px;border-left:4px solid #e0c44a;margin:12px 0;'>"
                            f"<div style='color:#ffd700;font-size:1.1rem;font-weight:700;margin-bottom:8px;'>"
                            f"背部評估：<span class='badge-{sev}'>{sev}</span></div>"
                            f"<div style='color:#c0d0e0;'>{back_result['description']}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                        bc1, bc2, bc3 = st.columns(3)
                        bc1.metric("背部寬度比例", f"{back_result.get('back_width_ratio', 0):.2f}")
                        bc2.metric("肩膀寬度", f"{back_result.get('shoulder_width_px', 0):.0f}px")
                        bc3.metric("肩臀比", f"{back_result.get('shoulder_hip_ratio', 0):.2f}", delta="理想 >1.2")
                        if back_result.get("back_thickness_ratio", 0) > 0:
                            st.metric("側面背部厚度比例", f"{back_result['back_thickness_ratio']:.2f}")

                        st.markdown("### 💉 背部醫美治療建議")
                        score = back_result.get("score", 0)

                        recs_back = []
                        if score > 0.5:
                            recs_back = [
                                {"名稱": "背部肉毒毒素（豎脊肌放鬆）", "說明": "針對背部豎脊肌過度發達或肌肉緊繃，放鬆肌肉改善背部寬厚感，讓背部線條更纖細", "劑量": "每側 50-100U", "效果": "2-4 週改善，維持 4-6 個月"},
                                {"名稱": "背部溶脂針", "說明": "針對背部脂肪堆積（虎背）、背部贅肉，溶解局部脂肪讓背部線條更俐落", "劑量": "4-8ml/療程", "效果": "4-8 週改善，效果持久"},
                                {"名稱": "背部音波拉提", "說明": "改善背部皮膚鬆弛，緊緻背部輪廓，讓背部肌膚更緊緻有彈性", "劑量": "300-500 發", "效果": "3-6 月顯效，維持 12-18 個月"},
                            ]
                        else:
                            recs_back = [
                                {"名稱": "背部水光注射（保濕）", "說明": "改善背部皮膚乾燥粗糙，提升膚質與光澤", "劑量": "2-4ml", "效果": "即時保濕，維持 4-6 個月"},
                                {"名稱": "背部皮秒雷射", "說明": "改善背部色斑、毛孔粗大、膚色不均", "劑量": "1-3 次療程", "效果": "每 4-6 週一次"},
                                {"名稱": "背部肉毒毒素（局部雕塑）", "說明": "針對局部肌肉輕度發達，做精細雕塑，提升整體背部線條美感", "劑量": "每側 30-50U", "效果": "2-4 週改善，維持 4-6 個月"},
                            ]

                        for rec in recs_back:
                            st.markdown(
                                f"<div style='background:#0d1a20;border-radius:8px;padding:14px;margin:8px 0;border-left:4px solid #e0c44a;'>"
                                f"<div style='color:#ffd700;font-weight:700;font-size:0.95rem;margin-bottom:6px;'>{rec['名稱']}</div>"
                                f"<div style='color:#c0d0e0;font-size:0.88rem;margin-bottom:4px;'>{rec['說明']}</div>"
                                f"<div style='color:#90a0b0;font-size:0.84rem;'>"
                                f"💉 建議劑量：{rec['劑量']} ｜ ✨ 效果：{rec['效果']}"
                                f"</div></div>",
                                unsafe_allow_html=True
                            )
                    else:
                        st.warning("⚠️ 無法偵測背部關鍵點。請確認：背對鏡頭站立、光線充足、完整上半身入鏡。")

                except Exception as e:
                    st.error(f"❌ 分析錯誤：{e}")
                    st.exception(e)
    else:
        st.info("👆 請上傳正背面照片以開始分析")

    st.markdown("---")
    st.warning("⚠️ 本系統為輔助參考工具，分析結果不構成醫療建議，實際治療請諮詢合法執照醫師。")
```

if **name** == “**main**”:
main()
