
AI 智能醫美面診輔助系統
版本：1.0.0
執行方式：streamlit run app.py
依賴：pip install streamlit mediapipe opencv-python numpy scipy Pillow jinja2


import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import base64
import math
import io
import json
from datetime import datetime
from PIL import Image
from scipy.spatial import distance
from jinja2 import Template

# ─────────────────────────────────────────

# 0. 頁面設定

# ─────────────────────────────────────────

st.set_page_config(
page_title=“AI 智能醫美面診輔助系統”,
page_icon=“🏥”,
layout=“wide”,
initial_sidebar_state=“collapsed”,
)

# ─────────────────────────────────────────

# 1. 產品知識庫 PRODUCT_DB

# ─────────────────────────────────────────

PRODUCT_DB = {
# ── 肉毒毒素 ──────────────────────────────────────────────────────────────
“肉毒毒素_抬頭紋”: {
“category”: “肉毒毒素”,
“brands”: [“奇蹟肉毒”, “天使肉毒”, “寶提拉肉毒”, “寶妥適肉毒”, “皇家肉毒”],
“indications”: [“抬頭紋”, “額頭皺紋”],
“dose”: {“輕度”: “10-15U”, “中度”: “15-25U”, “重度”: “25-40U”},
“dose_mid”: {“輕度”: 12, “中度”: 20, “重度”: 32},
“layer”: “額肌（肌肉層）”,
“method”: “多點注射，間距約 1.5cm”,
“effect”: “注射後 3-7 天見效，抬頭紋明顯平滑，效果維持 4-6 個月”,
},
“肉毒毒素_眉間紋”: {
“category”: “肉毒毒素”,
“brands”: [“奇蹟肉毒”, “天使肉毒”, “寶提拉肉毒”, “寶妥適肉毒”, “皇家肉毒”],
“indications”: [“眉間紋”],
“dose”: {“輕度”: “10-15U”, “中度”: “15-25U”, “重度”: “25-40U”},
“dose_mid”: {“輕度”: 12, “中度”: 20, “重度”: 32},
“layer”: “皺眉肌（肌肉層）”,
“method”: “5 點標準注射法”,
“effect”: “注射後 3-7 天眉間川字紋明顯改善，效果維持 4-6 個月”,
},
“肉毒毒素_魚尾紋”: {
“category”: “肉毒毒素”,
“brands”: [“奇蹟肉毒”, “天使肉毒”, “寶提拉肉毒”, “寶妥適肉毒”, “皇家肉毒”],
“indications”: [“魚尾紋”, “眼角細紋”],
“dose”: {“輕度”: “5-10U/側”, “中度”: “10-15U/側”, “重度”: “15-20U/側”},
“dose_mid”: {“輕度”: 7, “中度”: 12, “重度”: 17},
“layer”: “眼輪匝肌（淺層肌肉）”,
“method”: “眼外角扇形多點注射（3-4 點/側）”,
“effect”: “注射後 5-7 天魚尾紋平滑，眼周緊緻，效果維持 3-5 個月”,
},
“肉毒毒素_下頜緣”: {
“category”: “肉毒毒素”,
“brands”: [“奇蹟肉毒”, “天使肉毒”, “寶提拉肉毒”, “寶妥適肉毒”, “皇家肉毒”],
“indications”: [“下頜緣鬆弛”, “頸闊肌帶”],
“dose”: {“輕度”: “10-20U”, “中度”: “20-30U”, “重度”: “30-50U”},
“dose_mid”: {“輕度”: 15, “中度”: 25, “重度”: 40},
“layer”: “頸闊肌（肌肉層）”,
“method”: “沿頸闊肌條索線性注射”,
“effect”: “下頜緣輪廓提升，頸部細紋改善，效果維持 4-6 個月”,
},
“肉毒毒素_國字臉”: {
“category”: “肉毒毒素”,
“brands”: [“奇蹟肉毒”, “天使肉毒”, “寶提拉肉毒”, “寶妥適肉毒”, “皇家肉毒”],
“indications”: [“國字臉”, “咬肌肥大”],
“dose”: {“輕度”: “20-30U/側”, “中度”: “30-40U/側”, “重度”: “40-60U/側”},
“dose_mid”: {“輕度”: 25, “中度”: 35, “重度”: 50},
“layer”: “咬肌（深層肌肉）”,
“method”: “咬肌中下 1/3 定點注射（2-3 點/側）”,
“effect”: “2-4 週咬肌縮小，臉型由方轉橢圓，效果維持 6-12 個月”,
},


# ── 喬雅登系列 ────────────────────────────────────────────────────────────
"VOLUMA_蘋果肌": {
    "category": "玻尿酸",
    "brands": ["喬雅登 VOLUMA"],
    "indications": ["蘋果肌凹陷", "顴骨下垂", "面部豐盈"],
    "dose": {"輕度": "0.5-1ml/側", "中度": "1-1.5ml/側", "重度": "1.5-2ml/側"},
    "dose_mid": {"輕度": "0.75ml", "中度": "1.25ml", "重度": "1.75ml"},
    "layer": "骨膜上層或深層皮下脂肪",
    "method": "扇形注射 / 線性推注",
    "effect": "蘋果肌圓潤飽滿，面部年輕化，效果維持 18-24 個月",
},
"VOLUMA_下巴": {
    "category": "玻尿酸",
    "brands": ["喬雅登 VOLUMA"],
    "indications": ["下巴短小", "下巴後縮"],
    "dose": {"輕度": "0.5-1ml", "中度": "1-1.5ml", "重度": "1.5-2ml"},
    "dose_mid": {"輕度": "0.75ml", "中度": "1.25ml", "重度": "1.75ml"},
    "layer": "骨膜上層",
    "method": "單點或扇形注射",
    "effect": "下巴延長、翹挺，側面輪廓改善，效果維持 12-18 個月",
},
"VOLUMA_法令紋": {
    "category": "玻尿酸",
    "brands": ["喬雅登 VOLUMA"],
    "indications": ["法令紋（深層支撐）", "鼻唇溝凹陷"],
    "dose": {"輕度": "0.5-1ml/側", "中度": "1-1.5ml/側", "重度": "1.5-2ml/側"},
    "dose_mid": {"輕度": "0.75ml", "中度": "1.25ml", "重度": "1.75ml"},
    "layer": "深層皮下 / 骨膜上層（韌帶釋放）",
    "method": "逆行線性 + 扇形",
    "effect": "法令紋深度減少 60-80%，效果維持 12-18 個月",
},
"VOLUX_下頜輪廓": {
    "category": "玻尿酸",
    "brands": ["喬雅登 VOLUX"],
    "indications": ["下頜輪廓模糊", "下頜緣下垂"],
    "dose": {"輕度": "1-2ml", "中度": "2-3ml", "重度": "3-4ml"},
    "dose_mid": {"輕度": "1.5ml", "中度": "2.5ml", "重度": "3.5ml"},
    "layer": "骨膜上層",
    "method": "線性注射，沿下頜骨緣推注",
    "effect": "下頜輪廓清晰銳利，V 臉效果，維持 18-24 個月",
},
"VOLIFT_法令紋": {
    "category": "玻尿酸",
    "brands": ["喬雅登 VOLIFT（豐麗緹）"],
    "indications": ["中重度法令紋", "木偶紋"],
    "dose": {"輕度": "0.8-1ml/側", "中度": "1-1.5ml/側", "重度": "1.5-2ml/側"},
    "dose_mid": {"輕度": "0.9ml", "中度": "1.25ml", "重度": "1.75ml"},
    "layer": "真皮深層至皮下層",
    "method": "逆行線性注射 + 蕨葉技術",
    "effect": "法令紋平滑自然，效果維持 12-15 個月",
},
"VOLBELLA_淚溝": {
    "category": "玻尿酸",
    "brands": ["喬雅登 VOLBELLA（夢蓓菈）"],
    "indications": ["淚溝凹陷", "眼周細紋"],
    "dose": {"輕度": "0.3-0.5ml/側", "中度": "0.5-1ml/側", "重度": "1-1.5ml/側"},
    "dose_mid": {"輕度": "0.4ml", "中度": "0.75ml", "重度": "1.25ml"},
    "layer": "眶隔前脂肪層（SOOF）/ 骨膜上層",
    "method": "微量多點 / 線性注射",
    "effect": "淚溝填補，黑眼圈改善，效果維持 9-12 個月",
},
"VOLITE_全臉保濕": {
    "category": "玻尿酸",
    "brands": ["喬雅登 VOLITE（芙潤）"],
    "indications": ["皮膚乾燥缺水", "膚色暗沉", "皮膚質地粗糙"],
    "dose": {"輕度": "1-2ml", "中度": "2-3ml", "重度": "3-4ml"},
    "dose_mid": {"輕度": "1.5ml", "中度": "2.5ml", "重度": "3.5ml"},
    "layer": "真皮中層（水光注射）",
    "method": "多點均勻注射 / 水光槍輔助",
    "effect": "膚質細緻水嫩，光澤提升，效果維持 6-9 個月",
},

# ── 蜂巢玻尿酸 ────────────────────────────────────────────────────────────
"蜂巢玻尿酸_大分子": {
    "category": "玻尿酸",
    "brands": ["蜂巢玻尿酸（大分子）"],
    "indications": ["下巴雕塑", "鼻樑墊高", "法令紋深層填充"],
    "dose": {"輕度": "0.5-1ml", "中度": "1-1.5ml", "重度": "1.5-2.5ml"},
    "dose_mid": {"輕度": "0.75ml", "中度": "1.25ml", "重度": "2ml"},
    "layer": "骨膜上層 / 深層皮下",
    "method": "定點注射 / 扇形推注",
    "effect": "輪廓立體感提升，效果維持 12-18 個月",
},
"蜂巢玻尿酸_中分子": {
    "category": "玻尿酸",
    "brands": ["蜂巢玻尿酸（中分子）"],
    "indications": ["法令紋", "蘋果肌", "臉頰凹陷"],
    "dose": {"輕度": "0.5-1ml/側", "中度": "1-1.5ml/側", "重度": "1.5-2ml/側"},
    "dose_mid": {"輕度": "0.75ml", "中度": "1.25ml", "重度": "1.75ml"},
    "layer": "皮下脂肪層",
    "method": "線性 + 扇形",
    "effect": "臉部豐盈飽滿，效果維持 9-12 個月",
},
"蜂巢玻尿酸_小分子": {
    "category": "玻尿酸",
    "brands": ["蜂巢玻尿酸（小分子）"],
    "indications": ["淚溝", "嘴唇豐唇", "眼周細紋"],
    "dose": {"輕度": "0.3-0.5ml", "中度": "0.5-0.8ml", "重度": "0.8-1.2ml"},
    "dose_mid": {"輕度": "0.4ml", "中度": "0.65ml", "重度": "1ml"},
    "layer": "真皮中層 / 黏膜下層（嘴唇）",
    "method": "微量多點注射",
    "effect": "局部細膩修飾，效果維持 6-9 個月",
},

# ── 薇貝拉 ────────────────────────────────────────────────────────────────
"薇貝拉50MG": {
    "category": "玻尿酸",
    "brands": ["薇貝拉 50MG"],
    "indications": ["全臉水光補水", "淺層細紋", "膚質改善"],
    "dose": {"輕度": "1ml", "中度": "1.5-2ml", "重度": "2-3ml"},
    "dose_mid": {"輕度": "1ml", "中度": "1.75ml", "重度": "2.5ml"},
    "layer": "真皮淺層",
    "method": "水光槍多點均勻注射",
    "effect": "皮膚保濕彈力提升，膚色均勻，效果維持 4-6 個月",
},
"薇貝拉200MG": {
    "category": "玻尿酸",
    "brands": ["薇貝拉 200MG"],
    "indications": ["中重度法令紋", "深層填充", "顴骨豐盈"],
    "dose": {"輕度": "0.5-1ml/側", "中度": "1-1.5ml/側", "重度": "1.5-2ml/側"},
    "dose_mid": {"輕度": "0.75ml", "中度": "1.25ml", "重度": "1.75ml"},
    "layer": "深層皮下 / 骨膜上層",
    "method": "扇形 / 逆行線性注射",
    "effect": "深層支撐強，效果持久維持 12-18 個月",
},

# ── 海菲亞 ────────────────────────────────────────────────────────────────
"海菲亞玻尿酸": {
    "category": "玻尿酸",
    "brands": ["海菲亞玻尿酸"],
    "indications": ["法令紋", "蘋果肌", "臉頰填充", "下巴"],
    "dose": {"輕度": "1ml/側", "中度": "1.5ml/側", "重度": "2ml/側"},
    "dose_mid": {"輕度": "1ml", "中度": "1.5ml", "重度": "2ml"},
    "layer": "皮下脂肪層 / 骨膜上層",
    "method": "線性 + 扇形注射",
    "effect": "臉部填充效果自然，維持 12-18 個月",
},

# ── 舒顏萃 Sculptra ───────────────────────────────────────────────────────
"舒顏萃Sculptra": {
    "category": "膠原蛋白增生劑",
    "brands": ["舒顏萃 Sculptra"],
    "indications": ["整體老化鬆弛", "臉部體積流失", "法令紋", "蘋果肌下垂"],
    "dose": {"輕度": "1瓶（全臉）", "中度": "2瓶（全臉）", "重度": "3-4瓶（分次）"},
    "dose_mid": {"輕度": "1瓶", "中度": "2瓶", "重度": "3瓶"},
    "layer": "真皮深層至皮下層",
    "method": "扇形大範圍注射，按摩充分分散",
    "effect": "刺激膠原蛋白新生，效果 2-3 個月逐漸顯現，維持 18-24 個月",
},

# ── 麗珠蘭 PN ─────────────────────────────────────────────────────────────
"麗珠蘭PN1%": {
    "category": "PN 核酸修復",
    "brands": ["麗珠蘭 PN 1%"],
    "indications": ["皮膚暗沉", "膚質修復", "淺層細紋", "輕度老化"],
    "dose": {"輕度": "1ml", "中度": "1.5-2ml", "重度": "2-3ml"},
    "dose_mid": {"輕度": "1ml", "中度": "1.75ml", "重度": "2.5ml"},
    "layer": "真皮淺中層",
    "method": "水光槍 / 多點注射",
    "effect": "細胞修復活化，膚色提亮，每 2-4 週一次，建議 3-4 療程",
},
"麗珠蘭PN2%": {
    "category": "PN 核酸修復",
    "brands": ["麗珠蘭 PN 2%"],
    "indications": ["中重度老化", "深層修復", "膠原蛋白重建"],
    "dose": {"輕度": "1ml", "中度": "2ml", "重度": "3ml"},
    "dose_mid": {"輕度": "1ml", "中度": "2ml", "重度": "3ml"},
    "layer": "真皮中深層",
    "method": "多點注射 / 線性",
    "effect": "深層修復，促進膠原新生，效果較 1% 更顯著持久",
},

# ── PDO 埋線 ──────────────────────────────────────────────────────────────
"PDO小線螺旋": {
    "category": "埋線提拉",
    "brands": ["PDO 小線螺旋線"],
    "indications": ["皮膚鬆弛改善", "膚質提升", "毛孔縮小"],
    "dose": {"輕度": "20-30 根", "中度": "30-50 根", "重度": "50-80 根"},
    "dose_mid": {"輕度": "25根", "中度": "40根", "重度": "65根"},
    "layer": "真皮層",
    "method": "均勻密植，順紋路方向佈線",
    "effect": "膚質緊緻，膠原增生，效果維持 6-9 個月",
},
"鼻雕埋線": {
    "category": "埋線提拉",
    "brands": ["鼻雕埋線"],
    "indications": ["鼻樑塌陷", "鼻尖低垂"],
    "dose": {"輕度": "2-4 根", "中度": "4-6 根", "重度": "6-8 根"},
    "dose_mid": {"輕度": "3根", "中度": "5根", "重度": "7根"},
    "layer": "鼻背皮下層",
    "method": "沿鼻樑中線置入",
    "effect": "鼻樑挺直、鼻尖上翹，效果維持 6-12 個月",
},
"鳳凰埋線": {
    "category": "埋線提拉",
    "brands": ["鳳凰埋線（大V線）"],
    "indications": ["臉部鬆弛下垂", "法令紋", "下頜緣模糊", "蘋果肌下垂"],
    "dose": {"輕度": "4-6 根/側", "中度": "6-10 根/側", "重度": "10-16 根/側"},
    "dose_mid": {"輕度": "5根/側", "中度": "8根/側", "重度": "13根/側"},
    "layer": "SMAS 筋膜層 / 深層皮下",
    "method": "逆行進針，錨定點固定，雙向倒鉤提拉",
    "effect": "即時提拉效果顯著，效果維持 12-18 個月，建議搭配其他療程",
},

# ── 紐拉玻尿酸 ────────────────────────────────────────────────────────────
"紐拉玻尿酸": {
    "category": "玻尿酸",
    "brands": ["紐拉玻尿酸"],
    "indications": ["法令紋", "木偶紋", "臉頰豐盈", "下巴"],
    "dose": {"輕度": "0.5-1ml/側", "中度": "1-1.5ml/側", "重度": "1.5-2ml/側"},
    "dose_mid": {"輕度": "0.75ml", "中度": "1.25ml", "重度": "1.75ml"},
    "layer": "皮下脂肪層",
    "method": "線性 + 點狀注射",
    "effect": "自然填補，效果維持 9-15 個月",
},

# ── 溶脂針 ────────────────────────────────────────────────────────────────
"韓國溶脂針": {
    "category": "溶脂針",
    "brands": ["韓國溶脂針"],
    "indications": ["局部脂肪堆積", "雙下巴", "臉頰脂肪"],
    "dose": {"輕度": "1-2ml/區域", "中度": "2-4ml/區域", "重度": "4-6ml/區域"},
    "dose_mid": {"輕度": "1.5ml", "中度": "3ml", "重度": "5ml"},
    "layer": "皮下脂肪層",
    "method": "多點均勻注射，按摩分散",
    "effect": "4-8 週脂肪減少，臉型緊緻，建議 2-3 次療程",
},
"倍克脂溶脂針": {
    "category": "溶脂針",
    "brands": ["倍克脂溶脂針"],
    "indications": ["雙下巴脂肪", "頸部脂肪"],
    "dose": {"輕度": "2ml", "中度": "4ml", "重度": "6ml"},
    "dose_mid": {"輕度": "2ml", "中度": "4ml", "重度": "6ml"},
    "layer": "頦下皮下脂肪層",
    "method": "格狀多點注射",
    "effect": "溶解頦下脂肪，改善雙下巴，效果維持長久（脂肪細胞永久消除）",
},

# ── 能量療程 ──────────────────────────────────────────────────────────────
"皮秒雷射": {
    "category": "能量療程",
    "brands": ["皮秒雷射"],
    "indications": ["皮膚暗沉", "色斑", "毛孔粗大", "膚質不均"],
    "dose": {"輕度": "單次療程", "中度": "3-5 次療程", "重度": "5-8 次療程"},
    "dose_mid": {"輕度": "1次", "中度": "4次", "重度": "6次"},
    "layer": "表皮至真皮層",
    "method": "全臉掃描，依斑點加強",
    "effect": "膚色均勻提亮，斑點淡化，毛孔縮小，每 4-6 週一次",
},
"音波拉提": {
    "category": "能量療程",
    "brands": ["音波拉提 (Ultherapy)"],
    "indications": ["面部鬆弛下垂", "下頜緣模糊", "頸部鬆弛"],
    "dose": {"輕度": "200-400 發（局部）", "中度": "400-600 發（半臉）", "重度": "600-1000 發（全臉+頸）"},
    "dose_mid": {"輕度": "300發", "中度": "500發", "重度": "800發"},
    "layer": "SMAS 筋膜層（4.5mm 探頭）+ 真皮層（3mm 探頭）",
    "method": "線性掃描，依解剖層次分層施打",
    "effect": "刺激膠原新生，3-6 個月顯效，提拉效果維持 12-18 個月",
},
"電波拉提": {
    "category": "能量療程",
    "brands": ["電波拉提 (Thermage)"],
    "indications": ["皮膚鬆弛", "法令紋", "眼周細紋"],
    "dose": {"輕度": "900 發（臉）", "中度": "1200 發（臉+頸）", "重度": "1500 發（全臉頸+眼）"},
    "dose_mid": {"輕度": "900發", "中度": "1200發", "重度": "1500發"},
    "layer": "真皮深層至皮下層（射頻穿透）",
    "method": "全臉均勻掃描，多遍疊加",
    "effect": "即時緊緻感，3 個月顯著改善，效果維持 12-24 個月",
},


}

# ─────────────────────────────────────────

# 2. MediaPipe 初始化

# ─────────────────────────────────────────

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ─────────────────────────────────────────

# 3. 輔助函式

# ─────────────────────────────────────────

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
“”“PIL → OpenCV (BGR)”””
return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
“”“OpenCV (BGR) → PIL”””
return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def image_to_base64(img: Image.Image, fmt: str = “JPEG”) -> str:
“”“PIL Image → base64 字串”””
buf = io.BytesIO()
img.save(buf, format=fmt)
return base64.b64encode(buf.getvalue()).decode(“utf-8”)

def resize_image(img: Image.Image, max_size: int = 800) -> Image.Image:
“”“保持比例縮放至 max_size”””
w, h = img.size
if max(w, h) > max_size:
ratio = max_size / max(w, h)
img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
return img

def get_landmarks(img_bgr: np.ndarray):
“””
使用 MediaPipe Face Mesh 提取 468 個 3D 關鍵點
回傳：(landmarks_list, results) or (None, None)
“””
with mp_face_mesh.FaceMesh(
static_image_mode=True,
max_num_faces=1,
refine_landmarks=True,
min_detection_confidence=0.5,
) as face_mesh:
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
results = face_mesh.process(img_rgb)


if not results.multi_face_landmarks:
    return None, None

h, w = img_bgr.shape[:2]
lm = results.multi_face_landmarks[0].landmark
# 轉換為像素座標
landmarks = np.array([[l.x * w, l.y * h, l.z * w] for l in lm])
return landmarks, results


def estimate_yaw(landmarks: np.ndarray) -> float:
“””
估算頭部偏轉角（Yaw），單位：度
使用鼻尖[1]、左眼外角[33]、右眼外角[263] 的相對位置
“””
nose = landmarks[1]
left_eye = landmarks[33]
right_eye = landmarks[263]


# 左右眼相對於鼻尖的 x 距離
left_dist = abs(nose[0] - left_eye[0])
right_dist = abs(nose[0] - right_eye[0])

total = left_dist + right_dist
if total < 1e-6:
    return 0.0

# 正面時左右距離相等（ratio≈0.5），偏左時 left_dist 縮小
ratio = left_dist / total
# 映射到 yaw：正面=0°，完全左側=~-90°
yaw = (ratio - 0.5) * 180.0
return yaw


def classify_severity(value: float) -> str:
“”“將 0~1 的分數分級”””
if value < 0.3:
return “輕度”
elif value < 0.7:
return “中度”
else:
return “重度”

def safe_depth_std(landmarks: np.ndarray, indices: list) -> float:
“”“計算指定關鍵點 z 值的標準差（用於皺紋分析）”””
pts = landmarks[indices, 2]
return float(np.std(pts))

def normalize(value: float, min_v: float, max_v: float) -> float:
“”“線性歸一化至 [0, 1]”””
if max_v - min_v < 1e-6:
return 0.0
return max(0.0, min(1.0, (value - min_v) / (max_v - min_v)))

# ─────────────────────────────────────────

# 4. 特徵分析計算

# ─────────────────────────────────────────

def analyze_face(landmarks: np.ndarray, img_bgr: np.ndarray) -> dict:
“””
根據 468 個 3D 關鍵點計算各項面部特徵指標。
回傳字典，key 為特徵名稱，value 為 dict(score, severity, raw)
“””
h, w = img_bgr.shape[:2]
results = {}


# ── 三庭比例 ──────────────────────────────────────────────────────────────
# 上庭：髮際線[10] ~ 眉間[8]
# 中庭：眉間[8] ~ 鼻下[94]
# 下庭：鼻下[94] ~ 下巴[152]
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

# 理想比例各1/3 ≈ 0.333；差異越大分數越高
ideal = 1.0 / 3.0
upper_dev = abs(upper_ratio - ideal) / ideal
middle_dev = abs(middle_ratio - ideal) / ideal
lower_dev = abs(lower_ratio - ideal) / ideal

results["三庭比例"] = {
    "upper_ratio": round(upper_ratio, 3),
    "middle_ratio": round(middle_ratio, 3),
    "lower_ratio": round(lower_ratio, 3),
    "dominant": "上庭" if upper_ratio > middle_ratio and upper_ratio > lower_ratio
                else "中庭" if middle_ratio >= upper_ratio and middle_ratio > lower_ratio
                else "下庭",
}

# ── 五眼比例 ──────────────────────────────────────────────────────────────
# 臉寬：[234] ~ [454]（耳前點）
# 眼寬：[33]~[133]（左）, [362]~[263]（右）
# 兩眼距：[133]~[362]（內眼角間距）
face_width = abs(landmarks[454, 0] - landmarks[234, 0]) + 1e-6
left_eye_w = abs(landmarks[133, 0] - landmarks[33, 0])
right_eye_w = abs(landmarks[263, 0] - landmarks[362, 0])
inter_eye = abs(landmarks[362, 0] - landmarks[133, 0])

eye_avg = (left_eye_w + right_eye_w) / 2.0 + 1e-6
# 理想五眼：臉寬 = 5 * 眼寬，眼距 ≈ 1 眼寬
five_eye_ratio = face_width / (5 * eye_avg)  # 理想為 1.0
inter_ratio = inter_eye / eye_avg  # 理想為 1.0

results["五眼比例"] = {
    "face_width": round(face_width, 1),
    "eye_avg_width": round(eye_avg, 1),
    "inter_eye_distance": round(inter_eye, 1),
    "five_eye_ratio": round(five_eye_ratio, 3),
    "inter_ratio": round(inter_ratio, 3),
}

# ── 對稱性 ────────────────────────────────────────────────────────────────
# 對應點對（左[i] ~ 右[j]），計算 x 方向與鼻中線的偏差對稱性
nose_center_x = landmarks[1, 0]
sym_pairs = [(33, 263), (133, 362), (234, 454), (61, 291), (50, 280), (149, 378)]
asym_sum = 0.0
for li, ri in sym_pairs:
    l_dist = abs(landmarks[li, 0] - nose_center_x)
    r_dist = abs(landmarks[ri, 0] - nose_center_x)
    pair_max = max(l_dist, r_dist, 1e-6)
    asym_sum += abs(l_dist - r_dist) / pair_max

asym_score = asym_sum / len(sym_pairs)  # 0=完全對稱，1=極度不對稱
asym_norm = normalize(asym_score, 0.0, 0.3)
results["對稱性"] = {
    "score": round(asym_norm, 3),
    "severity": classify_severity(asym_norm),
    "raw": round(asym_score, 4),
    "description": "臉部左右對稱性",
}

# ── 淚溝凹陷 ──────────────────────────────────────────────────────────────
# 下眼瞼中部 [159,145] 與臉頰 [50,280] 的深度差
tear_z_avg = np.mean(landmarks[[159, 145], 2])
cheek_z_avg = np.mean(landmarks[[50, 280], 2])
tear_depth = abs(tear_z_avg - cheek_z_avg)
tear_norm = normalize(tear_depth, 0.001, 0.015)
results["淚溝"] = {
    "score": round(tear_norm, 3),
    "severity": classify_severity(tear_norm),
    "raw": round(tear_depth, 5),
    "description": "淚溝凹陷程度",
}

# ── 法令紋 ────────────────────────────────────────────────────────────────
# 鼻翼外側[49,279] 與嘴角[61,291] 深度差 + 鼻唇溝長度
nasal_z = np.mean(landmarks[[49, 279], 2])
mouth_corner_z = np.mean(landmarks[[61, 291], 2])
nasolabial_depth = abs(nasal_z - mouth_corner_z)

# 長度：鼻翼點到嘴角的直線距離
left_nasolabial_len = np.linalg.norm(landmarks[49, :2] - landmarks[61, :2])
right_nasolabial_len = np.linalg.norm(landmarks[279, :2] - landmarks[291, :2])
nasolabial_len_avg = (left_nasolabial_len + right_nasolabial_len) / 2.0

# 正規化（深度 + 長度綜合評估）
depth_norm = normalize(nasolabial_depth, 0.001, 0.015)
len_norm = normalize(nasolabial_len_avg, face_width * 0.1, face_width * 0.25)
nasolabial_score = 0.6 * depth_norm + 0.4 * len_norm
results["法令紋"] = {
    "score": round(nasolabial_score, 3),
    "severity": classify_severity(nasolabial_score),
    "raw_depth": round(nasolabial_depth, 5),
    "raw_length": round(nasolabial_len_avg, 2),
    "description": "法令紋（鼻唇溝）深度與長度",
}

# ── 蘋果肌 ────────────────────────────────────────────────────────────────
# 顴骨最高點[117,346] 凸度（相對深度）
cheekbone_z = np.mean(landmarks[[117, 346], 2])
cheek_ref_z = np.mean(landmarks[[50, 280], 2])
apple_muscle = abs(cheekbone_z - cheek_ref_z)
# 蘋果肌飽滿→凸度大→score 低（凹陷問題）
apple_norm = normalize(apple_muscle, 0.001, 0.012)
apple_score = 1.0 - apple_norm  # 轉換：score 高 = 蘋果肌凹陷/下垂
results["蘋果肌"] = {
    "score": round(apple_score, 3),
    "severity": classify_severity(apple_score),
    "raw": round(apple_muscle, 5),
    "description": "蘋果肌飽滿度（0=飽滿，1=凹陷）",
}

# ── 下頜緣輪廓 ────────────────────────────────────────────────────────────
# 下頜點 y 座標曲率（不規則程度）
jaw_indices = [152, 172, 171, 170, 169, 136, 135, 134, 58, 172]
jaw_pts = landmarks[jaw_indices, :2]
# 計算與最低點的 y 方向偏差
jaw_y_mean = np.mean(jaw_pts[:, 1])
jaw_curvature_std = np.std(jaw_pts[:, 1])
jaw_norm = normalize(jaw_curvature_std, 2.0, 25.0)
results["下頜緣"] = {
    "score": round(jaw_norm, 3),
    "severity": classify_severity(jaw_norm),
    "raw": round(jaw_curvature_std, 3),
    "description": "下頜緣輪廓清晰度（0=清晰，1=鬆弛）",
}

# ── 額頭皺紋 ──────────────────────────────────────────────────────────────
forehead_indices = [55, 107, 66, 105, 65, 52, 53, 46, 124, 156, 70, 63]
forehead_std = safe_depth_std(landmarks, forehead_indices)
forehead_norm = normalize(forehead_std, 0.001, 0.008)
results["抬頭紋"] = {
    "score": round(forehead_norm, 3),
    "severity": classify_severity(forehead_norm),
    "raw": round(forehead_std, 5),
    "description": "額頭橫紋深度",
}

# ── 眉間紋 ────────────────────────────────────────────────────────────────
glabella_indices = [168, 6, 197, 195, 5, 4, 8, 9]
glabella_std = safe_depth_std(landmarks, glabella_indices)
glabella_norm = normalize(glabella_std, 0.001, 0.008)
results["眉間紋"] = {
    "score": round(glabella_norm, 3),
    "severity": classify_severity(glabella_norm),
    "raw": round(glabella_std, 5),
    "description": "眉間川字紋深度",
}

# ── 魚尾紋 ────────────────────────────────────────────────────────────────
crow_feet_indices = [33, 133, 246, 161, 160, 159, 263, 362, 466, 388, 387, 386]
crow_std = safe_depth_std(landmarks, crow_feet_indices)
crow_norm = normalize(crow_std, 0.001, 0.01)
results["魚尾紋"] = {
    "score": round(crow_norm, 3),
    "severity": classify_severity(crow_norm),
    "raw": round(crow_std, 5),
    "description": "眼外角魚尾紋深度",
}

# ── 下巴形狀 ──────────────────────────────────────────────────────────────
chin_tip = landmarks[152]
chin_left = landmarks[172]
chin_right = landmarks[397]
chin_width = abs(chin_left[0] - chin_right[0])
chin_length = abs(chin_tip[1] - subnasale_y)
chin_ratio = chin_length / (chin_width + 1e-6)  # 越大越尖

chin_score = normalize(chin_ratio, 0.3, 1.0)
results["下巴"] = {
    "score": round(chin_score, 3),
    "severity": classify_severity(chin_score),
    "chin_ratio": round(chin_ratio, 3),
    "description": "下巴長寬比（高=尖型，低=寬型）",
}

# ── 皮膚質地 ──────────────────────────────────────────────────────────────
# 使用臉頰區域關鍵點深度變異係數
skin_indices = [50, 280, 205, 425, 117, 346, 187, 411]
skin_z = landmarks[skin_indices, 2]
skin_cv = np.std(skin_z) / (np.mean(np.abs(skin_z)) + 1e-6)
skin_norm = normalize(skin_cv, 0.05, 0.6)
results["皮膚質地"] = {
    "score": round(skin_norm, 3),
    "severity": classify_severity(skin_norm),
    "raw": round(skin_cv, 4),
    "description": "皮膚粗糙度（關鍵點深度變異）",
}

return results


# ─────────────────────────────────────────

# 5. 治療建議生成

# ─────────────────────────────────────────

# 問題 → 相關產品 key 的映射表

PROBLEM_TO_PRODUCTS = {
“抬頭紋”:  [“肉毒毒素_抬頭紋”, “皮秒雷射”, “麗珠蘭PN1%”],
“眉間紋”:  [“肉毒毒素_眉間紋”, “麗珠蘭PN2%”],
“魚尾紋”:  [“肉毒毒素_魚尾紋”, “VOLBELLA_淚溝”, “電波拉提”],
“淚溝”:    [“VOLBELLA_淚溝”, “蜂巢玻尿酸_小分子”, “麗珠蘭PN1%”],
“法令紋”:  [“VOLUMA_法令紋”, “VOLIFT_法令紋”, “蜂巢玻尿酸_中分子”, “舒顏萃Sculptra”, “鳳凰埋線”],
“蘋果肌”:  [“VOLUMA_蘋果肌”, “蜂巢玻尿酸_中分子”, “舒顏萃Sculptra”, “鳳凰埋線”],
“下頜緣”:  [“VOLUX_下頜輪廓”, “肉毒毒素_下頜緣”, “音波拉提”, “鳳凰埋線”],
“下巴”:    [“VOLUMA_下巴”, “蜂巢玻尿酸_大分子”, “紐拉玻尿酸”],
“皮膚質地”: [“VOLITE_全臉保濕”, “薇貝拉50MG”, “麗珠蘭PN1%”, “皮秒雷射”, “PDO小線螺旋”],
“對稱性”:  [“肉毒毒素_眉間紋”, “VOLUMA_蘋果肌”],
}

SCORE_THRESHOLD = 0.25  # 低於此值不建議治療

def generate_recommendations(analysis: dict) -> list:
“””
根據分析結果生成治療建議列表。
回傳：[{problem, score, severity, recommendations: [{product_key, product_info, dose}]}]
“””
rec_list = []
processed_problems = []


for problem, data in analysis.items():
    if problem in ("三庭比例", "五眼比例"):
        continue  # 這兩個是比例分析，不直接對應治療
    if not isinstance(data, dict) or "score" not in data:
        continue

    score = data.get("score", 0)
    severity = data.get("severity", "輕度")

    if score < SCORE_THRESHOLD:
        continue

    if problem not in PROBLEM_TO_PRODUCTS:
        continue

    products = []
    for pk in PROBLEM_TO_PRODUCTS[problem]:
        if pk in PRODUCT_DB:
            prod = PRODUCT_DB[pk]
            products.append({
                "product_key": pk,
                "product_name": prod["brands"][0],
                "category": prod.get("category", ""),
                "dose": prod["dose"].get(severity, "依診療評估"),
                "dose_mid": prod["dose_mid"].get(severity, "依醫師評估"),
                "layer": prod.get("layer", ""),
                "method": prod.get("method", ""),
                "effect": prod.get("effect", ""),
            })

    if products:
        rec_list.append({
            "problem": problem,
            "description": data.get("description", ""),
            "score": score,
            "severity": severity,
            "recommendations": products,
        })

return rec_list


# ─────────────────────────────────────────

# 6. 面相學評估

# ─────────────────────────────────────────

def physiognomy_reading(analysis: dict) -> list:
“”“根據面部比例與特徵給出面相學解讀”””
readings = []
three_zones = analysis.get(“三庭比例”, {})
five_eyes = analysis.get(“五眼比例”, {})


dominant = three_zones.get("dominant", "")
upper_r = three_zones.get("upper_ratio", 0.333)
middle_r = three_zones.get("middle_ratio", 0.333)
lower_r = three_zones.get("lower_ratio", 0.333)

# 三庭解讀
if dominant == "上庭":
    readings.append({
        "aspect": "上庭豐盛",
        "icon": "⭐",
        "reading": "上庭寬闊飽滿，代表早年運勢強健，智慧過人，父母緣深厚。然而需留意人際關係，避免因個性強勢而引發摩擦。",
    })
elif dominant == "中庭":
    readings.append({
        "aspect": "中庭主運",
        "icon": "💼",
        "reading": "中庭比例突出，中年事業運旺，適合自行創業或擔任管理職位。個性較為固執，宜學習傾聽他人意見，有助事業更上層樓。",
    })
elif dominant == "下庭":
    readings.append({
        "aspect": "下庭豐厚",
        "icon": "🌟",
        "reading": "下庭長且豐厚，晚年運勢佳，晚福深厚，子孫有緣。惟感情路上易有波折，需培養穩定的情感觀念。",
    })

# 眼距解讀
inter_r = five_eyes.get("inter_ratio", 1.0)
if inter_r < 0.85:
    readings.append({
        "aspect": "眉眼距窄",
        "icon": "⚡",
        "reading": "兩眼距離較窄，個性敏銳、反應快，但容易急躁衝動。宜學習放慢步調，深呼吸思考後再做決定。",
    })
elif inter_r > 1.2:
    readings.append({
        "aspect": "眉眼距寬",
        "icon": "🌊",
        "reading": "兩眼間距寬廣，心胸寬大，待人寬容，適合從事公關、外交等需廣結善緣的工作。",
    })

# 法令紋
nasolabial = analysis.get("法令紋", {})
if nasolabial.get("score", 0) > 0.5:
    readings.append({
        "aspect": "法令紋顯現",
        "icon": "👑",
        "reading": "法令紋深且明顯，面相學中象徵威嚴與領導力，主掌大局之相。惟外觀顯老，建議適當醫美調理以展現最佳狀態。",
    })

# 下巴形狀
chin = analysis.get("下巴", {})
chin_ratio = chin.get("chin_ratio", 0.5)
if chin_ratio > 0.7:
    readings.append({
        "aspect": "下巴尖細",
        "icon": "💕",
        "reading": "下巴尖秀，感情細膩豐富，藝術天分出眾。但內心缺乏安全感，在感情中需要更多的肯定與陪伴。",
    })
elif chin_ratio < 0.4:
    readings.append({
        "aspect": "下巴寬厚",
        "icon": "🏛️",
        "reading": "下巴寬厚有力，意志力堅定，做事有始有終，晚年物質生活豐裕，是福氣之相。",
    })

# 對稱性
sym = analysis.get("對稱性", {})
if sym.get("score", 0) > 0.5:
    readings.append({
        "aspect": "面相不對稱",
        "icon": "🎭",
        "reading": "臉部左右有一定程度的不對稱，面相學認為此象徵內外不一、表裡有別，個性複雜多面。適當調理後可提升整體協調感。",
    })

# 皮膚質地
skin = analysis.get("皮膚質地", {})
if skin.get("score", 0) < 0.3:
    readings.append({
        "aspect": "膚質細緻",
        "icon": "✨",
        "reading": "膚質細膩均勻，面相學中代表貴人緣佳，行事順遂，深得他人喜愛。",
    })

# 若無特殊解讀，給基本解讀
if not readings:
    readings.append({
        "aspect": "五官均衡",
        "icon": "🌸",
        "reading": "您的面部比例均衡，五官協調，面相學中屬於平穩溫和之相，一生運勢平順，人際緣分佳。",
    })

return readings


# ─────────────────────────────────────────

# 7. 繪製標註圖

# ─────────────────────────────────────────

def draw_annotations(img_bgr: np.ndarray, landmarks: np.ndarray, analysis: dict) -> np.ndarray:
“”“在圖片上繪製三庭五眼標線和關鍵點”””
img = img_bgr.copy()
h, w = img.shape[:2]


def pt(idx):
    return (int(landmarks[idx, 0]), int(landmarks[idx, 1]))

# 顏色
COLOR_LINE = (0, 220, 255)   # 黃色
COLOR_PT = (0, 100, 255)     # 橘紅
COLOR_TEXT = (255, 255, 255)

# ── 三庭橫線 ──────────────────────────────────────────────────────────────
lines_y = {
    "髮際": int(landmarks[10, 1]),
    "眉間": int(landmarks[8, 1]),
    "鼻下": int(landmarks[94, 1]),
    "下巴": int(landmarks[152, 1]),
}
for label, y in lines_y.items():
    cv2.line(img, (0, y), (w, y), COLOR_LINE, 1)
    cv2.putText(img, label, (5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

# ── 五眼垂直線 ────────────────────────────────────────────────────────────
eye_x_pts = [landmarks[234, 0], landmarks[33, 0], landmarks[133, 0],
             landmarks[362, 0], landmarks[263, 0], landmarks[454, 0]]
for x in eye_x_pts:
    cv2.line(img, (int(x), 0), (int(x), h), (200, 200, 0), 1)

# ── 關鍵點（部分展示） ────────────────────────────────────────────────────
key_points = [10, 8, 94, 152, 33, 133, 362, 263, 234, 454,
              159, 145, 49, 279, 61, 291, 117, 346, 1]
for idx in key_points:
    cv2.circle(img, pt(idx), 3, COLOR_PT, -1)

# ── 三庭標籤（右側） ──────────────────────────────────────────────────────
zones = analysis.get("三庭比例", {})
labels = [
    (lines_y["髮際"], lines_y["眉間"], f'上庭 {zones.get("upper_ratio", 0):.1%}'),
    (lines_y["眉間"], lines_y["鼻下"], f'中庭 {zones.get("middle_ratio", 0):.1%}'),
    (lines_y["鼻下"], lines_y["下巴"], f'下庭 {zones.get("lower_ratio", 0):.1%}'),
]
for y1, y2, text in labels:
    mid_y = (y1 + y2) // 2
    cv2.putText(img, text, (w - 130, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_LINE, 1, cv2.LINE_AA)

return img


# ─────────────────────────────────────────

# 8. HTML 報告生成

# ─────────────────────────────────────────

HTML_TEMPLATE = “””

<!DOCTYPE html>

<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 醫美面診報告</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
         background: #0f0f1a; color: #e0e0e0; line-height: 1.6; }
  .container { max-width: 1100px; margin: 0 auto; padding: 20px; }

/* Header */
.header { text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
border-radius: 16px; margin-bottom: 30px; }
.header h1 { font-size: 2.2rem; color: #e0c44a; letter-spacing: 2px; }
.header p { color: #a0b4c8; margin-top: 8px; font-size: 0.95rem; }
.report-date { font-size: 0.85rem; color: #7090a0; margin-top: 6px; }

/* Section */
.section { background: #1a1a2e; border-radius: 12px; padding: 24px; margin-bottom: 24px;
border-left: 4px solid #e0c44a; }
.section-title { font-size: 1.25rem; color: #e0c44a; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }

/* Photos */
.photo-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.photo-card { text-align: center; }
.photo-card img { width: 100%; border-radius: 8px; border: 2px solid #2a3a5a; }
.photo-label { font-size: 0.85rem; color: #a0b4c8; margin-top: 6px; }

/* Proportion bars */
.proportion-row { display: flex; align-items: center; margin: 8px 0; gap: 12px; }
.proportion-label { min-width: 60px; font-size: 0.9rem; color: #c0d0e0; }
.bar-bg { flex: 1; background: #0f1a2a; border-radius: 20px; height: 14px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 20px; transition: width 0.5s; }
.bar-fill.ideal { background: linear-gradient(90deg, #2ecc71, #1abc9c); }
.bar-fill.over { background: linear-gradient(90deg, #e0c44a, #e67e22); }
.bar-fill.low { background: linear-gradient(90deg, #3498db, #2980b9); }
.proportion-val { font-size: 0.85rem; color: #e0c44a; min-width: 50px; text-align: right; }

/* Severity badge */
.badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
.badge-輕度 { background: #1a3a2a; color: #2ecc71; border: 1px solid #2ecc71; }
.badge-中度 { background: #3a2a0a; color: #e0c44a; border: 1px solid #e0c44a; }
.badge-重度 { background: #3a1a1a; color: #e74c3c; border: 1px solid #e74c3c; }

/* Score meter */
.score-meter { display: flex; align-items: center; gap: 10px; }
.score-track { flex: 1; height: 8px; background: #0f1a2a; border-radius: 4px; overflow: hidden; }
.score-fill { height: 100%; border-radius: 4px; }

/* Problem list */
.problem-card { background: #0f1a2a; border-radius: 10px; padding: 18px; margin-bottom: 16px; }
.problem-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.problem-name { font-size: 1.05rem; color: #fff; font-weight: 600; }
.problem-desc { font-size: 0.82rem; color: #7090a0; margin-bottom: 10px; }
.rec-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.rec-table th { background: #1a2a3a; color: #a0c0d8; padding: 8px 10px; text-align: left; border-bottom: 1px solid #2a3a4a; }
.rec-table td { padding: 8px 10px; border-bottom: 1px solid #151f2f; color: #c0d0e0; vertical-align: top; }
.rec-table tr:last-child td { border-bottom: none; }
.product-name { color: #e0c44a; font-weight: 600; }

/* Physiognomy */
.physio-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }
.physio-card { background: #0f1a2a; border-radius: 10px; padding: 16px; }
.physio-icon { font-size: 1.6rem; margin-bottom: 8px; }
.physio-aspect { font-size: 0.95rem; color: #e0c44a; font-weight: 600; margin-bottom: 6px; }
.physio-text { font-size: 0.85rem; color: #a0b4c8; line-height: 1.7; }

/* Disclaimer */
.disclaimer { background: #1a0f0f; border: 1px solid #4a2a2a; border-radius: 10px;
padding: 16px; font-size: 0.82rem; color: #c08080; text-align: center; margin-top: 30px; }

/* Five-eye analysis */
.five-eye-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.stat-box { background: #0f1a2a; border-radius: 8px; padding: 14px; text-align: center; }
.stat-label { font-size: 0.78rem; color: #7090a0; }
.stat-value { font-size: 1.2rem; color: #e0c44a; font-weight: 700; margin-top: 4px; }

@media (max-width: 600px) {
.photo-grid { grid-template-columns: 1fr; }
.five-eye-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>

</head>
<body>
<div class="container">

  <!-- Header -->

  <div class="header">
    <h1>🏥 AI 智能醫美面診報告</h1>
    <p>由 AI 面部分析系統生成，僅供參考</p>
    <div class="report-date">報告日期：{{ report_date }}</div>
  </div>

  <!-- 照片縮圖 -->

  <div class="section">
    <div class="section-title">📷 上傳照片</div>
    <div class="photo-grid">
      <div class="photo-card">
        <img src="data:image/jpeg;base64,{{ img_front }}" alt="正面">
        <div class="photo-label">正面</div>
      </div>
      <div class="photo-card">
        <img src="data:image/jpeg;base64,{{ img_left45 }}" alt="左側45°">
        <div class="photo-label">左側 45°</div>
      </div>
      <div class="photo-card">
        <img src="data:image/jpeg;base64,{{ img_left90 }}" alt="左側90°">
        <div class="photo-label">左側 90°</div>
      </div>
    </div>
  </div>

  <!-- 標註圖 -->

  <div class="section">
    <div class="section-title">🎯 三庭五眼標註分析</div>
    <div class="photo-card">
      <img src="data:image/jpeg;base64,{{ img_annotated }}" alt="標註圖" style="max-width:480px; margin:0 auto; display:block;">
    </div>
  </div>

  <!-- 三庭比例 -->

  <div class="section">
    <div class="section-title">📐 三庭比例分析</div>
    <p style="font-size:0.85rem;color:#7090a0;margin-bottom:14px;">理想比例：上庭 ≈ 中庭 ≈ 下庭 ≈ 33.3%</p>


{% set zones = analysis["三庭比例"] %}
{% for zone_name, ratio_key, fill_class in [
    ("上庭", "upper_ratio", zones.upper_ratio),
    ("中庭", "middle_ratio", zones.middle_ratio),
    ("下庭", "lower_ratio", zones.lower_ratio)
] %}
{% set ratio_val = zones[ratio_key] %}
{% set fill_class2 = "ideal" if (ratio_val|float > 0.28 and ratio_val|float < 0.38) else ("over" if ratio_val|float >= 0.38 else "low") %}
<div class="proportion-row">
  <span class="proportion-label">{{ zone_name }}</span>
  <div class="bar-bg">
    <div class="bar-fill {{ fill_class2 }}" style="width:{{ (ratio_val|float * 100)|round(1) }}%;"></div>
  </div>
  <span class="proportion-val">{{ "%.1f"|format(ratio_val|float * 100) }}%</span>
</div>
{% endfor %}

<div style="margin-top:14px;font-size:0.88rem;color:#a0c0d8;">
  主導分區：<strong style="color:#e0c44a;">{{ zones.dominant }}</strong>
</div>


  </div>

  <!-- 五眼比例 -->

  <div class="section">
    <div class="section-title">👁️ 五眼比例分析</div>
    {% set fe = analysis["五眼比例"] %}
    <div class="five-eye-grid">
      <div class="stat-box">
        <div class="stat-label">臉部寬度</div>
        <div class="stat-value">{{ fe.face_width|round(0)|int }}px</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">平均眼寬</div>
        <div class="stat-value">{{ fe.eye_avg_width|round(0)|int }}px</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">兩眼距離</div>
        <div class="stat-value">{{ fe.inter_eye_distance|round(0)|int }}px</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">五眼比例</div>
        <div class="stat-value">{{ "%.2f"|format(fe.five_eye_ratio|float) }}</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">眼距比（理想≈1.0）</div>
        <div class="stat-value">{{ "%.2f"|format(fe.inter_ratio|float) }}</div>
      </div>
    </div>
  </div>

  <!-- 問題診斷 -->

  <div class="section">
    <div class="section-title">🔍 面部問題診斷</div>
    {% for item in scored_items %}
    <div style="margin-bottom:10px;">
      <div class="score-meter">
        <span style="min-width:70px;font-size:0.88rem;color:#c0d0e0;">{{ item.name }}</span>
        <div class="score-track" style="flex:1;">
          <div class="score-fill" style="width:{{ (item.score * 100)|round(1) }}%;
            background: {{ '#2ecc71' if item.score < 0.3 else ('#e0c44a' if item.score < 0.7 else '#e74c3c') }};"></div>
        </div>
        <span class="badge badge-{{ item.severity }}" style="min-width:40px;text-align:center;">{{ item.severity }}</span>
        <span style="min-width:40px;text-align:right;font-size:0.82rem;color:#7090a0;">{{ "%.0f"|format(item.score * 100) }}/100</span>
      </div>
    </div>
    {% endfor %}
  </div>

  <!-- 治療建議 -->

  <div class="section">
    <div class="section-title">💉 治療建議</div>
    {% if recommendations %}
      {% for rec in recommendations %}
      <div class="problem-card">
        <div class="problem-header">
          <div class="problem-name">{{ rec.problem }}</div>
          <span class="badge badge-{{ rec.severity }}">{{ rec.severity }}</span>
        </div>
        <div class="problem-desc">{{ rec.description }}</div>
        <table class="rec-table">
          <thead>
            <tr>
              <th>建議產品</th>
              <th>類別</th>
              <th>建議劑量</th>
              <th>注射層次</th>
              <th>注射方式</th>
              <th>預估效果</th>
            </tr>
          </thead>
          <tbody>
            {% for prod in rec.recommendations %}
            <tr>
              <td class="product-name">{{ prod.product_name }}</td>
              <td>{{ prod.category }}</td>
              <td>{{ prod.dose_mid }}</td>
              <td>{{ prod.layer }}</td>
              <td>{{ prod.method }}</td>
              <td style="font-size:0.78rem;">{{ prod.effect }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      {% endfor %}
    {% else %}
      <p style="color:#7090a0;">目前未偵測到需要治療的明顯問題，請保持良好的保養習慣。</p>
    {% endif %}
  </div>

  <!-- 面相學 -->

  <div class="section">
    <div class="section-title">🔮 面相學評估</div>
    <p style="font-size:0.82rem;color:#7090a0;margin-bottom:16px;">以下面相評語基於傳統面相學，僅供娛樂參考。</p>
    <div class="physio-cards">
      {% for reading in physiognomy %}
      <div class="physio-card">
        <div class="physio-icon">{{ reading.icon }}</div>
        <div class="physio-aspect">{{ reading.aspect }}</div>
        <div class="physio-text">{{ reading.reading }}</div>
      </div>
      {% endfor %}
    </div>
  </div>

  <!-- 免責聲明 -->

  <div class="disclaimer">
    ⚠️ <strong>免責聲明</strong>：本系統為輔助參考工具，分析結果基於 AI 演算法，
    存在一定誤差。實際診斷與治療計劃請務必諮詢具有合法執照的專業醫師。
    本報告不構成醫療建議，產品劑量僅為參考範圍，實際用量依個人狀況而定。
  </div>

</div>
</body>
</html>
"""

def generate_html_report(
img_front_pil, img_left45_pil, img_left90_pil,
img_annotated_cv2,
analysis, recommendations, physiognomy
) -> str:
“”“使用 Jinja2 渲染 HTML 報告”””
from jinja2 import Environment


env = Environment()
template = env.from_string(HTML_TEMPLATE)

# 轉 base64
b64_front = image_to_base64(resize_image(img_front_pil, 600))
b64_left45 = image_to_base64(resize_image(img_left45_pil, 600))
b64_left90 = image_to_base64(resize_image(img_left90_pil, 600))
b64_ann = image_to_base64(cv2_to_pil(img_annotated_cv2))

# 整理評分列表（過濾掉非評分項目）
scored_items = []
for key, val in analysis.items():
    if isinstance(val, dict) and "score" in val:
        scored_items.append({
            "name": key,
            "score": val["score"],
            "severity": val.get("severity", "輕度"),
        })
scored_items.sort(key=lambda x: x["score"], reverse=True)

html = template.render(
    report_date=datetime.now().strftime("%Y年%m月%d日 %H:%M"),
    img_front=b64_front,
    img_left45=b64_left45,
    img_left90=b64_left90,
    img_annotated=b64_ann,
    analysis=analysis,
    scored_items=scored_items,
    recommendations=recommendations,
    physiognomy=physiognomy,
)
return html


# ─────────────────────────────────────────

# 9. Streamlit 主介面

# ─────────────────────────────────────────

def main():
# ── CSS ──────────────────────────────────────────────────────────────────
st.markdown(”””
<style>
.stApp { background: #0d1117; }
.upload-label { font-size: 1rem; font-weight: 600; color: #e0c44a; margin-bottom: 8px; }
.angle-ok   { color: #2ecc71; font-size: 0.85rem; }
.angle-fail { color: #e74c3c; font-size: 0.85rem; }
h1, h2, h3 { color: #e0c44a !important; }
.stButton > button {
background: linear-gradient(135deg, #e0c44a, #c0a030) !important;
color: #000 !important; font-weight: 700 !important;
border-radius: 8px !important; border: none !important;
padding: 12px 30px !important; font-size: 1rem !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
</style>
“””, unsafe_allow_html=True)


st.title("🏥 AI 智能醫美面診輔助系統")
st.caption("上傳三張臉部照片，AI 自動分析並生成治療建議報告")

# ── 上傳區 ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📸 第一步：上傳三張臉部照片")
st.info("請依序上傳：**正面**、**左側 45°**、**左側 90°** 三張清晰臉部照片（JPG / PNG）。")

col1, col2, col3 = st.columns(3)
uploads = {}

with col1:
    st.markdown('<div class="upload-label">① 正面（0°）</div>', unsafe_allow_html=True)
    f_front = st.file_uploader("", type=["jpg", "jpeg", "png"], key="front", label_visibility="collapsed")
    if f_front:
        uploads["front"] = Image.open(f_front).convert("RGB")
        st.image(uploads["front"], use_container_width=True, caption="正面")

with col2:
    st.markdown('<div class="upload-label">② 左側 45°</div>', unsafe_allow_html=True)
    f_45 = st.file_uploader("", type=["jpg", "jpeg", "png"], key="left45", label_visibility="collapsed")
    if f_45:
        uploads["left45"] = Image.open(f_45).convert("RGB")
        st.image(uploads["left45"], use_container_width=True, caption="左側 45°")

with col3:
    st.markdown('<div class="upload-label">③ 左側 90°</div>', unsafe_allow_html=True)
    f_90 = st.file_uploader("", type=["jpg", "jpeg", "png"], key="left90", label_visibility="collapsed")
    if f_90:
        uploads["left90"] = Image.open(f_90).convert("RGB")
        st.image(uploads["left90"], use_container_width=True, caption="左側 90°")

# ── 角度驗證 ──────────────────────────────────────────────────────────────
# 每張照片的期望 yaw 角（度）及容忍誤差
ANGLE_CONFIG = {
    "front":  {"expected": 0,   "tolerance": 15, "label": "正面"},
    "left45": {"expected": -35, "tolerance": 18, "label": "左側 45°"},
    "left90": {"expected": -70, "tolerance": 20, "label": "左側 90°"},
}

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
                st.markdown(f'<div class="angle-fail">❌ {cfg["label"]}：無法偵測人臉</div>', unsafe_allow_html=True)
                angle_ok[key] = False
            else:
                yaw = estimate_yaw(lm)
                diff = abs(yaw - cfg["expected"])
                ok = diff <= cfg["tolerance"]
                angle_ok[key] = ok
                icon = "✅" if ok else "⚠️"
                cls = "angle-ok" if ok else "angle-fail"
                msg = f'{icon} {cfg["label"]}：Yaw≈{yaw:.1f}°（期望{cfg["expected"]}°±{cfg["tolerance"]}°）'
                st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)
                if not ok:
                    st.warning(f'{cfg["label"]} 角度偏差過大，建議重新拍攝。')

# ── 分析按鈕 ──────────────────────────────────────────────────────────────
st.markdown("---")
all_uploaded = all(k in uploads for k in ("front", "left45", "left90"))

if all_uploaded:
    if st.button("🚀 開始 AI 面診分析", use_container_width=True):
        with st.spinner("AI 正在分析中，請稍候..."):
            try:
                # 使用正面照進行主要分析
                front_bgr = pil_to_cv2(resize_image(uploads["front"], 800))
                lm_front, _ = get_landmarks(front_bgr)

                if lm_front is None:
                    st.error("❌ 無法偵測正面照片的人臉，請上傳更清晰的照片。")
                    return

                # 分析
                analysis = analyze_face(lm_front, front_bgr)

                # 繪製標註圖
                img_annotated = draw_annotations(front_bgr, lm_front, analysis)

                # 治療建議
                recommendations = generate_recommendations(analysis)

                # 面相學
                physio = physiognomy_reading(analysis)

                # ── 展示結果 ────────────────────────────────────────────────────
                st.success("✅ 分析完成！")

                # 標註圖
                st.subheader("🎯 三庭五眼標註")
                ann_pil = cv2_to_pil(img_annotated)
                st.image(ann_pil, use_container_width=True, clamp=True)

                # 三庭比例
                st.subheader("📐 三庭比例")
                zones = analysis["三庭比例"]
                zc1, zc2, zc3 = st.columns(3)
                zc1.metric("上庭", f'{zones["upper_ratio"]:.1%}', delta=f'{(zones["upper_ratio"]-0.333)/0.333:+.1%}')
                zc2.metric("中庭", f'{zones["middle_ratio"]:.1%}', delta=f'{(zones["middle_ratio"]-0.333)/0.333:+.1%}')
                zc3.metric("下庭", f'{zones["lower_ratio"]:.1%}', delta=f'{(zones["lower_ratio"]-0.333)/0.333:+.1%}')
                st.info(f'主導分區：**{zones["dominant"]}**')

                # 問題評分
                st.subheader("🔍 問題診斷")
                scored = [(k, v) for k, v in analysis.items() if isinstance(v, dict) and "score" in v]
                scored.sort(key=lambda x: x[1]["score"], reverse=True)
                for name, data in scored:
                    sev = data["severity"]
                    score = data["score"]
                    color = "#2ecc71" if sev == "輕度" else "#e0c44a" if sev == "中度" else "#e74c3c"
                    st.markdown(f"**{name}** &nbsp; <span style='color:{color}'>{sev}</span> &nbsp; `{score:.2f}`", unsafe_allow_html=True)
                    st.progress(score)

                # 治療建議
                st.subheader("💉 治療建議")
                if recommendations:
                    for rec in recommendations:
                        with st.expander(f"🔸 {rec['problem']} — {rec['severity']}（分數：{rec['score']:.2f}）"):
                            for prod in rec["recommendations"]:
                                st.markdown(f"""


## **產品**：{prod[‘product_name’]}  
**類別**：{prod[‘category’]}  
**建議劑量**：{prod[‘dose_mid’]}  
**注射層次**：{prod[‘layer’]}  
**注射方式**：{prod[‘method’]}  
**預估效果**：{prod[‘effect’]}

“””)
else:
st.success(“目前未偵測到明顯治療需求，維持良好保養即可。”)


                # 面相學
                st.subheader("🔮 面相學評估")
                p_cols = st.columns(2)
                for i, reading in enumerate(physio):
                    with p_cols[i % 2]:
                        st.markdown(f"""


<div style="background:#1a1a2e;border-radius:10px;padding:14px;margin-bottom:12px;">
<div style="font-size:1.4rem">{reading['icon']}</div>
<div style="color:#e0c44a;font-weight:600;margin:6px 0;">{reading['aspect']}</div>
<div style="font-size:0.85rem;color:#a0b4c8;line-height:1.7;">{reading['reading']}</div>
</div>
""", unsafe_allow_html=True)

```
                # 生成報告
                st.subheader("📄 下載 HTML 報告")
                html_report = generate_html_report(
                    uploads["front"], uploads["left45"], uploads["left90"],
                    img_annotated, analysis, recommendations, physio
                )
                st.download_button(
                    label="⬇️ 下載完整面診報告（HTML）",
                    data=html_report.encode("utf-8"),
                    file_name=f"face_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    use_container_width=True,
                )

                # 免責聲明
                st.markdown("---")
                st.warning(
                    "⚠️ **免責聲明**：本系統為輔助參考工具，分析結果僅供參考，不構成醫療建議。"
                    "實際診斷與治療計劃請務必諮詢具有合法執照的專業醫師。"
                    "產品劑量僅為參考範圍，實際用量依個人狀況及醫師評估而定。"
                )

            except Exception as e:
                st.error(f"❌ 分析過程發生錯誤：{e}")
                st.exception(e)
else:
    missing = [ANGLE_CONFIG[k]["label"] for k in ("front", "left45", "left90") if k not in uploads]
    if missing:
        st.info(f"請上傳以下照片後開始分析：{', '.join(missing)}")


if **name** == “**main**”:
main()
