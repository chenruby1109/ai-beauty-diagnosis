import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import base64
import io
import math
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ExifTags

st.set_page_config(page_title="AI Beauty System", page_icon="H",
                   layout="wide", initial_sidebar_state="collapsed")

# ─── Products ────────────────────────────────────────────────────────────────
PRODUCT_ZH = {
    "botox_forehead":  ("肉毒毒素（抬頭紋）",  "奇蹟/天使肉毒", "額肌（肌肉層）",     "多點注射間距1.5cm",   "3-7天見效，4-6個月"),
    "botox_glabella":  ("肉毒毒素（眉間紋）",  "奇蹟/天使肉毒", "皺眉肌",             "5點標準注射法",       "3-7天改善，4-6個月"),
    "botox_crowsfeet": ("肉毒毒素（魚尾紋）",  "奇蹟/天使肉毒", "眼輪匝肌",           "眼外角扇形多點",      "5-7天平滑，3-5個月"),
    "botox_jawline":   ("肉毒毒素（下頜緣）",  "奇蹟/天使肉毒", "頸闊肌",             "沿頸闊肌線性",        "輪廓提升，4-6個月"),
    "botox_calf":      ("肉毒毒素（腓腸肌）",  "奇蹟/天使肉毒", "腓腸肌內側頭",       "多點格狀，每點5-10U", "4-8週改善，6-9個月"),
    "ha_cheek":        ("玻尿酸（蘋果肌）",    "喬雅登VOLUMA",  "骨膜上層/深層脂肪",  "扇形/線性推注",       "維持18-24個月"),
    "ha_chin":         ("玻尿酸（下巴）",      "喬雅登VOLUMA",  "骨膜上層",           "單點或扇形",          "維持12-18個月"),
    "ha_nasolabial":   ("玻尿酸（法令紋）",    "喬雅登VOLUMA",  "深層皮下/骨膜上層",  "逆行線性+扇形",       "減少60-80%，12-18個月"),
    "ha_jawline":      ("玻尿酸（下頜輪廓）",  "喬雅登VOLUX",   "骨膜上層",           "線性沿下頜骨緣",      "維持18-24個月"),
    "ha_nasolabial2":  ("玻尿酸（法令紋II）",  "喬雅登VOLIFT",  "真皮深層至皮下",     "逆行線性+蕨葉技術",   "維持12-15個月"),
    "ha_tear":         ("玻尿酸（淚溝）",      "喬雅登VOLBELLA","眶隔前脂肪/骨膜上層","微量多點/線性",       "維持9-12個月"),
    "ha_skin":         ("玻尿酸（全臉保濕）",  "喬雅登VOLITE",  "真皮中層",           "多點注射/水光槍",     "維持6-9個月"),
    "sculptra":        ("舒顏萃Sculptra",      "Sculptra",      "真皮深層至皮下",     "扇形大範圍+按摩",     "2-3月顯現，18-24個月"),
    "thread":          ("鳳凰埋線",            "PDO大V線",      "SMAS筋膜/深層皮下",  "逆行進針雙向倒鉤",    "即時提拉，12-18個月"),
    "pn":              ("麗珠蘭PN 1%",         "Rejuran PN",    "真皮淺中層",         "水光槍/多點注射",     "膚色提亮，3-4療程"),
    "picosecond":      ("皮秒雷射",            "皮秒雷射",      "表皮至真皮層",       "全臉掃描依斑點加強",  "膚色均勻，每4-6週"),
    "ultherapy":       ("音波拉提Ultherapy",   "Ultherapy",     "SMAS筋膜+真皮",      "線性掃描分層",        "3-6月顯效，12-18個月"),
    "thermage":        ("電波拉提Thermage",    "Thermage FLX",  "真皮深層至皮下",     "全臉均勻掃描",        "即時緊緻，12-24個月"),
}

DOSE_TABLE = {
    "botox_forehead":  {"1":"6U","2":"10U","3":"15U","4":"22U","5":"32U"},
    "botox_glabella":  {"1":"6U","2":"10U","3":"15U","4":"22U","5":"32U"},
    "botox_crowsfeet": {"1":"4U/s","2":"6U/s","3":"9U/s","4":"13U/s","5":"17U/s"},
    "botox_jawline":   {"1":"8U","2":"12U","3":"18U","4":"28U","5":"40U"},
    "botox_calf":      {"1":"40U/s","2":"55U/s","3":"70U/s","4":"90U/s","5":"125U/s"},
    "ha_cheek":        {"1":"0.3ml/s","2":"0.5ml/s","3":"0.8ml/s","4":"1.2ml/s","5":"1.8ml/s"},
    "ha_chin":         {"1":"0.3ml","2":"0.5ml","3":"0.8ml","4":"1.2ml","5":"1.8ml"},
    "ha_nasolabial":   {"1":"0.5ml/s","2":"0.75ml/s","3":"1.0ml/s","4":"1.4ml/s","5":"1.8ml/s"},
    "ha_jawline":      {"1":"0.8ml","2":"1.2ml","3":"1.8ml","4":"2.5ml","5":"3.5ml"},
    "ha_nasolabial2":  {"1":"0.5ml/s","2":"0.75ml/s","3":"1.0ml/s","4":"1.4ml/s","5":"1.8ml/s"},
    "ha_tear":         {"1":"0.2ml/s","2":"0.3ml/s","3":"0.5ml/s","4":"0.8ml/s","5":"1.2ml/s"},
    "ha_skin":         {"1":"1ml","2":"1.5ml","3":"2ml","4":"2.8ml","5":"3.5ml"},
    "sculptra":        {"1":"0.5v","2":"1v","3":"1.5v","4":"2v","5":"3v"},
    "thread":          {"1":"2根/s","2":"3根/s","3":"5根/s","4":"8根/s","5":"13根/s"},
    "pn":              {"1":"0.5ml","2":"1ml","3":"1.5ml","4":"2ml","5":"2.5ml"},
    "picosecond":      {"1":"1次","2":"2次","3":"3次","4":"5次","5":"6次"},
    "ultherapy":       {"1":"150發","2":"250發","3":"350發","4":"500發","5":"800發"},
    "thermage":        {"1":"600發","2":"800發","3":"1000發","4":"1200發","5":"1500發"},
}

PROBLEM_ZH = {
    "forehead":   "抬頭紋",
    "glabella":   "眉間紋",
    "crowsfeet":  "魚尾紋",
    "tear":       "淚溝",
    "nasolabial": "法令紋",
    "cheek":      "蘋果肌",
    "jawline":    "下頜緣輪廓",
    "chin":       "下巴比例",
    "skin":       "皮膚質地",
    "symmetry":   "臉部對稱性",
}

# Landmark indices for annotation arrows (face landmark)
PROBLEM_LM = {
    "forehead":   [10],
    "glabella":   [8],
    "crowsfeet":  [33, 263],
    "tear":       [159, 386],
    "nasolabial": [49, 279],
    "cheek":      [117, 346],
    "jawline":    [172, 397],
    "chin":       [152],
    "skin":       [50],
    "symmetry":   [1],
}

PROBLEM_TO_PRODUCTS = {
    "forehead":   [("botox_forehead", True), ("pn", False)],
    "glabella":   [("botox_glabella", True), ("picosecond", False)],
    "crowsfeet":  [("botox_crowsfeet", True), ("thermage", False)],
    "tear":       [("ha_tear", True), ("pn", False)],
    "nasolabial": [("ha_nasolabial", True), ("ha_nasolabial2", True), ("sculptra", False), ("thread", False)],
    "cheek":      [("ha_cheek", True), ("sculptra", False)],
    "jawline":    [("ha_jawline", True), ("botox_jawline", True), ("ultherapy", False)],
    "chin":       [("ha_chin", True)],
    "skin":       [("ha_skin", True), ("picosecond", False), ("pn", False)],
    "symmetry":   [("botox_glabella", True), ("ha_cheek", False)],
}

# 5-level grading
LEVELS = [
    (0.10, "1", "極輕微", "#2ecc71"),
    (0.28, "2", "輕微",   "#27ae60"),
    (0.48, "3", "中度",   "#f39c12"),
    (0.68, "4", "明顯",   "#e67e22"),
    (1.01, "5", "重度",   "#e74c3c"),
]

SCORE_THRESHOLD = 0.10

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose


def grade(score):
    for threshold, level, name, color in LEVELS:
        if score < threshold:
            return level, name, color
    return "5", "重度", "#e74c3c"


# ─── Image helpers ────────────────────────────────────────────────────────────
def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_b64(img, fmt="JPEG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def resize_image(img, max_size=800):
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return img


def auto_rotate_face(pil_img):
    """Try 4 rotations, return the one where face is detected upright."""
    best = None
    best_score = -1
    rotations = [0, 90, 180, 270]
    for angle in rotations:
        if angle == 0:
            candidate = pil_img.copy()
        else:
            candidate = pil_img.rotate(angle, expand=True)
        img_bgr = pil_to_cv2(resize_image(candidate, 640))
        with mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.4,
        ) as fm:
            res = fm.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            continue
        h2, w2 = img_bgr.shape[:2]
        lm = res.multi_face_landmarks[0].landmark
        nose_y = lm[1].y
        chin_y = lm[152].y
        forehead_y = lm[10].y
        left_eye_y = lm[33].y
        right_eye_y = lm[263].y
        # Face is upright: forehead < nose < chin (y increases downward)
        # Eyes should be roughly same height
        upright_score = (
            (forehead_y < nose_y) * 2 +
            (nose_y < chin_y) * 2 +
            (abs(left_eye_y - right_eye_y) < 0.08) * 1
        )
        # Detection confidence
        conf = res.multi_face_landmarks[0].landmark[1].visibility if hasattr(res.multi_face_landmarks[0].landmark[1], 'visibility') else 1.0
        total = upright_score + conf
        if total > best_score:
            best_score = total
            best = candidate
    return best if best is not None else pil_img


# ─── Face detection ───────────────────────────────────────────────────────────
def get_landmarks(img_bgr):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.4,
    ) as fm:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = fm.process(rgb)
    if not results.multi_face_landmarks:
        return None, None
    h, w = img_bgr.shape[:2]
    lm = results.multi_face_landmarks[0].landmark
    landmarks = np.array([[l.x * w, l.y * h, l.z * w] for l in lm])
    return landmarks, results


def estimate_yaw_frontal(lm):
    """Yaw estimation - only reliable for near-frontal faces"""
    nose, le, re = lm[1], lm[33], lm[263]
    ld = abs(nose[0] - le[0])
    rd = abs(nose[0] - re[0])
    total = ld + rd
    if total < 1e-6:
        return 0.0
    return (ld / total - 0.5) * 180.0


# ─── Scoring (re-calibrated) ──────────────────────────────────────────────────
def norm(v, lo, hi):
    if hi - lo < 1e-6:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def safe_std(lm, idx):
    """z-std in pixel units (lm[:,2] = l.z * image_width)"""
    return float(np.std(lm[idx, 2]))


def analyze_face(lm, img_bgr):
    h, w = img_bgr.shape[:2]
    R = {}

    # Three zones
    hy = lm[10, 1]; gy = lm[8, 1]; ny = lm[94, 1]; cy = lm[152, 1]
    upper  = abs(gy - hy)
    middle = abs(ny - gy)
    lower  = abs(cy - ny)
    total  = upper + middle + lower + 1e-6
    R["zones"] = {
        "upper_ratio":  round(upper / total, 4),
        "middle_ratio": round(middle / total, 4),
        "lower_ratio":  round(lower / total, 4),
        "dominant": (
            "upper"  if upper  > middle and upper  > lower else
            "middle" if middle > upper  and middle > lower else "lower"
        ),
    }

    # Five eyes
    fw = abs(lm[454, 0] - lm[234, 0]) + 1e-6
    lew = abs(lm[133, 0] - lm[33, 0])
    rew = abs(lm[263, 0] - lm[362, 0])
    inter = abs(lm[362, 0] - lm[133, 0])
    eavg  = (lew + rew) / 2.0 + 1e-6
    R["five_eyes"] = {
        "face_width":     round(fw, 1),
        "eye_avg":        round(eavg, 1),
        "five_eye_ratio": round(fw / (5 * eavg), 3),
        "inter_ratio":    round(inter / eavg, 3),
    }

    # Symmetry (2D geometry only - reliable)
    nx = lm[1, 0]
    sym_pairs = [(33,263),(133,362),(234,454),(61,291),(50,280),(149,378),(58,288),(172,397),(70,300)]
    asym = sum(
        abs(abs(lm[li,0]-nx) - abs(lm[ri,0]-nx)) /
        max(abs(lm[li,0]-nx), abs(lm[ri,0]-nx), 1e-6)
        for li, ri in sym_pairs
    ) / len(sym_pairs)
    R["symmetry"] = {"score": round(norm(asym, 0.03, 0.22), 4),
                     "severity": "", "raw": round(asym, 4),
                     "description": "左右不對稱程度"}

    # Tear trough (z-depth, recalibrated for pixel units)
    tz = np.mean(lm[[159,145,386,374], 2])
    cz = np.mean(lm[[50,280,205,425], 2])
    tear_raw = abs(tz - cz)  # in pixel units, expect 0-40
    R["tear"] = {"score": round(norm(tear_raw, 1.0, 18.0), 4),
                 "severity": "", "raw": round(tear_raw, 2),
                 "description": "淚溝凹陷深度"}

    # Nasolabial folds (2D length + z-depth)
    ll = np.linalg.norm(lm[49,:2] - lm[61,:2])
    rl = np.linalg.norm(lm[279,:2] - lm[291,:2])
    nl_len = (ll + rl) / 2.0
    nl_z   = abs(np.mean(lm[[49,279], 2]) - np.mean(lm[[61,291], 2]))
    len_n  = norm(nl_len, fw*0.08, fw*0.22)
    dep_n  = norm(nl_z, 1.0, 18.0)
    R["nasolabial"] = {"score": round(0.5*len_n + 0.5*dep_n, 4),
                       "severity": "", "raw_len": round(nl_len, 1),
                       "description": "法令紋深度與長度"}

    # Apple cheek (z-depth)
    cbz = np.mean(lm[[117,346,118,347], 2])
    ckz = np.mean(lm[[50,280], 2])
    apple_raw = abs(cbz - ckz)
    # Low z-diff = flat cheek = needs filling
    apple_n = 1.0 - norm(apple_raw, 2.0, 20.0)
    R["cheek"] = {"score": round(apple_n, 4),
                  "severity": "", "raw": round(apple_raw, 2),
                  "description": "蘋果肌飽滿度（越高=越需填充）"}

    # Jawline (2D geometry)
    jaw_idx = [172,171,170,169,168,136,135,134,133,58,288,172,397]
    jaw_pts = lm[jaw_idx, :2]
    jaw_std = float(np.std(jaw_pts[:, 1]))
    R["jawline"] = {"score": round(norm(jaw_std, 1.5, 18.0), 4),
                    "severity": "", "raw": round(jaw_std, 2),
                    "description": "下頜緣輪廓清晰度"}

    # Forehead lines (z-std, recalibrated: in pixel units 0-40)
    fh_std = safe_std(lm, [55,107,66,105,65,52,53,46,124,156,70,63,108,337,336,296])
    R["forehead"] = {"score": round(norm(fh_std, 3.0, 35.0), 4),
                     "severity": "", "raw": round(fh_std, 2),
                     "description": "額頭橫紋深度"}

    # Glabella (z-std)
    gl_std = safe_std(lm, [168,6,197,195,5,4,8,9,107,336,223,443,55,285])
    R["glabella"] = {"score": round(norm(gl_std, 3.0, 35.0), 4),
                     "severity": "", "raw": round(gl_std, 2),
                     "description": "眉間川字紋深度"}

    # Crow's feet (z-std)
    cr_std = safe_std(lm, [33,133,246,161,160,159,263,362,466,388,387,386,130,359,226,446])
    R["crowsfeet"] = {"score": round(norm(cr_std, 3.0, 35.0), 4),
                      "severity": "", "raw": round(cr_std, 2),
                      "description": "眼外角魚尾紋深度"}

    # Chin ratio (2D geometry)
    chin_w = abs(lm[172,0] - lm[397,0]) + 1e-6
    chin_l = abs(lm[152,1] - ny)
    chin_ratio = chin_l / chin_w
    R["chin"] = {"score": round(norm(chin_ratio, 0.3, 0.85), 4),
                 "severity": "", "ratio": round(chin_ratio, 3),
                 "description": "下巴長寬比（偏低=過短）"}

    # Skin texture (2D: Laplacian of face region)
    face_roi = extract_face_roi(img_bgr, lm)
    if face_roi is not None and face_roi.size > 0:
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray_roi, cv2.CV_64F)
        texture_var = float(np.var(lap))
        # High Laplacian variance = rough skin
        # Young smooth skin: ~100-500, textured skin: 500-3000
        skin_n = norm(texture_var, 200.0, 3000.0)
    else:
        skin_n = 0.2
    R["skin"] = {"score": round(skin_n, 4),
                 "severity": "", "description": "皮膚粗糙度/質地"}

    return R


def extract_face_roi(img_bgr, lm):
    """Extract cheek region for texture analysis"""
    try:
        cheek_pts = lm[[50, 280, 205, 425, 117, 346, 147, 376], :2].astype(int)
        x1 = max(0, int(np.min(cheek_pts[:,0])))
        y1 = max(0, int(np.min(cheek_pts[:,1])))
        x2 = min(img_bgr.shape[1], int(np.max(cheek_pts[:,0])))
        y2 = min(img_bgr.shape[0], int(np.max(cheek_pts[:,1])))
        if x2 > x1 and y2 > y1:
            return img_bgr[y1:y2, x1:x2]
    except Exception:
        pass
    return None


# ─── Recs ────────────────────────────────────────────────────────────────────
def generate_recs(analysis):
    recs = []
    for pk, pname in PROBLEM_ZH.items():
        data = analysis.get(pk, {})
        if not data or "score" not in data:
            continue
        score = data["score"]
        if score < SCORE_THRESHOLD:
            continue
        lv, lv_name, lv_color = grade(score)
        if pk not in PROBLEM_TO_PRODUCTS:
            continue
        primary, alts = [], []
        for prod_key, is_primary in PROBLEM_TO_PRODUCTS[pk]:
            if prod_key not in PRODUCT_ZH:
                continue
            name, brand, layer, method, effect = PRODUCT_ZH[prod_key]
            dose = DOSE_TABLE[prod_key].get(lv, "依醫師評估")
            info = {"key": prod_key, "name": name, "brand": brand,
                    "dose": dose, "layer": layer, "method": method, "effect": effect}
            if is_primary:
                primary.append(info)
            else:
                alts.append(info)
        recs.append({"key": pk, "name": pname, "score": score,
                     "level": lv, "level_name": lv_name, "level_color": lv_color,
                     "primary": primary, "alternatives": alts,
                     "description": data.get("description", "")})
    return recs


# ─── Annotations ─────────────────────────────────────────────────────────────
def draw_basic_annotations(img_bgr, lm, analysis):
    img = img_bgr.copy()
    h, w = img.shape[:2]
    COLOR = (0, 220, 255)
    TEXT  = (255, 255, 255)
    zones = analysis.get("zones", {})
    ys = {
        "top": int(lm[10, 1]),
        "m1":  int(lm[8, 1]),
        "m2":  int(lm[94, 1]),
        "bot": int(lm[152, 1]),
    }
    for k, y in ys.items():
        cv2.line(img, (0, y), (w, y), COLOR, 1)
    for x in [lm[234,0], lm[33,0], lm[133,0], lm[362,0], lm[263,0], lm[454,0]]:
        cv2.line(img, (int(x), 0), (int(x), h), (200, 200, 0), 1)
    for idx in [10, 8, 94, 152, 33, 133, 362, 263, 234, 454, 1]:
        cv2.circle(img, (int(lm[idx,0]), int(lm[idx,1])), 3, (0, 100, 255), -1)
    zone_vals = [
        (ys["top"], ys["m1"], "U:{:.1%}".format(zones.get("upper_ratio", 0))),
        (ys["m1"],  ys["m2"], "M:{:.1%}".format(zones.get("middle_ratio", 0))),
        (ys["m2"],  ys["bot"],"L:{:.1%}".format(zones.get("lower_ratio", 0))),
    ]
    for y1, y2, text in zone_vals:
        cv2.putText(img, text, (w - 95, (y1+y2)//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR, 1, cv2.LINE_AA)
    return img


def draw_treatment_map(img_bgr, lm, recs):
    """Draw numbered arrows. All text is ASCII/numbers to avoid font issues."""
    h, w = img_bgr.shape[:2]
    pil = cv2_to_pil(img_bgr.copy())
    draw = ImageDraw.Draw(pil)

    try:
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        font_reg  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font_bold = ImageFont.load_default()
        font_reg  = font_bold

    top_recs = sorted([r for r in recs if r["score"] >= SCORE_THRESHOLD],
                      key=lambda x: x["score"], reverse=True)[:8]

    placed = []

    for idx, rec in enumerate(top_recs):
        lm_indices = PROBLEM_LM.get(rec["key"], [1])
        pts = lm[lm_indices, :2]
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))

        # Red circle on face
        draw.ellipse([(cx-7, cy-7), (cx+7, cy+7)],
                     fill=(200, 40, 40), outline=(255, 220, 0), width=2)
        draw.text((cx-4, cy-6), str(idx+1), fill=(255,255,255), font=font_reg)

        # Label position
        if cx < w * 0.5:
            lx = 4
            ax = cx - 8
        else:
            lx = w - 182
            ax = cx + 8

        ly = cy - 22
        # Avoid overlap
        for (px, py, pw, ph) in placed:
            if abs(ly - py) < ph + 5:
                ly = py + ph + 5
        ly = max(4, min(h - 60, ly))

        bw, bh = 176, 52
        # Semi-transparent background
        overlay = Image.new("RGBA", pil.size, (0,0,0,0))
        ov_draw = ImageDraw.Draw(overlay)
        ov_draw.rectangle([(lx, ly), (lx+bw, ly+bh)], fill=(8, 12, 30, 210))
        pil = pil.convert("RGBA")
        pil = Image.alpha_composite(pil, overlay)
        pil = pil.convert("RGB")
        draw = ImageDraw.Draw(pil)

        # Border color by level
        border_color = rec.get("level_color", "#e74c3c")
        bc = tuple(int(border_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        draw.rectangle([(lx, ly), (lx+bw, ly+bh)], outline=bc, width=2)

        # Number badge
        draw.ellipse([(lx+3, ly+3), (lx+19, ly+19)], fill=(200, 40, 40))
        draw.text((lx+7, ly+4), str(idx+1), fill=(255,255,255), font=font_reg)

        # Level + score bar
        lv_pct = int(rec["score"] * (bw - 24))
        draw.rectangle([(lx+22, ly+5), (lx+22+lv_pct, ly+12)], fill=bc)

        # Line 1: short problem ID
        prob_short = {
            "forehead": "Forehead", "glabella": "Glabella", "crowsfeet": "Crow's ft",
            "tear": "Tear trgh", "nasolabial": "Nasolab.", "cheek": "Cheek",
            "jawline": "Jawline", "chin": "Chin", "skin": "Skin", "symmetry": "Symmetry",
        }.get(rec["key"], rec["key"])
        draw.text((lx+22, ly+14), "[{}] {} Lv{}".format(idx+1, prob_short, rec["level"]),
                  fill=(255, 215, 0), font=font_bold)

        # Line 2: dose if primary exists
        if rec["primary"]:
            prod = rec["primary"][0]
            draw.text((lx+22, ly+32), "Dose: {}  [Primary]".format(prod["dose"]),
                      fill=(150, 240, 150), font=font_reg)
        else:
            draw.text((lx+22, ly+32), "Score: {:.2f}".format(rec["score"]),
                      fill=(180, 180, 180), font=font_reg)

        placed.append((lx, ly, bw, bh))

        # Arrow
        ay = ly + bh // 2
        if cx < w * 0.5:
            arrow_end = (lx, ay)
        else:
            arrow_end = (lx + bw, ay)
        draw.line([(ax, cy), arrow_end], fill=(0, 220, 255), width=2)
        # Arrowhead at face point
        angle = math.atan2(cy - ay, cx - arrow_end[0])
        for da in [0.45, -0.45]:
            ex = int(ax - 14 * math.cos(angle + da))
            ey = int(cy - 14 * math.sin(angle + da))
            draw.line([(ax, cy), (ex, ey)], fill=(0, 220, 255), width=2)

    # Legend strip at bottom
    leg_y = h - 22
    draw.rectangle([(0, leg_y-2), (w, h)], fill=(8, 12, 30))
    names_map = {
        "forehead": "抬頭紋", "glabella": "眉間紋", "crowsfeet": "魚尾紋",
        "tear": "淚溝", "nasolabial": "法令紋", "cheek": "蘋果肌",
        "jawline": "下頜緣", "chin": "下巴", "skin": "皮膚", "symmetry": "對稱性",
    }
    for i, rec in enumerate(top_recs):
        x_pos = 5 + i * (w // len(top_recs))
        # Use ASCII number label
        label = "[{}]".format(i+1)
        draw.text((x_pos, leg_y), label, fill=(255, 200, 50), font=font_reg)

    return pil


# ─── Physiognomy ─────────────────────────────────────────────────────────────
def physio_analysis(analysis):
    readings = []
    zones = analysis.get("zones", {})
    dominant = zones.get("dominant", "")
    zone_data = [
        ("upper", "上庭", zones.get("upper_ratio", 0.333),
         "上庭寬闊，早年運佳，智慧過人，父母緣深厚",
         "若額頭有皺紋或過窄，可透過肉毒放鬆額肌改善抬頭紋，或以玻尿酸填充額頭弧度，讓上庭更飽滿圓潤。",
         ["肉毒毒素（抬頭紋）", "玻尿酸（全臉保濕）"]),
        ("middle", "中庭", zones.get("middle_ratio", 0.333),
         "中庭均衡，中年事業運旺，適合創業或管理職",
         "若鼻樑較低或法令紋明顯，可透過玻尿酸填充，讓中庭比例更完美，面相上增強事業運。",
         ["玻尿酸（法令紋）", "玻尿酸（法令紋II）"]),
        ("lower", "下庭", zones.get("lower_ratio", 0.333),
         "下庭豐厚，晚年運佳，福氣深厚，子孫有緣",
         "若下巴短或下頜緣不清晰，可透過玻尿酸墊下巴或埋線提拉，加強晚年財運與福氣。",
         ["玻尿酸（下巴）", "玻尿酸（下頜輪廓）"]),
    ]
    for zone_key, zone_name, ratio, good, improve, products in zone_data:
        readings.append({
            "aspect": zone_name + "分析", "zone": zone_name,
            "ratio": "{:.1%}".format(ratio),
            "reading": good if dominant == zone_key else zone_name + "比例偏低，建議調整",
            "improve": improve, "products": products,
        })
    inter_r = analysis.get("five_eyes", {}).get("inter_ratio", 1.0)
    if inter_r < 0.82:
        readings.append({
            "aspect": "眼距偏窄", "zone": "", "ratio": "{:.2f}".format(inter_r),
            "reading": "眼距較窄，個性敏銳，容易急躁",
            "improve": "可透過眉形調整或淚溝填充，視覺上拉寬眼距。",
            "products": ["玻尿酸（淚溝）"],
        })
    elif inter_r > 1.22:
        readings.append({
            "aspect": "眼距寬廣", "zone": "", "ratio": "{:.2f}".format(inter_r),
            "reading": "眼距寬廣，心胸寬大，人緣佳，適合公關外交",
            "improve": "眼距已相當理想，搭配皮秒改善眼周膚質即可。",
            "products": ["皮秒雷射"],
        })
    if analysis.get("nasolabial", {}).get("score", 0) > 0.40:
        readings.append({
            "aspect": "法令紋顯現", "zone": "", "ratio": "",
            "reading": "法令紋深象徵威嚴與領導力，主掌大局之相",
            "improve": "法令紋象徵威嚴，但視覺顯老。可透過玻尿酸填充或舒顏萃刺激膠原，保持威嚴同時更年輕。",
            "products": ["玻尿酸（法令紋）", "舒顏萃Sculptra"],
        })
    return readings


# ─── HTML Report ──────────────────────────────────────────────────────────────
def generate_html_report(analysis, recs, physio_data,
                         img_basic_pil, img_map_pil,
                         legend_html, timestamp):
    b64_basic = pil_to_b64(img_basic_pil)
    b64_map   = pil_to_b64(img_map_pil)
    zones = analysis.get("zones", {})
    fe    = analysis.get("five_eyes", {})
    dom_map = {"upper": "上庭", "middle": "中庭", "lower": "下庭"}
    dominant = dom_map.get(zones.get("dominant", ""), "")

    scored_html = ""
    all_scored = sorted(
        [(k, v) for k, v in analysis.items() if isinstance(v, dict) and "score" in v],
        key=lambda x: x[1]["score"], reverse=True,
    )
    for pk, data in all_scored:
        name  = PROBLEM_ZH.get(pk, pk)
        score = data["score"]
        lv, lv_name, lv_color = grade(score)
        pct = int(score * 100)
        desc = data.get("description", "")
        scored_html += """
        <div class="score-row">
          <div class="score-label">{name}</div>
          <div class="score-bar-wrap">
            <div class="score-bar" style="width:{pct}%;background:{color};"></div>
          </div>
          <span class="lv-badge" style="background:{color}20;color:{color};border-color:{color};">Lv{lv} {lv_name}</span>
          <span class="score-num">{score:.2f}</span>
          <span class="score-desc">{desc}</span>
        </div>""".format(name=name, pct=pct, color=lv_color,
                         lv=lv, lv_name=lv_name, score=score, desc=desc)

    recs_html = ""
    for rec in recs:
        lv_color = rec["level_color"]
        primary_cards = ""
        for prod in rec["primary"]:
            primary_cards += """
            <div class="prod-card" style="border-left:3px solid #2ecc71;background:#0a1f10;">
              <div class="prod-name">{name} <span class="tag-p">首選</span></div>
              <table class="pt">
                <tr><td>品牌</td><td>{brand}</td></tr>
                <tr><td>建議劑量</td><td><strong style="color:#ffd700">{dose}</strong></td></tr>
                <tr><td>注射層次</td><td>{layer}</td></tr>
                <tr><td>注射方式</td><td>{method}</td></tr>
                <tr><td>預估效果</td><td>{effect}</td></tr>
              </table>
            </div>""".format(**prod)
        alt_cards = ""
        for prod in rec["alternatives"]:
            alt_cards += """
            <div class="prod-card" style="border-left:3px solid #8888ff;background:#0a0a20;">
              <div class="prod-name">{name} <span class="tag-a">備選</span></div>
              <div style="color:#8090c0;font-size:0.8rem;">劑量：{dose}｜{effect}</div>
            </div>""".format(**prod)
        recs_html += """
        <div class="rec-sec" style="border-left:4px solid {color};">
          <div class="rec-hdr">
            <span class="rec-name">{name}</span>
            <span class="lv-badge" style="background:{color}20;color:{color};border-color:{color};">Lv{lv} {lv_name}</span>
            <span class="rec-score">評分 {score:.2f}</span>
          </div>
          <div class="rec-desc">{desc}</div>
          <div class="prod-grid">{primary}</div>
          {alt_sec}
        </div>""".format(
            color=lv_color, name=rec["name"],
            lv=rec["level"], lv_name=rec["level_name"],
            score=rec["score"], desc=rec.get("description",""),
            primary=primary_cards,
            alt_sec=('<div class="alt-label">備選方案</div><div class="prod-grid">'+alt_cards+'</div>') if alt_cards else "",
        )
    if not recs_html:
        recs_html = '<div style="color:#2ecc71;padding:10px;">未偵測到明顯問題，繼續維持良好保養即可！</div>'

    physio_html = ""
    for r in physio_data:
        zone_label = "（{} {}）".format(r["zone"], r["ratio"]) if r.get("zone") else ""
        physio_html += """
        <div class="physio-card">
          <div class="physio-title">{aspect}{zone}</div>
          <div class="physio-reading"><span class="pl1">面相解讀：</span>{reading}</div>
          <div class="physio-improve"><span class="pl2">醫美改善：</span>{improve}</div>
          <div class="physio-prods">建議產品：{products}</div>
        </div>""".format(
            aspect=r["aspect"], zone=zone_label,
            reading=r["reading"], improve=r["improve"],
            products="、".join(r.get("products",[])),
        )

    html = """<!DOCTYPE html>
<html lang="zh-TW"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AI 智能醫美面診報告</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:#0d1117;color:#e8e8e8;font-family:'Helvetica Neue',Arial,sans-serif;padding:20px;}}
.container{{max-width:960px;margin:0 auto;}}
.header{{text-align:center;padding:28px 0 18px;border-bottom:2px solid #e0c44a;margin-bottom:22px;}}
.header h1{{color:#ffd700;font-size:1.7rem;margin-bottom:5px;}}
.header .date{{color:#a0b4c8;font-size:0.88rem;}}
.section{{background:#161b22;border-radius:12px;padding:18px;margin-bottom:18px;border-left:4px solid #e0c44a;}}
.section h2{{color:#ffd700;font-size:1.05rem;margin-bottom:12px;}}
.img-row{{display:flex;gap:14px;flex-wrap:wrap;}}
.img-box{{flex:1;min-width:260px;}}
.img-box img{{width:100%;border-radius:8px;border:1px solid #30363d;}}
.img-caption{{color:#a0b4c8;font-size:0.8rem;text-align:center;margin-top:4px;}}
.legend-box{{background:#0d1520;border-radius:8px;padding:12px;margin-top:10px;font-size:0.82rem;color:#c0d0e0;line-height:1.8;}}
.zones-row{{display:flex;gap:10px;margin-bottom:10px;}}
.zone-card{{flex:1;background:#0d2030;border-radius:8px;padding:10px;text-align:center;}}
.zone-card .zn{{color:#a0b4c8;font-size:0.8rem;}}
.zone-card .zv{{color:#ffd700;font-size:1.4rem;font-weight:700;}}
.zone-card .zd{{font-size:0.76rem;}}
.zone-dom{{border:2px solid #e0c44a!important;}}
.info-box{{background:#0d2137;border-radius:6px;padding:8px 12px;color:#c0d8f0;font-size:0.86rem;margin-top:8px;}}
.score-row{{display:flex;align-items:center;gap:8px;margin-bottom:7px;flex-wrap:wrap;}}
.score-label{{min-width:85px;font-weight:600;color:#fff;font-size:0.86rem;}}
.score-bar-wrap{{flex:1;min-width:80px;height:7px;background:#30363d;border-radius:4px;overflow:hidden;}}
.score-bar{{height:100%;border-radius:4px;}}
.score-num{{min-width:34px;color:#a0b4c8;font-size:0.8rem;text-align:right;}}
.score-desc{{color:#505868;font-size:0.74rem;min-width:100px;}}
.lv-badge{{padding:2px 8px;border-radius:20px;font-size:0.73rem;font-weight:700;border:1px solid;white-space:nowrap;}}
.rec-sec{{background:#0f141c;border-radius:10px;padding:14px;margin-bottom:14px;border:1px solid #21262d;}}
.rec-hdr{{display:flex;align-items:center;gap:8px;margin-bottom:7px;flex-wrap:wrap;}}
.rec-name{{color:#fff;font-weight:700;font-size:0.98rem;}}
.rec-score{{color:#a0b4c8;font-size:0.8rem;margin-left:auto;}}
.rec-desc{{color:#7090a0;font-size:0.8rem;margin-bottom:9px;}}
.prod-grid{{display:flex;gap:8px;flex-wrap:wrap;}}
.prod-card{{flex:1;min-width:190px;border-radius:7px;padding:11px;}}
.prod-name{{color:#ffd700;font-weight:700;margin-bottom:7px;font-size:0.9rem;}}
.pt{{width:100%;font-size:0.79rem;border-collapse:collapse;}}
.pt td{{padding:2px 3px;color:#c0e8c0;vertical-align:top;}}
.pt td:first-child{{color:#7090a0;width:65px;white-space:nowrap;}}
.tag-p{{background:#1a3a2a;color:#2ecc71;border:1px solid #2ecc71;padding:1px 6px;border-radius:10px;font-size:0.7rem;margin-left:4px;}}
.tag-a{{background:#1a1a3a;color:#8888ff;border:1px solid #8888ff;padding:1px 6px;border-radius:10px;font-size:0.7rem;margin-left:4px;}}
.alt-label{{color:#8888ff;font-weight:600;font-size:0.83rem;margin:9px 0 5px;}}
.physio-card{{background:#0f1520;border-radius:9px;padding:13px;margin-bottom:11px;border-left:3px solid #e0c44a;}}
.physio-title{{color:#ffd700;font-weight:700;margin-bottom:7px;}}
.physio-reading,.physio-improve{{font-size:0.86rem;margin-bottom:5px;}}
.physio-improve{{background:#0a1f10;border-radius:5px;padding:7px;color:#a0d0a0;}}
.physio-prods{{color:#7090a0;font-size:0.78rem;margin-top:5px;}}
.pl1{{color:#ffd700;font-weight:600;}}
.pl2{{color:#2ecc71;font-weight:600;}}
.footer{{text-align:center;padding:18px;color:#505868;font-size:0.78rem;border-top:1px solid #21262d;margin-top:18px;}}
</style></head><body><div class="container">

<div class="header">
  <h1>AI 智能醫美面診報告</h1>
  <div class="date">分析日期：{timestamp}</div>
</div>

<div class="section">
  <h2>臉部治療標注圖</h2>
  <div class="img-row">
    <div class="img-box">
      <img src="data:image/jpeg;base64,{b64_map}" alt="治療標注圖">
      <div class="img-caption">治療部位標注（數字對應下方說明）</div>
    </div>
    <div class="img-box">
      <img src="data:image/jpeg;base64,{b64_basic}" alt="三庭五眼">
      <div class="img-caption">三庭五眼比例標注</div>
    </div>
  </div>
  <div class="legend-box">{legend_html}</div>
</div>

<div class="section">
  <h2>三庭比例分析</h2>
  <div class="zones-row">
    <div class="zone-card {cls_u}">
      <div class="zn">上庭</div><div class="zv">{upper:.1%}</div>
      <div class="zd" style="color:{cu}">{du:+.1%}</div>
    </div>
    <div class="zone-card {cls_m}">
      <div class="zn">中庭</div><div class="zv">{middle:.1%}</div>
      <div class="zd" style="color:{cm}">{dm:+.1%}</div>
    </div>
    <div class="zone-card {cls_l}">
      <div class="zn">下庭</div><div class="zv">{lower:.1%}</div>
      <div class="zd" style="color:{cl}">{dl:+.1%}</div>
    </div>
  </div>
  <div class="info-box">主導分區：<strong style="color:#ffd700">{dominant}</strong>｜理想各區均為 33.3%</div>
</div>

<div class="section">
  <h2>五眼比例</h2>
  <div class="zones-row">
    <div class="zone-card">
      <div class="zn">五眼比例</div><div class="zv">{fer:.2f}</div>
      <div class="zd" style="color:{fec}">{fed:+.2f} 理想1.0</div>
    </div>
    <div class="zone-card">
      <div class="zn">眼距比</div><div class="zv">{ir:.2f}</div>
      <div class="zd" style="color:{irc}">{ird:+.2f} 理想1.0</div>
    </div>
  </div>
</div>

<div class="section">
  <h2>問題診斷評分（5級制：Lv1極輕微 → Lv5重度）</h2>
  {scored_html}
</div>

<div class="section">
  <h2>個人化治療建議</h2>
  {recs_html}
</div>

<div class="section">
  <h2>面相學分析與醫美改善建議</h2>
  {physio_html}
</div>

<div class="footer">本報告由 AI 智能醫美面診輔助系統生成，僅供參考，不構成醫療建議。<br>實際治療請諮詢合法執照醫師。</div>
</div></body></html>""".format(
        timestamp=timestamp,
        b64_map=b64_map, b64_basic=b64_basic,
        legend_html=legend_html,
        upper=zones.get("upper_ratio", 0),
        middle=zones.get("middle_ratio", 0),
        lower=zones.get("lower_ratio", 0),
        du=zones.get("upper_ratio", 0)-0.333,
        dm=zones.get("middle_ratio", 0)-0.333,
        dl=zones.get("lower_ratio", 0)-0.333,
        cls_u="zone-dom" if zones.get("dominant")=="upper" else "",
        cls_m="zone-dom" if zones.get("dominant")=="middle" else "",
        cls_l="zone-dom" if zones.get("dominant")=="lower" else "",
        cu="#2ecc71" if abs(zones.get("upper_ratio",0)-0.333)<0.06 else "#e74c3c",
        cm="#2ecc71" if abs(zones.get("middle_ratio",0)-0.333)<0.06 else "#e74c3c",
        cl="#2ecc71" if abs(zones.get("lower_ratio",0)-0.333)<0.06 else "#e74c3c",
        dominant=dominant,
        fer=fe.get("five_eye_ratio",1.0),
        fed=fe.get("five_eye_ratio",1.0)-1.0,
        fec="#2ecc71" if abs(fe.get("five_eye_ratio",1.0)-1.0)<0.12 else "#e74c3c",
        ir=fe.get("inter_ratio",1.0),
        ird=fe.get("inter_ratio",1.0)-1.0,
        irc="#2ecc71" if abs(fe.get("inter_ratio",1.0)-1.0)<0.18 else "#e74c3c",
        scored_html=scored_html,
        recs_html=recs_html,
        physio_html=physio_html,
    )
    return html


# ─── Calf / Back ─────────────────────────────────────────────────────────────
def analyze_calf(img_bgr):
    R = {"detected": False, "score": 0.0, "severity": "mild", "calf_ratio": 0.0}
    try:
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            res = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return R
        h, w = img_bgr.shape[:2]
        lm = res.pose_landmarks.landmark
        lk = lm[mp_pose.PoseLandmark.LEFT_KNEE]
        la = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
        rk = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
        ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
        if abs(lk.y - la.y) * h < 10:
            return R
        calf_w = abs(lk.x - rk.x) * w
        calf_ratio = calf_w / (w * 0.6 + 1e-6)
        score = norm(float(calf_ratio), 0.2, 0.6)
        sv = grade(score)[1]
        R = {
            "detected": True, "score": round(score, 3), "severity": sv,
            "calf_ratio": round(float(calf_ratio), 3),
            "left_knee":  (int(lk.x*w), int(lk.y*h)),
            "left_ankle": (int(la.x*w), int(la.y*h)),
            "right_knee": (int(rk.x*w), int(rk.y*h)),
            "right_ankle":(int(ra.x*w), int(ra.y*h)),
        }
    except Exception:
        pass
    return R


def analyze_back(img_front, img_side=None):
    R = {"detected": False, "score": 0.0, "severity": "mild",
         "back_width_ratio": 0.0, "shoulder_width_px": 0.0,
         "hip_width_px": 0.0, "shoulder_hip_ratio": 0.0, "back_thickness_ratio": 0.0}
    try:
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            res = pose.process(cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return R
        h, w = img_front.shape[:2]
        lm = res.pose_landmarks.landmark
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lhip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        rhip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        sw = abs(ls.x - rs.x) * w
        hw = abs(lhip.x - rhip.x) * w
        bwr = sw / (w * 0.8 + 1e-6)
        score = norm(bwr, 0.3, 0.7)
        lv, lv_name, _ = grade(score)
        R = {"detected": True, "score": round(score, 3), "severity": lv_name,
             "back_width_ratio": round(float(bwr), 3),
             "shoulder_width_px": round(float(sw), 1),
             "hip_width_px": round(float(hw), 1),
             "shoulder_hip_ratio": round(float(sw / (hw + 1e-6)), 3),
             "back_thickness_ratio": 0.0}
        if img_side is not None:
            with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
                res2 = pose.process(cv2.cvtColor(img_side, cv2.COLOR_BGR2RGB))
            if res2.pose_landmarks:
                h2, w2 = img_side.shape[:2]
                lm2 = res2.pose_landmarks.landmark
                ns2 = lm2[mp_pose.PoseLandmark.NOSE]
                ls2 = lm2[mp_pose.PoseLandmark.LEFT_SHOULDER]
                R["back_thickness_ratio"] = round(abs(ns2.x - ls2.x), 3)
    except Exception:
        pass
    return R


# ─── CSS ────────────────────────────────────────────────────────────────────
def apply_styles():
    st.markdown("""
<style>
.stApp{background:#0d1117;}
h1,h2,h3,h4{color:#ffd700!important;}
p,li{color:#e8e8e8!important;}
.stMarkdown p{color:#e8e8e8!important;}
label,.stRadio label{color:#e8e8e8!important;font-weight:600;}
.stCaption{color:#a0b4c8!important;}
.ulabel{font-size:.95rem;font-weight:700;color:#ffd700;margin-bottom:6px;display:block;}
.aok{color:#2ecc71;font-size:.88rem;font-weight:600;}
.afail{color:#e74c3c;font-size:.88rem;font-weight:600;}
.awarn{color:#f39c12;font-size:.88rem;}
.stButton>button{background:linear-gradient(135deg,#e0c44a,#c0a030)!important;color:#000!important;font-weight:700!important;border-radius:8px!important;border:none!important;}
div[data-testid="metric-container"] label{color:#a0b4c8!important;}
div[data-testid="metric-container"] [data-testid="metric-value"]{color:#ffd700!important;}
div[data-testid="stInfo"] p{color:#c0d8f0!important;}
div[data-testid="stSuccess"] p{color:#c0f0c0!important;}
div[data-testid="stWarning"] p{color:#f0d080!important;}
.stExpander summary p{color:#ffd700!important;font-weight:600;}
</style>""", unsafe_allow_html=True)


def lv_badge_html(score):
    lv, lv_name, color = grade(score)
    return "<span style='background:{c}20;color:{c};border:1px solid {c};padding:2px 10px;border-radius:20px;font-size:.76rem;font-weight:700;'>Lv{lv} {name}</span>".format(
        c=color, lv=lv, name=lv_name)


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    apply_styles()
    st.title("AI 智能醫美面診輔助系統")
    st.caption("上傳照片，AI 自動分析並生成個人化 HTML 治療建議報告")
    st.markdown("---")

    mode = st.radio("**選擇分析模式**",
                    ["臉部分析", "小腿肌肉分析", "背部分析"], horizontal=True)

    # ══ 臉部分析 ══════════════════════════════════════════════════════════════
    if mode == "臉部分析":
        st.markdown("---")
        st.subheader("上傳臉部照片（最多5個角度）")
        st.info("系統會自動偵測人臉方向並旋轉。側臉角度僅供參考，系統不管角度都會進行分析。")

        ANGLES = {
            "front":   {"expected": 0,   "tol": 20, "label": "① 正面（0度）"},
            "left45":  {"expected": -35, "tol": 25, "label": "② 左側45度"},
            "left90":  {"expected": -70, "tol": 30, "label": "③ 左側90度"},
            "right45": {"expected": 35,  "tol": 25, "label": "④ 右側45度"},
            "right90": {"expected": 70,  "tol": 30, "label": "⑤ 右側90度"},
        }

        c1, c2, c3, c4, c5 = st.columns(5)
        cols = {"front": c1, "left45": c2, "left90": c3, "right45": c4, "right90": c5}
        uploads = {}

        for key, col in cols.items():
            with col:
                st.markdown("<span class='ulabel'>{}</span>".format(ANGLES[key]["label"]),
                            unsafe_allow_html=True)
                f = st.file_uploader("", type=["jpg","jpeg","png"],
                                     key=key, label_visibility="collapsed")
                if f:
                    pil = Image.open(f).convert("RGB")
                    pil = auto_rotate_face(pil)
                    uploads[key] = pil
                    st.image(pil, use_container_width=True)

        if uploads:
            st.markdown("---")
            st.subheader("角度驗證（僅供參考，不影響分析）")
            vcols = st.columns(len(uploads))
            for i, (key, pil_img) in enumerate(uploads.items()):
                cfg = ANGLES[key]
                img_bgr = pil_to_cv2(resize_image(pil_img, 640))
                lm, _ = get_landmarks(img_bgr)
                with vcols[i]:
                    if lm is None:
                        st.markdown("<div class='awarn'>⚠ 偵測不到人臉<br>仍可進行分析</div>",
                                    unsafe_allow_html=True)
                    else:
                        yaw = estimate_yaw_frontal(lm)
                        # For side angles, yaw estimation is unreliable
                        if key == "front":
                            ok = abs(yaw - cfg["expected"]) <= cfg["tol"]
                            cls = "aok" if ok else "awarn"
                            msg = "OK" if ok else "略偏"
                        else:
                            msg = "已上傳"
                            cls = "aok"
                        st.markdown("<div class='{}'>{}<br>（Yaw~{:.0f}）</div>".format(
                            cls, msg, yaw), unsafe_allow_html=True)

        st.markdown("---")

        if "front" in uploads:
            if st.button("開始 AI 面診分析（生成 HTML 報告）", use_container_width=True):
                with st.spinner("AI 分析中，請稍候..."):
                    try:
                        front_bgr = pil_to_cv2(resize_image(uploads["front"], 800))
                        lm_front, _ = get_landmarks(front_bgr)
                        if lm_front is None:
                            st.error("無法偵測正面人臉，請上傳更清晰的照片。")
                            return

                        analysis = analyze_face(lm_front, front_bgr)
                        recs     = generate_recs(analysis)
                        physio   = physio_analysis(analysis)
                        ts       = datetime.now().strftime("%Y年%m月%d日 %H:%M")

                        img_basic_cv = draw_basic_annotations(front_bgr, lm_front, analysis)
                        img_map_pil  = draw_treatment_map(front_bgr, lm_front, recs)
                        img_basic_pil = cv2_to_pil(img_basic_cv)

                        # Build legend HTML
                        top_recs = sorted([r for r in recs if r["score"] >= SCORE_THRESHOLD],
                                          key=lambda x: x["score"], reverse=True)[:8]
                        legend_parts = []
                        for i, rec in enumerate(top_recs):
                            lv_color = rec["level_color"]
                            dose_str = rec["primary"][0]["dose"] if rec["primary"] else "-"
                            prod_name = rec["primary"][0]["name"] if rec["primary"] else "-"
                            legend_parts.append(
                                "<strong style='color:{c}'>[{n}] {prob}</strong>：{pname}，劑量 <strong style='color:#ffd700'>{dose}</strong>（首選）".format(
                                    c=lv_color, n=i+1, prob=rec["name"],
                                    pname=prod_name, dose=dose_str))
                        legend_html_str = "<br>".join(legend_parts)

                        st.success("分析完成！")

                        # Display images
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.subheader("治療部位標注圖")
                            st.image(img_map_pil, use_container_width=True)
                        with col_b:
                            st.subheader("三庭五眼標注圖")
                            st.image(img_basic_pil, use_container_width=True)

                        # Legend table
                        st.subheader("標注說明（數字對應上圖）")
                        for i, rec in enumerate(top_recs):
                            dose_str = rec["primary"][0]["dose"] if rec["primary"] else "-"
                            prod_name = rec["primary"][0]["name"] if rec["primary"] else "-"
                            prod_layer = rec["primary"][0]["layer"] if rec["primary"] else "-"
                            st.markdown(
                                "<div style='display:flex;align-items:center;gap:10px;"
                                "margin-bottom:6px;padding:8px;background:#0f141c;"
                                "border-radius:8px;border-left:3px solid {color};'>"
                                "<span style='background:#c82828;color:#fff;border-radius:50%;"
                                "width:24px;height:24px;display:inline-flex;align-items:center;"
                                "justify-content:center;font-weight:700;flex-shrink:0;'>{n}</span>"
                                "<span style='color:#ffd700;font-weight:700;min-width:70px;'>{prob}</span>"
                                "{lv_badge}"
                                "<span style='color:#c0e8c0;font-size:.88rem;flex:1;'>"
                                "{prod} ｜ 劑量：<strong style='color:#ffd700'>{dose}</strong> ｜ 層次：{layer}</span>"
                                "</div>".format(
                                    color=rec["level_color"], n=i+1,
                                    prob=rec["name"],
                                    lv_badge=lv_badge_html(rec["score"]),
                                    prod=prod_name, dose=dose_str, layer=prod_layer),
                                unsafe_allow_html=True)

                        # Zones
                        st.subheader("三庭比例")
                        zones = analysis["zones"]
                        zc1, zc2, zc3 = st.columns(3)
                        dm = {"upper": "上庭", "middle": "中庭", "lower": "下庭"}
                        zc1.metric("上庭", "{:.1%}".format(zones["upper_ratio"]),
                                   "{:+.1%}".format(zones["upper_ratio"]-0.333))
                        zc2.metric("中庭", "{:.1%}".format(zones["middle_ratio"]),
                                   "{:+.1%}".format(zones["middle_ratio"]-0.333))
                        zc3.metric("下庭", "{:.1%}".format(zones["lower_ratio"]),
                                   "{:+.1%}".format(zones["lower_ratio"]-0.333))
                        st.info("主導分區：{} ｜ 理想各區均為33.3%".format(
                            dm.get(zones["dominant"], zones["dominant"])))

                        # Scores
                        st.subheader("問題診斷（5級制：Lv1極輕微 → Lv5重度）")
                        all_scored = sorted(
                            [(k, v) for k, v in analysis.items()
                             if isinstance(v, dict) and "score" in v],
                            key=lambda x: x[1]["score"], reverse=True)
                        for pk, data in all_scored:
                            name = PROBLEM_ZH.get(pk, pk)
                            score = data["score"]
                            lv, lv_name, lv_color = grade(score)
                            desc = data.get("description", "")
                            st.markdown(
                                "<div style='display:flex;align-items:center;gap:8px;"
                                "margin-bottom:4px;'>"
                                "<span style='color:#fff;font-weight:700;min-width:85px;"
                                "font-size:.88rem;'>{name}</span>"
                                "<span style='background:{c}20;color:{c};border:1px solid {c};"
                                "padding:1px 8px;border-radius:16px;font-size:.72rem;"
                                "font-weight:700;white-space:nowrap;'>Lv{lv} {lv_name}</span>"
                                "<span style='color:#555;font-size:.76rem;flex:1;'>{desc}</span>"
                                "<span style='color:{c};font-weight:700;font-size:.85rem;"
                                "min-width:32px;text-align:right;'>{score:.2f}</span>"
                                "</div>".format(
                                    name=name, c=lv_color, lv=lv, lv_name=lv_name,
                                    desc=desc, score=score),
                                unsafe_allow_html=True)
                            st.progress(score)

                        # Recs
                        st.subheader("個人化治療建議")
                        if recs:
                            for rec in recs:
                                label = "{} — Lv{} {} ({:.2f})".format(
                                    rec["name"], rec["level"], rec["level_name"], rec["score"])
                                with st.expander(label, expanded=rec["score"] > 0.35):
                                    if rec["primary"]:
                                        st.markdown(
                                            "<div style='color:#2ecc71;font-weight:700;"
                                            "margin-bottom:6px;'>首選治療方案</div>",
                                            unsafe_allow_html=True)
                                        for prod in rec["primary"]:
                                            st.markdown(
                                                "<div style='background:#0a1f10;border-radius:8px;"
                                                "padding:13px;margin:5px 0;border-left:4px solid #2ecc71;'>"
                                                "<div style='color:#ffd700;font-weight:700;margin-bottom:5px;'>{name}</div>"
                                                "<div style='color:#c0e8c0;font-size:.87rem;line-height:1.9;'>"
                                                "品牌：{brand}<br>"
                                                "建議劑量：<strong style='color:#ffd700'>{dose}</strong><br>"
                                                "注射層次：{layer}<br>注射方式：{method}<br>預估效果：{effect}"
                                                "</div></div>".format(**prod),
                                                unsafe_allow_html=True)
                                    if rec["alternatives"]:
                                        st.markdown(
                                            "<div style='color:#8888ff;font-weight:700;"
                                            "margin:9px 0 5px;'>備選方案</div>",
                                            unsafe_allow_html=True)
                                        acols = st.columns(min(len(rec["alternatives"]), 3))
                                        for j, prod in enumerate(rec["alternatives"]):
                                            with acols[j % len(acols)]:
                                                st.markdown(
                                                    "<div style='background:#0a0a20;border-radius:7px;"
                                                    "padding:9px;border-left:3px solid #8888ff;'>"
                                                    "<div style='color:#c0c8ff;font-weight:600;font-size:.88rem;'>{name}</div>"
                                                    "<div style='color:#8090c0;font-size:.79rem;'>"
                                                    "劑量：{dose}<br>{effect}</div>"
                                                    "</div>".format(**prod),
                                                    unsafe_allow_html=True)
                        else:
                            st.success("未偵測到明顯問題，繼續維持良好保養即可！")

                        # Physio
                        st.subheader("面相學分析與醫美改善建議")
                        for r in physio:
                            zone_label = "｜{} {}".format(r["zone"], r["ratio"]) if r.get("zone") else ""
                            st.markdown(
                                "<div style='background:#1a1a2e;border-radius:10px;padding:15px;"
                                "margin-bottom:13px;border-left:4px solid #e0c44a;'>"
                                "<div style='color:#ffd700;font-weight:700;font-size:.98rem;margin-bottom:7px;'>{}{}</div>"
                                "<div style='background:#0f1520;border-radius:5px;padding:9px;margin-bottom:7px;'>"
                                "<span style='color:#ffd700;font-size:.83rem;font-weight:600;'>面相解讀：</span>"
                                "<span style='color:#c8d8e8;font-size:.86rem;'>{}</span></div>"
                                "<div style='background:#0f2015;border-radius:5px;padding:9px;'>"
                                "<span style='color:#2ecc71;font-size:.83rem;font-weight:600;'>醫美改善建議：</span>"
                                "<span style='color:#a0d0a0;font-size:.86rem;'>{}</span><br>"
                                "<span style='color:#7090a0;font-size:.79rem;'>建議產品：{}</span>"
                                "</div></div>".format(
                                    r["aspect"], zone_label, r["reading"],
                                    r["improve"], "、".join(r.get("products",[]))),
                                unsafe_allow_html=True)

                        # Download HTML
                        st.subheader("下載精美 HTML 報告")
                        html_report = generate_html_report(
                            analysis, recs, physio,
                            img_basic_pil, img_map_pil,
                            legend_html_str, ts)
                        st.download_button(
                            "下載完整 HTML 報告（含圖片，用瀏覽器開啟）",
                            data=html_report.encode("utf-8"),
                            file_name="face_report_{}.html".format(
                                datetime.now().strftime("%Y%m%d_%H%M")),
                            mime="text/html",
                            use_container_width=True)
                        st.warning("免責聲明：本系統為輔助參考工具，不構成醫療建議，實際治療請諮詢合法執照醫師。")

                    except Exception as e:
                        st.error("分析錯誤：{}".format(e))
                        st.exception(e)
        else:
            st.info("請先上傳正面照片後開始分析。")

    # ══ 小腿 ══════════════════════════════════════════════════════════════════
    elif mode == "小腿肌肉分析":
        st.markdown("---")
        st.subheader("小腿肌肉肥大分析")
        st.info("全身正面站立，膝蓋至腳踝完整入鏡，光線充足。系統會自動旋轉方向。")
        col1, col2 = st.columns(2)
        calf_n = None
        calf_t = None
        with col1:
            st.markdown("<span class='ulabel'>① 自然站姿（正面）</span>", unsafe_allow_html=True)
            f1 = st.file_uploader("", type=["jpg","jpeg","png"],
                                  key="calf_n", label_visibility="collapsed")
            if f1:
                calf_n = auto_rotate_face(Image.open(f1).convert("RGB"))
                st.image(calf_n, use_container_width=True)
        with col2:
            st.markdown("<span class='ulabel'>② 墊腳尖（選填）</span>", unsafe_allow_html=True)
            f2 = st.file_uploader("", type=["jpg","jpeg","png"],
                                  key="calf_t", label_visibility="collapsed")
            if f2:
                calf_t = auto_rotate_face(Image.open(f2).convert("RGB"))
                st.image(calf_t, use_container_width=True)

        if calf_n:
            if st.button("開始小腿分析", use_container_width=True):
                with st.spinner("分析中..."):
                    try:
                        img_bgr = pil_to_cv2(resize_image(calf_n, 800))
                        result = analyze_calf(img_bgr)
                        ann = img_bgr.copy()
                        if result["detected"]:
                            C = (0, 220, 255)
                            for k in ("left_knee","left_ankle","right_knee","right_ankle"):
                                if k in result:
                                    cv2.circle(ann, result[k], 8, C, -1)
                            if "left_knee" in result and "left_ankle" in result:
                                cv2.line(ann, result["left_knee"], result["left_ankle"], C, 2)
                            if "right_knee" in result and "right_ankle" in result:
                                cv2.line(ann, result["right_knee"], result["right_ankle"], C, 2)
                        st.success("分析完成！")
                        st.image(cv2_to_pil(ann), use_container_width=True, caption="小腿關鍵點標注")
                        if result["detected"]:
                            score = result["score"]
                            lv, lv_name, lv_color = grade(score)
                            st.markdown(
                                "<div style='background:#1a1a2e;border-radius:10px;padding:15px;"
                                "border-left:4px solid {c};margin:12px 0;'>"
                                "<div style='color:#ffd700;font-size:1.05rem;font-weight:700;margin-bottom:7px;'>"
                                "小腿肌肉肥大程度：<span style='color:{c};'>Lv{lv} {name}</span></div>"
                                "<div style='color:#c0d0e0;'>寬度比例：{ratio:.3f} ｜ AI評分：{score:.2f}/1.00</div>"
                                "</div>".format(c=lv_color, lv=lv, name=lv_name,
                                               ratio=result["calf_ratio"], score=score),
                                unsafe_allow_html=True)
                            st.progress(score)
                            dose = DOSE_TABLE["botox_calf"].get(lv, "依醫師評估")
                            name, brand, layer, method, effect = PRODUCT_ZH["botox_calf"]
                            st.subheader("小腿肉毒毒素治療建議")
                            st.markdown(
                                "<div style='background:#0a1f10;border-radius:10px;padding:15px;"
                                "border-left:4px solid #2ecc71;'>"
                                "<div style='color:#ffd700;font-size:1rem;font-weight:700;margin-bottom:9px;'>"
                                "腓腸肌肉毒毒素注射</div>"
                                "<div style='color:#c0e8c0;font-size:.9rem;line-height:2;'>"
                                "品牌：{brand}<br>"
                                "建議劑量（Lv{lv} {lv_name}）：<strong style='color:#ffd700'>{dose}</strong><br>"
                                "注射層次：{layer}<br>注射方式：{method}<br>預估效果：{effect}"
                                "</div></div>".format(brand=brand, lv=lv, lv_name=lv_name,
                                                     dose=dose, layer=layer, method=method, effect=effect),
                                unsafe_allow_html=True)
                            for tip in ["治療前2週停止高強度腿部訓練",
                                        "注射後4-6小時內請勿按摩注射部位",
                                        "注射後24小時內避免劇烈運動、三溫暖及飲酒",
                                        "建議每6-9個月維持一次療程",
                                        "可搭配小腿拉伸運動加速放鬆效果"]:
                                st.markdown("<div style='color:#c0d0e0;padding:3px 0;'>• {}</div>".format(tip),
                                            unsafe_allow_html=True)
                            if calf_t:
                                st.subheader("墊腳尖對比分析")
                                tip_bgr = pil_to_cv2(resize_image(calf_t, 800))
                                tip_result = analyze_calf(tip_bgr)
                                tip_ann = tip_bgr.copy()
                                if tip_result["detected"]:
                                    C = (0, 220, 255)
                                    for k in ("left_knee","left_ankle","right_knee","right_ankle"):
                                        if k in tip_result:
                                            cv2.circle(tip_ann, tip_result[k], 8, C, -1)
                                ca, cb = st.columns(2)
                                with ca:
                                    st.image(cv2_to_pil(ann), caption="自然站姿", use_container_width=True)
                                    st.metric("肌肉評分", "{:.2f}".format(result["score"]))
                                with cb:
                                    st.image(cv2_to_pil(tip_ann), caption="墊腳尖", use_container_width=True)
                                    if tip_result["detected"]:
                                        delta = tip_result["score"] - result["score"]
                                        st.metric("肌肉評分", "{:.2f}".format(tip_result["score"]),
                                                  delta="{:+.2f}".format(delta))
                        else:
                            st.warning("無法偵測小腿關鍵點，請確認膝蓋至腳踝完整入鏡。")
                    except Exception as e:
                        st.error("分析錯誤：{}".format(e))
                        st.exception(e)
        else:
            st.info("請上傳自然站姿照片。")

    # ══ 背部 ══════════════════════════════════════════════════════════════════
    elif mode == "背部分析":
        st.markdown("---")
        st.subheader("背部肌肉與輪廓分析")
        st.info("請上傳：正背面（背對鏡頭）及側背面（選填）。系統會自動旋轉方向。")
        col1, col2 = st.columns(2)
        back_f = None
        back_s = None
        with col1:
            st.markdown("<span class='ulabel'>① 正背面（背對鏡頭）</span>", unsafe_allow_html=True)
            bf = st.file_uploader("", type=["jpg","jpeg","png"],
                                  key="back_f", label_visibility="collapsed")
            if bf:
                back_f = auto_rotate_face(Image.open(bf).convert("RGB"))
                st.image(back_f, use_container_width=True)
        with col2:
            st.markdown("<span class='ulabel'>② 側背面（選填）</span>", unsafe_allow_html=True)
            bs = st.file_uploader("", type=["jpg","jpeg","png"],
                                  key="back_s", label_visibility="collapsed")
            if bs:
                back_s = auto_rotate_face(Image.open(bs).convert("RGB"))
                st.image(back_s, use_container_width=True)

        if back_f:
            if st.button("開始背部分析", use_container_width=True):
                with st.spinner("分析中..."):
                    try:
                        fbgr = pil_to_cv2(resize_image(back_f, 800))
                        sbgr = pil_to_cv2(resize_image(back_s, 800)) if back_s else None
                        br = analyze_back(fbgr, sbgr)
                        st.success("分析完成！")
                        if br["detected"]:
                            score = br["score"]
                            lv, lv_name, lv_color = grade(score)
                            st.markdown(
                                "<div style='background:#1a1a2e;border-radius:10px;padding:15px;"
                                "border-left:4px solid {c};margin:12px 0;'>"
                                "<div style='color:#ffd700;font-size:1.05rem;font-weight:700;margin-bottom:7px;'>"
                                "背部評估：<span style='color:{c};'>Lv{lv} {lv_name}</span></div>"
                                "<div style='color:#c0d0e0;'>"
                                "背部寬度比例：{bwr:.2f} ｜ 肩臀比：{shr:.2f}</div>"
                                "</div>".format(c=lv_color, lv=lv, lv_name=lv_name,
                                               bwr=br.get("back_width_ratio",0),
                                               shr=br.get("shoulder_hip_ratio",0)),
                                unsafe_allow_html=True)
                            bc1, bc2, bc3 = st.columns(3)
                            bc1.metric("背部寬度比例", "{:.2f}".format(br.get("back_width_ratio",0)))
                            bc2.metric("肩膀寬度", "{:.0f}px".format(br.get("shoulder_width_px",0)))
                            bc3.metric("肩臀比", "{:.2f}".format(br.get("shoulder_hip_ratio",0)))
                            if br.get("back_thickness_ratio",0) > 0:
                                st.metric("側面背部厚度", "{:.2f}".format(br["back_thickness_ratio"]))
                            st.subheader("背部醫美治療建議")
                            if score > 0.5:
                                recs_back = [
                                    ("背部肉毒毒素（豎脊肌放鬆）","針對背部豎脊肌過度發達，放鬆肌肉改善背部寬厚感","每側50-100U","2-4週改善，4-6個月"),
                                    ("背部溶脂針","針對背部脂肪堆積，溶解局部脂肪","4-8ml/療程","4-8週，效果持久"),
                                    ("背部音波拉提","改善背部皮膚鬆弛，緊緻輪廓","300-500發","3-6月顯效，12-18個月"),
                                ]
                            else:
                                recs_back = [
                                    ("背部水光注射","改善背部皮膚乾燥粗糙","2-4ml","即時保濕，4-6個月"),
                                    ("背部皮秒雷射","改善背部色斑、毛孔粗大","1-3次療程","每4-6週一次"),
                                    ("背部肉毒毒素（局部雕塑）","局部肌肉精細雕塑","每側30-50U","2-4週，4-6個月"),
                                ]
                            for n, d, dose, eff in recs_back:
                                st.markdown(
                                    "<div style='background:#0d1a20;border-radius:8px;padding:13px;"
                                    "margin:7px 0;border-left:4px solid #e0c44a;'>"
                                    "<div style='color:#ffd700;font-weight:700;margin-bottom:5px;'>{}</div>"
                                    "<div style='color:#c0d0e0;font-size:.87rem;margin-bottom:3px;'>{}</div>"
                                    "<div style='color:#90a0b0;font-size:.82rem;'>劑量：{} ｜ 效果：{}</div>"
                                    "</div>".format(n, d, dose, eff),
                                    unsafe_allow_html=True)
                        else:
                            st.warning("無法偵測背部關鍵點，請確認照片清晰，背對鏡頭，完整上半身入鏡。")
                    except Exception as e:
                        st.error("分析錯誤：{}".format(e))
                        st.exception(e)
        else:
            st.info("請上傳正背面照片以開始分析。")

        st.markdown("---")
        st.warning("免責聲明：本系統為輔助參考工具，不構成醫療建議，實際治療請諮詢合法執照醫師。")


if __name__ == "__main__":
    main()
