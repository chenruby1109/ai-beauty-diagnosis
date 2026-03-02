import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import base64
import io
import math
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ExifTags

st.set_page_config(
    page_title="AI Beauty System",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Product DB ─────────────────────────────────────────────────────────────
PRODUCT_ZH = {
    "botox_forehead":  ("肉毒毒素（抬頭紋）",  "奇蹟/天使肉毒", "額肌（肌肉層）",       "多點注射，間距約1.5cm",      "3-7天見效，維持4-6個月"),
    "botox_glabella":  ("肉毒毒素（眉間紋）",  "奇蹟/天使肉毒", "皺眉肌",               "5點標準注射法",              "3-7天改善，維持4-6個月"),
    "botox_crowsfeet": ("肉毒毒素（魚尾紋）",  "奇蹟/天使肉毒", "眼輪匝肌",             "眼外角扇形多點注射",          "5-7天平滑，維持3-5個月"),
    "botox_jawline":   ("肉毒毒素（下頜緣）",  "奇蹟/天使肉毒", "頸闊肌",               "沿頸闊肌線性注射",            "輪廓提升，維持4-6個月"),
    "botox_calf":      ("肉毒毒素（腓腸肌）",  "奇蹟/天使肉毒", "腓腸肌內側頭",         "多點格狀注射，每點5-10U",    "4-8週改善，維持6-9個月"),
    "ha_cheek":        ("玻尿酸（蘋果肌）",    "喬雅登VOLUMA",  "骨膜上層或深層脂肪",   "扇形注射/線性推注",          "維持18-24個月"),
    "ha_chin":         ("玻尿酸（下巴）",      "喬雅登VOLUMA",  "骨膜上層",             "單點或扇形注射",              "維持12-18個月"),
    "ha_nasolabial":   ("玻尿酸（法令紋）",    "喬雅登VOLUMA",  "深層皮下/骨膜上層",    "逆行線性+扇形",              "法令紋減少60-80%，維持12-18個月"),
    "ha_jawline":      ("玻尿酸（下頜輪廓）",  "喬雅登VOLUX",   "骨膜上層",             "線性注射沿下頜骨緣",          "維持18-24個月"),
    "ha_nasolabial2":  ("玻尿酸（法令紋II）",  "喬雅登VOLIFT",  "真皮深層至皮下層",     "逆行線性+蕨葉技術",          "維持12-15個月"),
    "ha_tear":         ("玻尿酸（淚溝）",      "喬雅登VOLBELLA","眶隔前脂肪/骨膜上層",  "微量多點/線性注射",          "維持9-12個月"),
    "ha_skin":         ("玻尿酸（全臉保濕）",  "喬雅登VOLITE",  "真皮中層",             "多點注射/水光槍輔助",         "維持6-9個月"),
    "sculptra":        ("舒顏萃Sculptra",      "Sculptra",      "真皮深層至皮下層",     "扇形大範圍注射+按摩",         "2-3月顯現，維持18-24個月"),
    "thread":          ("鳳凰埋線",            "PDO大V線",      "SMAS筋膜層/深層皮下",  "逆行進針，雙向倒鉤提拉",      "即時提拉，維持12-18個月"),
    "pn":              ("麗珠蘭PN 1%",         "Rejuran PN",    "真皮淺中層",           "水光槍/多點注射",             "膚色提亮，建議3-4療程"),
    "picosecond":      ("皮秒雷射",            "皮秒雷射",      "表皮至真皮層",         "全臉掃描依斑點加強",          "膚色均勻，每4-6週一次"),
    "ultherapy":       ("音波拉提Ultherapy",   "Ultherapy",     "SMAS筋膜+真皮層",      "線性掃描，分層施打",          "3-6月顯效，維持12-18個月"),
    "thermage":        ("電波拉提Thermage",    "Thermage FLX",  "真皮深層至皮下層",     "全臉均勻掃描",                "即時緊緻，維持12-24個月"),
}

DOSE_TABLE = {
    "botox_forehead":  {"mild": "12U",       "moderate": "20U",       "severe": "32U"},
    "botox_glabella":  {"mild": "12U",       "moderate": "20U",       "severe": "32U"},
    "botox_crowsfeet": {"mild": "7U/側",     "moderate": "12U/側",    "severe": "17U/側"},
    "botox_jawline":   {"mild": "15U",       "moderate": "25U",       "severe": "40U"},
    "botox_calf":      {"mild": "65U/側",    "moderate": "90U/側",    "severe": "125U/側"},
    "ha_cheek":        {"mild": "0.75ml/側", "moderate": "1.25ml/側", "severe": "1.75ml/側"},
    "ha_chin":         {"mild": "0.75ml",    "moderate": "1.25ml",    "severe": "1.75ml"},
    "ha_nasolabial":   {"mild": "0.75ml/側", "moderate": "1.25ml/側", "severe": "1.75ml/側"},
    "ha_jawline":      {"mild": "1.5ml",     "moderate": "2.5ml",     "severe": "3.5ml"},
    "ha_nasolabial2":  {"mild": "0.9ml/側",  "moderate": "1.25ml/側", "severe": "1.75ml/側"},
    "ha_tear":         {"mild": "0.4ml/側",  "moderate": "0.75ml/側", "severe": "1.25ml/側"},
    "ha_skin":         {"mild": "1.5ml",     "moderate": "2.5ml",     "severe": "3.5ml"},
    "sculptra":        {"mild": "1瓶",       "moderate": "2瓶",       "severe": "3瓶"},
    "thread":          {"mild": "5根/側",    "moderate": "8根/側",    "severe": "13根/側"},
    "pn":              {"mild": "1ml",       "moderate": "1.75ml",    "severe": "2.5ml"},
    "picosecond":      {"mild": "1次",       "moderate": "4次",       "severe": "6次"},
    "ultherapy":       {"mild": "300發",     "moderate": "500發",     "severe": "800發"},
    "thermage":        {"mild": "900發",     "moderate": "1200發",    "severe": "1500發"},
}

PROBLEM_ZH = {
    "forehead":   "抬頭紋",
    "glabella":   "眉間紋",
    "crowsfeet":  "魚尾紋",
    "tear":       "淚溝",
    "nasolabial": "法令紋",
    "cheek":      "蘋果肌飽滿度",
    "jawline":    "下頜緣輪廓",
    "chin":       "下巴比例",
    "skin":       "皮膚質地",
    "symmetry":   "臉部對稱性",
}

# Problem → face landmark indices for annotation arrows
PROBLEM_LM = {
    "forehead":   [10, 338, 297],
    "glabella":   [8, 9, 168],
    "crowsfeet":  [33, 263],
    "tear":       [159, 386],
    "nasolabial": [49, 279],
    "cheek":      [117, 346],
    "jawline":    [172, 397],
    "chin":       [152],
    "skin":       [50, 280],
    "symmetry":   [1],
}

PROBLEM_TO_PRODUCTS = {
    "forehead":   [("botox_forehead", True), ("picosecond", False), ("pn", False)],
    "glabella":   [("botox_glabella", True), ("picosecond", False)],
    "crowsfeet":  [("botox_crowsfeet", True), ("thermage", False)],
    "tear":       [("ha_tear", True), ("pn", False)],
    "nasolabial": [("ha_nasolabial", True), ("ha_nasolabial2", True), ("sculptra", False), ("thread", False)],
    "cheek":      [("ha_cheek", True), ("sculptra", False), ("thread", False)],
    "jawline":    [("ha_jawline", True), ("botox_jawline", True), ("ultherapy", False)],
    "chin":       [("ha_chin", True)],
    "skin":       [("ha_skin", True), ("picosecond", False), ("pn", False)],
    "symmetry":   [("botox_glabella", True), ("ha_cheek", False)],
}

# Score threshold - only show significant findings
SCORE_THRESHOLD = 0.20

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose


# ─── Image helpers ───────────────────────────────────────────────────────────
def fix_orientation(pil_img):
    """Auto-rotate based on EXIF data"""
    try:
        exif = pil_img._getexif()
        if exif is None:
            return pil_img
        for tag, val in exif.items():
            if ExifTags.TAGS.get(tag) == "Orientation":
                if val == 3:
                    return pil_img.rotate(180, expand=True)
                elif val == 6:
                    return pil_img.rotate(270, expand=True)
                elif val == 8:
                    return pil_img.rotate(90, expand=True)
    except Exception:
        pass
    return pil_img


def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img):
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def resize_image(img, max_size=800):
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return img


def pil_to_b64(img, fmt="JPEG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── Face detection ──────────────────────────────────────────────────────────
def get_landmarks(img_bgr):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    ) as fm:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = fm.process(rgb)
    if not results.multi_face_landmarks:
        return None, None
    h, w = img_bgr.shape[:2]
    lm = results.multi_face_landmarks[0].landmark
    landmarks = np.array([[l.x * w, l.y * h, l.z * w] for l in lm])
    return landmarks, results


def estimate_yaw(lm):
    nose, le, re = lm[1], lm[33], lm[263]
    ld = abs(nose[0] - le[0])
    rd = abs(nose[0] - re[0])
    total = ld + rd
    if total < 1e-6:
        return 0.0
    return (ld / total - 0.5) * 180.0


# ─── Scoring helpers ─────────────────────────────────────────────────────────
def norm(v, lo, hi):
    if hi - lo < 1e-6:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def sev(v):
    if v < 0.25:
        return "mild"
    elif v < 0.55:
        return "moderate"
    return "severe"


def sev_zh(s):
    return {"mild": "輕度", "moderate": "中度", "severe": "重度"}.get(s, "輕度")


def safe_std(lm, idx):
    return float(np.std(lm[idx, 2]))


# ─── Face Analysis ───────────────────────────────────────────────────────────
def analyze_face(lm, img_bgr):
    h, w = img_bgr.shape[:2]
    R = {}

    # Three zones
    hy = lm[10, 1]
    gy = lm[8, 1]
    ny = lm[94, 1]
    cy = lm[152, 1]
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
        "face_width": round(fw, 1),
        "eye_avg":    round(eavg, 1),
        "five_eye_ratio": round(fw / (5 * eavg), 3),
        "inter_ratio":    round(inter / eavg, 3),
    }

    # Symmetry - using more pairs for robustness
    nx = lm[1, 0]
    sym_pairs = [(33,263),(133,362),(234,454),(61,291),(50,280),(149,378),(58,288),(172,397)]
    asym = sum(
        abs(abs(lm[li,0]-nx) - abs(lm[ri,0]-nx)) /
        max(abs(lm[li,0]-nx), abs(lm[ri,0]-nx), 1e-6)
        for li, ri in sym_pairs
    ) / len(sym_pairs)
    # More sensitive normalization
    asym_n = norm(asym, 0.02, 0.25)
    R["symmetry"] = {"score": round(asym_n, 4), "severity": sev(asym_n),
                     "raw": round(asym, 4), "description": "左右不對稱程度"}

    # Tear trough
    tz = np.mean(lm[[159,145,386,374], 2])
    cz = np.mean(lm[[50,280,205,425], 2])
    tear_raw = abs(tz - cz)
    tear_n = norm(tear_raw, 0.002, 0.018)
    R["tear"] = {"score": round(tear_n, 4), "severity": sev(tear_n),
                 "raw": round(tear_raw, 5), "description": "淚溝凹陷深度"}

    # Nasolabial folds
    nl_z = np.mean(lm[[49,279,48,278], 2])
    mc_z = np.mean(lm[[61,291,62,292], 2])
    nl_depth = abs(nl_z - mc_z)
    ll = np.linalg.norm(lm[49,:2] - lm[61,:2])
    rl = np.linalg.norm(lm[279,:2] - lm[291,:2])
    nl_len = (ll + rl) / 2.0
    depth_n = norm(nl_depth, 0.002, 0.018)
    len_n   = norm(nl_len, fw*0.12, fw*0.28)
    nl_score = 0.55*depth_n + 0.45*len_n
    R["nasolabial"] = {"score": round(nl_score, 4), "severity": sev(nl_score),
                       "raw_depth": round(nl_depth, 5), "raw_length": round(nl_len, 2),
                       "description": "法令紋深度與長度"}

    # Apple cheek fullness
    cbz = np.mean(lm[[117,346,118,347], 2])
    ckz = np.mean(lm[[50,280,205,425], 2])
    apple_raw = abs(cbz - ckz)
    apple_n = 1.0 - norm(apple_raw, 0.002, 0.014)
    R["cheek"] = {"score": round(apple_n, 4), "severity": sev(apple_n),
                  "raw": round(apple_raw, 5), "description": "蘋果肌飽滿度（越高越需填充）"}

    # Jawline definition
    jaw_idx = [172,171,170,169,168,136,135,134,133,58,172,397]
    jaw_pts = lm[jaw_idx, :2]
    jaw_std = np.std(jaw_pts[:, 1])
    jaw_n = norm(jaw_std, 1.5, 20.0)
    R["jawline"] = {"score": round(jaw_n, 4), "severity": sev(jaw_n),
                    "raw": round(jaw_std, 3), "description": "下頜緣輪廓清晰度"}

    # Forehead lines
    fh_std = safe_std(lm, [55,107,66,105,65,52,53,46,124,156,70,63,108,337])
    fh_n = norm(fh_std, 0.0008, 0.009)
    R["forehead"] = {"score": round(fh_n, 4), "severity": sev(fh_n),
                     "raw": round(fh_std, 6), "description": "額頭橫紋深度"}

    # Glabella
    gl_std = safe_std(lm, [168,6,197,195,5,4,8,9,107,336])
    gl_n = norm(gl_std, 0.0008, 0.009)
    R["glabella"] = {"score": round(gl_n, 4), "severity": sev(gl_n),
                     "raw": round(gl_std, 6), "description": "眉間川字紋深度"}

    # Crow's feet
    cr_std = safe_std(lm, [33,133,246,161,160,159,263,362,466,388,387,386,130,359])
    cr_n = norm(cr_std, 0.0008, 0.012)
    R["crowsfeet"] = {"score": round(cr_n, 4), "severity": sev(cr_n),
                      "raw": round(cr_std, 6), "description": "眼外角魚尾紋深度"}

    # Chin ratio
    chin_w = abs(lm[172,0] - lm[397,0]) + 1e-6
    chin_l = abs(lm[152,1] - ny)
    chin_ratio = chin_l / chin_w
    chin_n = norm(chin_ratio, 0.35, 0.95)
    R["chin"] = {"score": round(chin_n, 4), "severity": sev(chin_n),
                 "ratio": round(chin_ratio, 3), "description": "下巴長寬比（偏低=過短）"}

    # Skin texture
    sk_z = lm[[50,280,205,425,117,346,187,411,147,376], 2]
    sk_cv = np.std(sk_z) / (np.mean(np.abs(sk_z)) + 1e-6)
    sk_n = norm(sk_cv, 0.04, 0.55)
    R["skin"] = {"score": round(sk_n, 4), "severity": sev(sk_n),
                 "raw": round(sk_cv, 4), "description": "皮膚粗糙度/質地"}

    return R


# ─── Generate recommendations ────────────────────────────────────────────────
def generate_recs(analysis):
    recs = []
    for prob_key, prob_name in PROBLEM_ZH.items():
        data = analysis.get(prob_key, {})
        if not data or "score" not in data:
            continue
        score = data["score"]
        if score < SCORE_THRESHOLD:
            continue
        sv = data["severity"]
        if prob_key not in PROBLEM_TO_PRODUCTS:
            continue
        primary, alternatives = [], []
        for pk, is_primary in PROBLEM_TO_PRODUCTS[prob_key]:
            if pk not in PRODUCT_ZH:
                continue
            name, brand, layer, method, effect = PRODUCT_ZH[pk]
            dose = DOSE_TABLE[pk].get(sv, "依醫師評估")
            info = {"key": pk, "name": name, "brand": brand,
                    "dose": dose, "layer": layer, "method": method, "effect": effect}
            if is_primary:
                primary.append(info)
            else:
                alternatives.append(info)
        recs.append({"key": prob_key, "name": prob_name, "score": score,
                     "severity": sv, "primary": primary, "alternatives": alternatives,
                     "description": data.get("description", "")})
    return recs


# ─── Draw face landmark annotation ───────────────────────────────────────────
def draw_basic_annotations(img_bgr, lm, analysis):
    """Standard three-zones + five-eyes lines"""
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
    labels = {"top": "髮際", "m1": "眉間", "m2": "鼻下", "bot": "下巴"}
    for k, y in ys.items():
        cv2.line(img, (0, y), (w, y), COLOR, 1)
        cv2.putText(img, labels[k], (5, max(y-4, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1, cv2.LINE_AA)
    for x in [lm[234,0], lm[33,0], lm[133,0], lm[362,0], lm[263,0], lm[454,0]]:
        cv2.line(img, (int(x), 0), (int(x), h), (200, 200, 0), 1)
    zone_vals = [
        (ys["top"], ys["m1"],
         "上庭 {:.1%}".format(zones.get("upper_ratio", 0))),
        (ys["m1"],  ys["m2"],
         "中庭 {:.1%}".format(zones.get("middle_ratio", 0))),
        (ys["m2"],  ys["bot"],
         "下庭 {:.1%}".format(zones.get("lower_ratio", 0))),
    ]
    for y1, y2, text in zone_vals:
        cv2.putText(img, text, (w - 130, (y1+y2)//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR, 1, cv2.LINE_AA)
    for idx in [10, 8, 94, 152, 33, 133, 362, 263, 234, 454, 1]:
        cv2.circle(img, (int(lm[idx,0]), int(lm[idx,1])), 3, (0, 100, 255), -1)
    return img


def draw_treatment_map(img_bgr, lm, recs):
    """
    Draw arrows from face landmark to treatment labels.
    Returns a PIL image.
    """
    h, w = img_bgr.shape[:2]
    pil = cv2_to_pil(img_bgr.copy())
    draw = ImageDraw.Draw(pil)

    # Use a basic font (no Chinese support without font file, use numbers+ASCII)
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_tiny  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font_small = ImageFont.load_default()
        font_tiny  = font_small

    COLORS = {
        "primary": (50, 220, 100),
        "arrow":   (0, 220, 255),
        "box_bg":  (10, 10, 30, 200),
        "text":    (255, 255, 200),
    }

    # Sort by score descending, take top significant ones
    top_recs = sorted([r for r in recs if r["score"] >= 0.25],
                      key=lambda x: x["score"], reverse=True)[:7]

    placed_labels = []

    for idx, rec in enumerate(top_recs):
        prob_key = rec["key"]
        lm_indices = PROBLEM_LM.get(prob_key, [1])
        pts = lm[lm_indices, :2]
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))

        # Draw dot on face
        draw.ellipse([(cx-6, cy-6), (cx+6, cy+6)],
                     fill=(255, 80, 80), outline=(255, 200, 0), width=2)

        # Determine label side
        if cx < w / 2:
            lx = 5
            arrow_start = (cx - 10, cy)
        else:
            lx = w - 185
            arrow_start = (cx + 10, cy)

        # Spread labels vertically to avoid overlap
        ly = cy - 20
        for (px, py, pw, ph) in placed_labels:
            if abs(ly - py) < ph + 4 and abs(lx - px) < pw + 4:
                ly = py + ph + 6

        ly = max(5, min(h - 60, ly))

        # Build label text (ASCII-safe, numbers+short codes)
        if rec["primary"]:
            prod = rec["primary"][0]
            line1 = "[{}] {}".format(idx+1, sev_zh(rec["severity"]))
            line2 = prod["dose"]
            line3 = "(primary)"
        else:
            line1 = "[{}] {}".format(idx+1, sev_zh(rec["severity"]))
            line2 = ""
            line3 = ""

        box_w, box_h = 178, 48

        # Draw box background
        box_img = Image.new("RGBA", (box_w, box_h), (10, 10, 30, 210))
        pil_rgba = pil.convert("RGBA")
        pil_rgba.paste(box_img, (lx, ly), box_img)
        pil = pil_rgba.convert("RGB")
        draw = ImageDraw.Draw(pil)

        # Draw box border
        draw.rectangle([(lx, ly), (lx+box_w, ly+box_h)],
                        outline=(0, 220, 255), width=1)

        # Number badge
        badge_r = 10
        draw.ellipse([(lx+3, ly+3), (lx+3+badge_r*2, ly+3+badge_r*2)],
                     fill=(255, 80, 80))
        draw.text((lx+7, ly+4), str(idx+1), fill=(255,255,255), font=font_tiny)

        # Score bar
        bar_w = int((box_w - 20) * rec["score"])
        draw.rectangle([(lx+18, ly+4), (lx+18+bar_w, ly+10)],
                        fill=(50, 220, 100))

        draw.text((lx+18, ly+12), line1, fill=(255, 220, 50), font=font_small)
        if line2:
            draw.text((lx+18, ly+28), line2, fill=(180, 255, 180), font=font_tiny)
        if line3:
            draw.text((lx+100, ly+28), line3, fill=(100, 200, 255), font=font_tiny)

        placed_labels.append((lx, ly, box_w, box_h))

        # Draw arrow
        arrow_end = (lx + box_w if cx >= w/2 else lx, ly + box_h // 2)
        draw.line([arrow_start, arrow_end], fill=(0, 220, 255), width=2)
        # Arrowhead
        angle = math.atan2(cy - arrow_end[1], cx - arrow_end[0])
        tip = arrow_start
        for da in [0.4, -0.4]:
            ex = int(tip[0] - 12 * math.cos(angle + da))
            ey = int(tip[1] - 12 * math.sin(angle + da))
            draw.line([tip, (ex, ey)], fill=(0, 220, 255), width=2)

    # Add legend at bottom
    legend_y = h - 28
    for i, rec in enumerate(top_recs):
        name = PROBLEM_ZH.get(rec["key"], rec["key"])
        short = "[{}]={}".format(i+1, name[:4])
        draw.text((5 + i * 115, legend_y), short, fill=(255, 220, 100), font=font_tiny)

    return pil


# ─── HTML Report Generator ───────────────────────────────────────────────────
def generate_html_report(analysis, recs, physio_data,
                         img_basic_pil, img_map_pil, timestamp):
    """Generate a beautiful HTML report with embedded images"""

    b64_basic = pil_to_b64(img_basic_pil)
    b64_map   = pil_to_b64(img_map_pil)

    zones = analysis.get("zones", {})
    fe    = analysis.get("five_eyes", {})
    dom_map = {"upper": "上庭", "middle": "中庭", "lower": "下庭"}
    dominant = dom_map.get(zones.get("dominant", ""), "")

    # Score bars for all problems
    scored_html = ""
    all_scored = sorted(
        [(k, v) for k, v in analysis.items()
         if isinstance(v, dict) and "score" in v],
        key=lambda x: x[1]["score"], reverse=True,
    )
    for pk, data in all_scored:
        name = PROBLEM_ZH.get(pk, pk)
        score = data["score"]
        sv = data["severity"]
        sv_zh_str = sev_zh(sv)
        bar_color = {"mild": "#2ecc71", "moderate": "#f1c40f", "severe": "#e74c3c"}.get(sv, "#999")
        pct = int(score * 100)
        desc = data.get("description", "")
        scored_html += """
        <div class="score-row">
          <div class="score-label">{name}</div>
          <div class="score-bar-wrap">
            <div class="score-bar" style="width:{pct}%;background:{color};"></div>
          </div>
          <span class="score-badge badge-{sv}">{sv_zh}</span>
          <span class="score-num">{score:.2f}</span>
          <span class="score-desc">{desc}</span>
        </div>""".format(
            name=name, pct=pct, color=bar_color,
            sv=sv, sv_zh=sv_zh_str, score=score, desc=desc,
        )

    # Treatment cards
    recs_html = ""
    for rec in recs:
        sv = rec["severity"]
        sv_zh_str = sev_zh(sv)
        badge_cls = "badge-" + sv
        primary_cards = ""
        for prod in rec["primary"]:
            primary_cards += """
            <div class="prod-card primary-card">
              <div class="prod-name">{name} <span class="tag-primary">首選</span></div>
              <table class="prod-table">
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
            <div class="prod-card alt-card">
              <div class="prod-name">{name} <span class="tag-alt">備選</span></div>
              <div class="prod-mini">劑量：{dose}｜{effect}</div>
            </div>""".format(**prod)

        recs_html += """
        <div class="rec-section">
          <div class="rec-header">
            <span class="rec-icon">▶</span>
            <span class="rec-name">{name}</span>
            <span class="{badge}">{sv_zh}</span>
            <span class="rec-score">評分 {score:.2f}</span>
          </div>
          <div class="rec-desc">{desc}</div>
          <div class="prod-grid">{primary}</div>
          {alt_section}
        </div>""".format(
            name=rec["name"], badge=badge_cls, sv_zh=sv_zh_str,
            score=rec["score"], desc=rec.get("description", ""),
            primary=primary_cards,
            alt_section=(
                '<div class="alt-label">備選方案</div>'
                '<div class="prod-grid">' + alt_cards + '</div>'
                if alt_cards else ""
            ),
        )

    if not recs_html:
        recs_html = '<div class="no-issues">未偵測到明顯問題，繼續維持良好保養即可！</div>'

    # Physiognomy
    physio_html = ""
    for r in physio_data:
        zone_label = "（{} {}）".format(r["zone"], r["ratio"]) if r.get("zone") else ""
        products_str = "、".join(r.get("products", []))
        physio_html += """
        <div class="physio-card">
          <div class="physio-title">{aspect}{zone}</div>
          <div class="physio-reading"><span class="physio-label">面相解讀：</span>{reading}</div>
          <div class="physio-improve"><span class="physio-label2">醫美改善：</span>{improve}</div>
          <div class="physio-prods">建議產品：{products}</div>
        </div>""".format(
            aspect=r["aspect"], zone=zone_label,
            reading=r["reading"], improve=r["improve"], products=products_str,
        )

    html = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 智能醫美面診報告</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0d1117; color: #e8e8e8; font-family: 'Helvetica Neue', Arial, sans-serif; padding: 20px; }}
.container {{ max-width: 960px; margin: 0 auto; }}
.header {{ text-align: center; padding: 30px 0 20px; border-bottom: 2px solid #e0c44a; margin-bottom: 24px; }}
.header h1 {{ color: #ffd700; font-size: 1.8rem; margin-bottom: 6px; }}
.header .date {{ color: #a0b4c8; font-size: 0.9rem; }}

.section {{ background: #161b22; border-radius: 12px; padding: 20px; margin-bottom: 20px; border-left: 4px solid #e0c44a; }}
.section h2 {{ color: #ffd700; font-size: 1.1rem; margin-bottom: 14px; }}

/* Images */
.img-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.img-box {{ flex: 1; min-width: 280px; }}
.img-box img {{ width: 100%; border-radius: 8px; border: 1px solid #30363d; }}
.img-caption {{ color: #a0b4c8; font-size: 0.82rem; text-align: center; margin-top: 4px; }}

/* Zones metrics */
.zones-row {{ display: flex; gap: 12px; margin-bottom: 12px; }}
.zone-card {{ flex: 1; background: #0d2030; border-radius: 8px; padding: 12px; text-align: center; }}
.zone-card .zone-name {{ color: #a0b4c8; font-size: 0.82rem; }}
.zone-card .zone-val {{ color: #ffd700; font-size: 1.5rem; font-weight: 700; }}
.zone-card .zone-delta {{ font-size: 0.78rem; }}
.zone-dominant {{ border: 2px solid #e0c44a !important; }}
.info-box {{ background: #0d2137; border-radius: 6px; padding: 8px 14px; color: #c0d8f0; font-size: 0.88rem; margin-top: 8px; }}

/* Score rows */
.score-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }}
.score-label {{ min-width: 90px; font-weight: 600; color: #ffffff; font-size: 0.88rem; }}
.score-bar-wrap {{ flex: 1; min-width: 100px; height: 8px; background: #30363d; border-radius: 4px; overflow: hidden; }}
.score-bar {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
.score-num {{ min-width: 36px; color: #a0b4c8; font-size: 0.82rem; text-align: right; }}
.score-desc {{ color: #606878; font-size: 0.76rem; min-width: 120px; }}

/* Badges */
.score-badge, .badge-mild, .badge-moderate, .badge-severe, .tag-primary, .tag-alt {{ padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; white-space: nowrap; }}
.badge-mild    {{ background: #1a3a2a; color: #2ecc71; border: 1px solid #2ecc71; }}
.badge-moderate{{ background: #3a2a0a; color: #f1c40f; border: 1px solid #f1c40f; }}
.badge-severe  {{ background: #3a1a1a; color: #e74c3c; border: 1px solid #e74c3c; }}
.tag-primary   {{ background: #1a3a2a; color: #2ecc71; border: 1px solid #2ecc71; margin-left: 6px; }}
.tag-alt       {{ background: #1a1a3a; color: #8888ff; border: 1px solid #8888ff; margin-left: 6px; }}

/* Rec sections */
.rec-section {{ background: #0f141c; border-radius: 10px; padding: 16px; margin-bottom: 16px; border: 1px solid #21262d; }}
.rec-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }}
.rec-icon {{ color: #e0c44a; }}
.rec-name {{ color: #ffffff; font-weight: 700; font-size: 1rem; }}
.rec-score {{ color: #a0b4c8; font-size: 0.82rem; margin-left: auto; }}
.rec-desc {{ color: #7090a0; font-size: 0.82rem; margin-bottom: 10px; }}
.prod-grid {{ display: flex; gap: 10px; flex-wrap: wrap; }}
.prod-card {{ flex: 1; min-width: 200px; border-radius: 8px; padding: 12px; }}
.primary-card {{ background: #0a1f10; border-left: 3px solid #2ecc71; }}
.alt-card {{ background: #0a0a20; border-left: 3px solid #8888ff; }}
.prod-name {{ color: #ffd700; font-weight: 700; margin-bottom: 8px; }}
.prod-table {{ width: 100%; font-size: 0.82rem; border-collapse: collapse; }}
.prod-table td {{ padding: 3px 4px; color: #c0e8c0; vertical-align: top; }}
.prod-table td:first-child {{ color: #7090a0; width: 70px; white-space: nowrap; }}
.prod-mini {{ color: #8090c0; font-size: 0.8rem; margin-top: 4px; }}
.alt-label {{ color: #8888ff; font-weight: 600; font-size: 0.85rem; margin: 10px 0 6px; }}
.no-issues {{ color: #2ecc71; font-size: 0.9rem; padding: 10px; }}

/* Physio */
.physio-card {{ background: #0f1520; border-radius: 10px; padding: 14px; margin-bottom: 12px; border-left: 3px solid #e0c44a; }}
.physio-title {{ color: #ffd700; font-weight: 700; margin-bottom: 8px; }}
.physio-reading {{ color: #c8d8e8; font-size: 0.88rem; margin-bottom: 6px; }}
.physio-improve {{ background: #0a1f10; border-radius: 6px; padding: 8px; font-size: 0.86rem; margin-bottom: 6px; color: #a0d0a0; }}
.physio-prods {{ color: #7090a0; font-size: 0.8rem; }}
.physio-label {{ color: #ffd700; font-weight: 600; }}
.physio-label2 {{ color: #2ecc71; font-weight: 600; }}

.footer {{ text-align: center; padding: 20px; color: #606878; font-size: 0.8rem; border-top: 1px solid #21262d; margin-top: 20px; }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>AI 智能醫美面診報告</h1>
    <div class="date">分析日期：{timestamp}</div>
  </div>

  <!-- Treatment Map + Basic Annotation -->
  <div class="section">
    <h2>臉部治療標注圖</h2>
    <div class="img-row">
      <div class="img-box">
        <img src="data:image/jpeg;base64,{b64_map}" alt="治療標注圖">
        <div class="img-caption">治療部位標注（數字對應下方建議）</div>
      </div>
      <div class="img-box">
        <img src="data:image/jpeg;base64,{b64_basic}" alt="三庭五眼標注">
        <div class="img-caption">三庭五眼比例標注</div>
      </div>
    </div>
  </div>

  <!-- Three zones -->
  <div class="section">
    <h2>三庭比例分析</h2>
    <div class="zones-row">
      <div class="zone-card {cls_upper}">
        <div class="zone-name">上庭</div>
        <div class="zone-val">{upper:.1%}</div>
        <div class="zone-delta" style="color:{col_upper}">{delta_upper:+.1%}</div>
      </div>
      <div class="zone-card {cls_middle}">
        <div class="zone-name">中庭</div>
        <div class="zone-val">{middle:.1%}</div>
        <div class="zone-delta" style="color:{col_middle}">{delta_middle:+.1%}</div>
      </div>
      <div class="zone-card {cls_lower}">
        <div class="zone-name">下庭</div>
        <div class="zone-val">{lower:.1%}</div>
        <div class="zone-delta" style="color:{col_lower}">{delta_lower:+.1%}</div>
      </div>
    </div>
    <div class="info-box">主導分區：<strong style="color:#ffd700">{dominant}</strong>｜理想各區均為 33.3%</div>
  </div>

  <!-- Five eyes -->
  <div class="section">
    <h2>五眼比例</h2>
    <div class="zones-row">
      <div class="zone-card">
        <div class="zone-name">五眼比例</div>
        <div class="zone-val">{fer:.2f}</div>
        <div class="zone-delta" style="color:{fer_c}">{fer_d:+.2f} 理想=1.0</div>
      </div>
      <div class="zone-card">
        <div class="zone-name">眼距比</div>
        <div class="zone-val">{ir:.2f}</div>
        <div class="zone-delta" style="color:{ir_c}">{ir_d:+.2f} 理想=1.0</div>
      </div>
    </div>
  </div>

  <!-- Scores -->
  <div class="section">
    <h2>問題診斷評分</h2>
    {scored_html}
  </div>

  <!-- Treatments -->
  <div class="section">
    <h2>個人化治療建議</h2>
    {recs_html}
  </div>

  <!-- Physiognomy -->
  <div class="section">
    <h2>面相學分析與醫美改善建議</h2>
    {physio_html}
  </div>

  <div class="footer">
    本報告由 AI 智能醫美面診輔助系統生成，僅供參考，不構成醫療建議。<br>
    實際治療請諮詢合法執照醫師。
  </div>
</div>
</body>
</html>""".format(
        timestamp=timestamp,
        b64_map=b64_map,
        b64_basic=b64_basic,
        upper=zones.get("upper_ratio", 0),
        middle=zones.get("middle_ratio", 0),
        lower=zones.get("lower_ratio", 0),
        delta_upper=zones.get("upper_ratio", 0) - 0.333,
        delta_middle=zones.get("middle_ratio", 0) - 0.333,
        delta_lower=zones.get("lower_ratio", 0) - 0.333,
        cls_upper="zone-dominant" if zones.get("dominant") == "upper" else "",
        cls_middle="zone-dominant" if zones.get("dominant") == "middle" else "",
        cls_lower="zone-dominant" if zones.get("dominant") == "lower" else "",
        col_upper="#2ecc71" if abs(zones.get("upper_ratio",0)-0.333)<0.05 else "#e74c3c",
        col_middle="#2ecc71" if abs(zones.get("middle_ratio",0)-0.333)<0.05 else "#e74c3c",
        col_lower="#2ecc71" if abs(zones.get("lower_ratio",0)-0.333)<0.05 else "#e74c3c",
        dominant=dominant,
        fer=fe.get("five_eye_ratio", 1.0),
        fer_d=fe.get("five_eye_ratio", 1.0) - 1.0,
        fer_c="#2ecc71" if abs(fe.get("five_eye_ratio",1.0)-1.0)<0.1 else "#e74c3c",
        ir=fe.get("inter_ratio", 1.0),
        ir_d=fe.get("inter_ratio", 1.0) - 1.0,
        ir_c="#2ecc71" if abs(fe.get("inter_ratio",1.0)-1.0)<0.15 else "#e74c3c",
        scored_html=scored_html,
        recs_html=recs_html,
        physio_html=physio_html,
    )
    return html


# ─── Physiognomy ─────────────────────────────────────────────────────────────
def physio_analysis(analysis):
    readings = []
    zones = analysis.get("zones", {})
    dominant = zones.get("dominant", "")
    zone_data = [
        ("upper", "上庭", zones.get("upper_ratio", 0.333),
         "上庭寬闊，早年運佳，智慧過人，父母緣深厚",
         "若額頭有皺紋或過窄，可透過肉毒放鬆額肌改善抬頭紋，或以玻尿酸填充額頭弧度，讓上庭更飽滿圓潤，提升整體氣場。",
         ["肉毒毒素（抬頭紋）", "玻尿酸（全臉保濕）"]),
        ("middle", "中庭", zones.get("middle_ratio", 0.333),
         "中庭均衡，中年事業運旺，適合創業或管理職",
         "若鼻樑較低或法令紋明顯，可透過玻尿酸填充，讓中庭比例更完美，面相上增強事業運與威嚴感。",
         ["玻尿酸（法令紋）", "玻尿酸（法令紋II）"]),
        ("lower", "下庭", zones.get("lower_ratio", 0.333),
         "下庭豐厚，晚年運佳，福氣深厚，子孫有緣",
         "若下巴短或下頜緣不清晰，可透過玻尿酸墊下巴或埋線提拉，讓下庭比例更協調，加強晚年財運與福氣。",
         ["玻尿酸（下巴）", "玻尿酸（下頜輪廓）"]),
    ]
    for zone_key, zone_name, ratio, good, improve, products in zone_data:
        readings.append({
            "aspect": zone_name + "分析",
            "zone": zone_name,
            "ratio": "{:.1%}".format(ratio),
            "reading": good if dominant == zone_key else zone_name + "比例偏低，建議調整",
            "improve": improve,
            "products": products,
            "is_dominant": (dominant == zone_key),
        })

    inter_r = analysis.get("five_eyes", {}).get("inter_ratio", 1.0)
    if inter_r < 0.85:
        readings.append({
            "aspect": "眼距偏窄", "zone": "", "ratio": "{:.2f}".format(inter_r),
            "reading": "眼距較窄，個性敏銳反應快，容易急躁",
            "improve": "可透過眉形調整或淚溝填充，視覺上拉寬眼距，讓面部比例更協調。",
            "products": ["玻尿酸（淚溝）"], "is_dominant": False,
        })
    elif inter_r > 1.2:
        readings.append({
            "aspect": "眼距寬廣", "zone": "", "ratio": "{:.2f}".format(inter_r),
            "reading": "眼距寬廣，心胸寬大，人緣佳，適合公關外交",
            "improve": "眼距已相當理想，可搭配眼周保養或皮秒改善眼周膚質。",
            "products": ["皮秒雷射"], "is_dominant": True,
        })

    if analysis.get("nasolabial", {}).get("score", 0) > 0.45:
        readings.append({
            "aspect": "法令紋顯現", "zone": "", "ratio": "",
            "reading": "法令紋深象徵威嚴與領導力，主掌大局之相",
            "improve": "法令紋象徵威嚴，但視覺顯老。可透過玻尿酸填充或舒顏萃刺激膠原新生，保持命格威嚴同時外觀更年輕。",
            "products": ["玻尿酸（法令紋）", "舒顏萃Sculptra"], "is_dominant": False,
        })
    return readings


# ─── Calf analysis ───────────────────────────────────────────────────────────
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
        if abs(lk.y - la.y) * h < 10 or abs(rk.y - ra.y) * h < 10:
            return R
        calf_w = abs(lk.x - rk.x) * w
        calf_ratio = calf_w / (w * 0.6 + 1e-6)
        score = norm(float(calf_ratio), 0.2, 0.6)
        R = {
            "detected": True,
            "score": round(score, 3),
            "severity": sev(score),
            "calf_ratio": round(float(calf_ratio), 3),
            "left_knee":  (int(lk.x * w), int(lk.y * h)),
            "left_ankle": (int(la.x * w), int(la.y * h)),
            "right_knee": (int(rk.x * w), int(rk.y * h)),
            "right_ankle":(int(ra.x * w), int(ra.y * h)),
        }
    except Exception:
        pass
    return R


# ─── Back analysis ────────────────────────────────────────────────────────────
def analyze_back(img_front_bgr, img_side_bgr):
    R = {"detected": False, "score": 0.0, "severity": "mild",
         "back_width_ratio": 0.0, "shoulder_width_px": 0.0,
         "hip_width_px": 0.0, "shoulder_hip_ratio": 0.0, "back_thickness_ratio": 0.0}
    try:
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            res = pose.process(cv2.cvtColor(img_front_bgr, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return R
        h, w = img_front_bgr.shape[:2]
        lm = res.pose_landmarks.landmark
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lhip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        rhip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        sw = abs(ls.x - rs.x) * w
        hw = abs(lhip.x - rhip.x) * w
        bwr = sw / (w * 0.8 + 1e-6)
        score = norm(bwr, 0.3, 0.7)
        R = {"detected": True, "score": round(score, 3), "severity": sev(score),
             "back_width_ratio": round(float(bwr), 3),
             "shoulder_width_px": round(float(sw), 1),
             "hip_width_px": round(float(hw), 1),
             "shoulder_hip_ratio": round(float(sw / (hw + 1e-6)), 3),
             "back_thickness_ratio": 0.0}
        if img_side_bgr is not None:
            with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
                res2 = pose.process(cv2.cvtColor(img_side_bgr, cv2.COLOR_BGR2RGB))
            if res2.pose_landmarks:
                h2, w2 = img_side_bgr.shape[:2]
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
.stApp { background: #0d1117; }
h1, h2, h3, h4 { color: #ffd700 !important; }
p, li { color: #e8e8e8 !important; }
.stMarkdown p { color: #e8e8e8 !important; }
label, .stRadio label { color: #e8e8e8 !important; font-weight: 600; }
.stCaption { color: #a0b4c8 !important; }
.ulabel { font-size: 0.95rem; font-weight: 700; color: #ffd700; margin-bottom: 6px; display: block; }
.aok  { color: #2ecc71; font-size: 0.88rem; font-weight: 600; }
.afail{ color: #e74c3c; font-size: 0.88rem; font-weight: 600; }
.bmild{ background:#1a3a2a; color:#2ecc71; border:1px solid #2ecc71; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.bmod { background:#3a2a0a; color:#ffcc00; border:1px solid #ffcc00; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.bsev { background:#3a1a1a; color:#ff6060; border:1px solid #ff6060; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.stButton > button { background: linear-gradient(135deg,#e0c44a,#c0a030) !important; color:#000 !important; font-weight:700 !important; border-radius:8px !important; border:none !important; }
div[data-testid="metric-container"] label { color:#a0b4c8 !important; }
div[data-testid="metric-container"] [data-testid="metric-value"] { color:#ffd700 !important; }
div[data-testid="stInfo"] p { color:#c0d8f0 !important; }
div[data-testid="stSuccess"] p { color:#c0f0c0 !important; }
div[data-testid="stWarning"] p { color:#f0d080 !important; }
.stExpander summary p { color:#ffd700 !important; font-weight:600; }
</style>
""", unsafe_allow_html=True)


def badge_html(sv):
    cls = {"mild": "bmild", "moderate": "bmod", "severe": "bsev"}.get(sv, "bmild")
    return "<span class='{}'>{}</span>".format(cls, sev_zh(sv))


# ─── Main App ────────────────────────────────────────────────────────────────
def main():
    apply_styles()
    st.title("AI 智能醫美面診輔助系統")
    st.caption("上傳照片，AI 自動分析並生成個人化 HTML 治療建議報告")
    st.markdown("---")

    mode = st.radio("**選擇分析模式**",
                    ["臉部分析", "小腿肌肉分析", "背部分析"], horizontal=True)

    # ══════════════════ 臉部分析 ══════════════════
    if mode == "臉部分析":
        st.markdown("---")
        st.subheader("上傳臉部照片（最多5個角度）")
        st.info("請上傳：正面、左側45度、左側90度、右側45度、右側90度（至少需要正面）")

        ANGLES = {
            "front":   {"expected": 0,   "tol": 15, "label": "① 正面（0度）"},
            "left45":  {"expected": -35, "tol": 18, "label": "② 左側45度"},
            "left90":  {"expected": -70, "tol": 20, "label": "③ 左側90度"},
            "right45": {"expected": 35,  "tol": 18, "label": "④ 右側45度"},
            "right90": {"expected": 70,  "tol": 20, "label": "⑤ 右側90度"},
        }

        c1, c2, c3, c4, c5 = st.columns(5)
        cols = {"front": c1, "left45": c2, "left90": c3,
                "right45": c4, "right90": c5}
        uploads = {}

        for key, col in cols.items():
            with col:
                st.markdown("<span class='ulabel'>{}</span>".format(
                    ANGLES[key]["label"]), unsafe_allow_html=True)
                f = st.file_uploader("", type=["jpg","jpeg","png"],
                                     key=key, label_visibility="collapsed")
                if f:
                    pil = Image.open(f).convert("RGB")
                    pil = fix_orientation(pil)
                    uploads[key] = pil
                    st.image(pil, use_container_width=True)

        if uploads:
            st.markdown("---")
            st.subheader("角度驗證")
            vcols = st.columns(len(uploads))
            for i, (key, pil_img) in enumerate(uploads.items()):
                cfg = ANGLES[key]
                img_bgr = pil_to_cv2(resize_image(pil_img, 640))
                lm, _ = get_landmarks(img_bgr)
                with vcols[i]:
                    if lm is None:
                        st.markdown("<div class='afail'>無法偵測人臉</div>",
                                    unsafe_allow_html=True)
                    else:
                        yaw = estimate_yaw(lm)
                        ok = abs(yaw - cfg["expected"]) <= cfg["tol"]
                        cls = "aok" if ok else "afail"
                        icon = "OK" if ok else "角度偏差"
                        st.markdown(
                            "<div class='{}'>{}<br>Yaw={:.1f}</div>".format(
                                cls, icon, yaw),
                            unsafe_allow_html=True)

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
                        recs = generate_recs(analysis)
                        physio = physio_analysis(analysis)
                        timestamp = datetime.now().strftime("%Y年%m月%d日 %H:%M")

                        # Generate images
                        img_basic = draw_basic_annotations(front_bgr, lm_front, analysis)
                        img_map = draw_treatment_map(front_bgr, lm_front, recs)

                        img_basic_pil = cv2_to_pil(img_basic)
                        img_map_pil = img_map  # already PIL

                        st.success("分析完成！")

                        # Show images in app
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.subheader("治療部位標注圖")
                            st.image(img_map_pil, use_container_width=True)
                        with col_b:
                            st.subheader("三庭五眼標注圖")
                            st.image(img_basic_pil, use_container_width=True)

                        # Zones
                        st.subheader("三庭比例")
                        zones = analysis["zones"]
                        zc1, zc2, zc3 = st.columns(3)
                        dom_map = {"upper": "上庭", "middle": "中庭", "lower": "下庭"}
                        zc1.metric("上庭", "{:.1%}".format(zones["upper_ratio"]),
                                   "{:+.1%}".format(zones["upper_ratio"]-0.333))
                        zc2.metric("中庭", "{:.1%}".format(zones["middle_ratio"]),
                                   "{:+.1%}".format(zones["middle_ratio"]-0.333))
                        zc3.metric("下庭", "{:.1%}".format(zones["lower_ratio"]),
                                   "{:+.1%}".format(zones["lower_ratio"]-0.333))
                        st.info("主導分區：{} ｜ 理想各區均為33.3%".format(
                            dom_map.get(zones["dominant"], zones["dominant"])))

                        # Scores
                        st.subheader("問題診斷評分")
                        all_scored = sorted(
                            [(k, v) for k, v in analysis.items()
                             if isinstance(v, dict) and "score" in v],
                            key=lambda x: x[1]["score"], reverse=True)
                        for pk, data in all_scored:
                            name = PROBLEM_ZH.get(pk, pk)
                            sv_val = data["severity"]
                            sv_cls = {"mild":"bmild","moderate":"bmod","severe":"bsev"}.get(sv_val,"bmild")
                            desc = data.get("description", "")
                            st.markdown(
                                "<div style='display:flex;align-items:center;gap:10px;"
                                "margin-bottom:4px;'>"
                                "<span style='color:#fff;font-weight:700;min-width:90px;"
                                "font-size:0.9rem;'>{}</span>"
                                "<span class='{}'>{}</span>"
                                "<span style='color:#aaa;font-size:0.8rem;flex:1;margin-left:6px;'>{}</span>"
                                "<span style='color:#ffd700;font-size:0.88rem;font-weight:700;"
                                "min-width:36px;text-align:right;'>{:.2f}</span>"
                                "</div>".format(
                                    name, sv_cls, sev_zh(sv_val), desc, data["score"]),
                                unsafe_allow_html=True)
                            st.progress(data["score"])

                        # Recs in app
                        st.subheader("個人化治療建議")
                        if recs:
                            for rec in recs:
                                label = "{} — {} ({:.2f})".format(
                                    rec["name"], sev_zh(rec["severity"]), rec["score"])
                                with st.expander(label, expanded=rec["score"] > 0.45):
                                    if rec["primary"]:
                                        st.markdown(
                                            "<div style='color:#2ecc71;font-weight:700;"
                                            "margin-bottom:6px;'>首選治療方案</div>",
                                            unsafe_allow_html=True)
                                        for prod in rec["primary"]:
                                            st.markdown(
                                                "<div style='background:#0a1f10;border-radius:8px;"
                                                "padding:14px;margin:6px 0;border-left:4px solid #2ecc71;'>"
                                                "<div style='color:#ffd700;font-weight:700;margin-bottom:6px;'>"
                                                "{name}</div>"
                                                "<div style='color:#c0e8c0;font-size:0.88rem;line-height:2;'>"
                                                "品牌：{brand}<br>建議劑量：<strong style='color:#ffd700'>{dose}</strong><br>"
                                                "注射層次：{layer}<br>注射方式：{method}<br>預估效果：{effect}"
                                                "</div></div>".format(**prod),
                                                unsafe_allow_html=True)
                                    if rec["alternatives"]:
                                        st.markdown(
                                            "<div style='color:#8888ff;font-weight:700;"
                                            "margin:10px 0 6px;'>備選方案</div>",
                                            unsafe_allow_html=True)
                                        acols = st.columns(min(len(rec["alternatives"]), 3))
                                        for j, prod in enumerate(rec["alternatives"]):
                                            with acols[j % len(acols)]:
                                                st.markdown(
                                                    "<div style='background:#0a0a20;border-radius:8px;"
                                                    "padding:10px;border-left:3px solid #8888ff;'>"
                                                    "<div style='color:#c0c8ff;font-weight:600;'>{name}</div>"
                                                    "<div style='color:#8090c0;font-size:0.8rem;'>"
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
                                "<div style='background:#1a1a2e;border-radius:10px;padding:16px;"
                                "margin-bottom:14px;border-left:4px solid #e0c44a;'>"
                                "<div style='color:#ffd700;font-weight:700;font-size:1rem;margin-bottom:8px;'>"
                                "{}{}</div>"
                                "<div style='background:#0f1520;border-radius:6px;padding:10px;margin-bottom:8px;'>"
                                "<span style='color:#ffd700;font-size:0.85rem;font-weight:600;'>面相解讀：</span>"
                                "<span style='color:#c8d8e8;font-size:0.88rem;'>{}</span></div>"
                                "<div style='background:#0f2015;border-radius:6px;padding:10px;'>"
                                "<span style='color:#2ecc71;font-size:0.85rem;font-weight:600;'>醫美改善建議：</span>"
                                "<span style='color:#a0d0a0;font-size:0.88rem;'>{}</span><br>"
                                "<span style='color:#7090a0;font-size:0.82rem;'>建議產品：{}</span>"
                                "</div></div>".format(
                                    r["aspect"], zone_label, r["reading"],
                                    r["improve"], "、".join(r.get("products", []))),
                                unsafe_allow_html=True)

                        # HTML Report download
                        st.subheader("下載精美 HTML 報告")
                        html_report = generate_html_report(
                            analysis, recs, physio,
                            img_basic_pil, img_map_pil, timestamp)

                        st.download_button(
                            "下載完整 HTML 報告（含圖片）",
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

    # ══════════════════ 小腿 ══════════════════
    elif mode == "小腿肌肉分析":
        st.markdown("---")
        st.subheader("小腿肌肉肥大分析")
        st.info("拍攝要求：全身正面站立，膝蓋至腳踝完整入鏡，光線充足。")

        col1, col2 = st.columns(2)
        calf_normal = None
        calf_tiptoe = None
        with col1:
            st.markdown("<span class='ulabel'>① 自然站姿（正面）</span>", unsafe_allow_html=True)
            f1 = st.file_uploader("", type=["jpg","jpeg","png"],
                                  key="calf_n", label_visibility="collapsed")
            if f1:
                calf_normal = fix_orientation(Image.open(f1).convert("RGB"))
                st.image(calf_normal, use_container_width=True)
        with col2:
            st.markdown("<span class='ulabel'>② 墊腳尖（選填）</span>", unsafe_allow_html=True)
            f2 = st.file_uploader("", type=["jpg","jpeg","png"],
                                  key="calf_t", label_visibility="collapsed")
            if f2:
                calf_tiptoe = fix_orientation(Image.open(f2).convert("RGB"))
                st.image(calf_tiptoe, use_container_width=True)

        if calf_normal:
            if st.button("開始小腿分析", use_container_width=True):
                with st.spinner("分析中..."):
                    try:
                        img_bgr = pil_to_cv2(resize_image(calf_normal, 800))
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
                            sv_val = result["severity"]
                            sv_cls = {"mild":"bmild","moderate":"bmod","severe":"bsev"}.get(sv_val,"bmild")
                            st.markdown(
                                "<div style='background:#1a1a2e;border-radius:10px;padding:16px;"
                                "border-left:4px solid #e0c44a;margin:12px 0;'>"
                                "<div style='color:#ffd700;font-size:1.1rem;font-weight:700;margin-bottom:8px;'>"
                                "小腿肌肉肥大程度：<span class='{}'>{}</span></div>"
                                "<div style='color:#c0d0e0;'>寬度比例：{:.3f} ｜ AI評分：{:.2f}/1.00</div>"
                                "</div>".format(sv_cls, sev_zh(sv_val),
                                               result["calf_ratio"], result["score"]),
                                unsafe_allow_html=True)
                            st.progress(result["score"])

                            dose_map = {
                                "mild": "65U/側", "moderate": "90U/側", "severe": "125U/側"
                            }
                            dose = dose_map.get(sv_val, "依醫師評估")
                            name, brand, layer, method, effect = PRODUCT_ZH["botox_calf"]
                            st.subheader("小腿肉毒毒素治療建議")
                            st.markdown(
                                "<div style='background:#0a1f10;border-radius:10px;padding:16px;"
                                "border-left:4px solid #2ecc71;'>"
                                "<div style='color:#ffd700;font-size:1.05rem;font-weight:700;margin-bottom:10px;'>"
                                "腓腸肌肉毒毒素注射</div>"
                                "<div style='color:#c0e8c0;font-size:0.9rem;line-height:2;'>"
                                "品牌：{}<br>"
                                "建議劑量（{}）：<strong style='color:#ffd700'>{}</strong><br>"
                                "注射層次：{}<br>注射方式：{}<br>預估效果：{}"
                                "</div></div>".format(
                                    brand, sev_zh(sv_val), dose, layer, method, effect),
                                unsafe_allow_html=True)

                            for tip in [
                                "治療前2週停止高強度腿部訓練",
                                "注射後4-6小時內請勿按摩注射部位",
                                "注射後24小時內避免劇烈運動、三溫暖及飲酒",
                                "建議每6-9個月維持一次療程",
                                "可搭配小腿拉伸運動加速放鬆效果",
                            ]:
                                st.markdown(
                                    "<div style='color:#c0d0e0;padding:3px 0;'>• {}</div>".format(tip),
                                    unsafe_allow_html=True)

                            if calf_tiptoe:
                                st.subheader("墊腳尖對比分析")
                                tip_bgr = pil_to_cv2(resize_image(calf_tiptoe, 800))
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
                                if result["detected"] and tip_result.get("detected"):
                                    diff = tip_result["score"] - result["score"]
                                    if diff > 0.1:
                                        st.info("墊腳尖時肌肉明顯收縮，腓腸肌活躍度高，建議劑量可略增加。")
                                    else:
                                        st.info("兩姿勢差異不大，以靜態肌肉增大為主，標準劑量即可。")
                        else:
                            st.warning("無法偵測小腿關鍵點，請確認照片包含膝蓋至腳踝完整範圍。")

                    except Exception as e:
                        st.error("分析錯誤：{}".format(e))
                        st.exception(e)
        else:
            st.info("請上傳自然站姿照片以開始分析。")

    # ══════════════════ 背部 ══════════════════
    elif mode == "背部分析":
        st.markdown("---")
        st.subheader("背部肌肉與輪廓分析")
        st.info("請上傳：正背面（背對鏡頭站立）及側背面（側面站立，選填）")

        col1, col2 = st.columns(2)
        back_front = None
        back_side = None
        with col1:
            st.markdown("<span class='ulabel'>① 正背面（背對鏡頭）</span>", unsafe_allow_html=True)
            bf = st.file_uploader("", type=["jpg","jpeg","png"],
                                  key="back_f", label_visibility="collapsed")
            if bf:
                back_front = fix_orientation(Image.open(bf).convert("RGB"))
                st.image(back_front, use_container_width=True)
        with col2:
            st.markdown("<span class='ulabel'>② 側背面（選填）</span>", unsafe_allow_html=True)
            bs = st.file_uploader("", type=["jpg","jpeg","png"],
                                  key="back_s", label_visibility="collapsed")
            if bs:
                back_side = fix_orientation(Image.open(bs).convert("RGB"))
                st.image(back_side, use_container_width=True)

        if back_front:
            if st.button("開始背部分析", use_container_width=True):
                with st.spinner("分析中..."):
                    try:
                        front_bgr = pil_to_cv2(resize_image(back_front, 800))
                        side_bgr = pil_to_cv2(resize_image(back_side, 800)) if back_side else None
                        back_result = analyze_back(front_bgr, side_bgr)

                        st.success("分析完成！")

                        if back_result["detected"]:
                            sv_val = back_result["severity"]
                            sv_cls = {"mild":"bmild","moderate":"bmod","severe":"bsev"}.get(sv_val,"bmild")
                            st.markdown(
                                "<div style='background:#1a1a2e;border-radius:10px;padding:16px;"
                                "border-left:4px solid #e0c44a;margin:12px 0;'>"
                                "<div style='color:#ffd700;font-size:1.1rem;font-weight:700;margin-bottom:8px;'>"
                                "背部評估：<span class='{}'>{}</span></div>"
                                "<div style='color:#c0d0e0;'>"
                                "背部寬度比例：{:.2f} ｜ 肩臀比：{:.2f}</div>"
                                "</div>".format(
                                    sv_cls, sev_zh(sv_val),
                                    back_result.get("back_width_ratio", 0),
                                    back_result.get("shoulder_hip_ratio", 0)),
                                unsafe_allow_html=True)

                            bc1, bc2, bc3 = st.columns(3)
                            bc1.metric("背部寬度比例", "{:.2f}".format(back_result.get("back_width_ratio", 0)))
                            bc2.metric("肩膀寬度", "{:.0f}px".format(back_result.get("shoulder_width_px", 0)))
                            bc3.metric("肩臀比", "{:.2f}".format(back_result.get("shoulder_hip_ratio", 0)))
                            if back_result.get("back_thickness_ratio", 0) > 0:
                                st.metric("側面背部厚度比例", "{:.2f}".format(back_result["back_thickness_ratio"]))

                            st.subheader("背部醫美治療建議")
                            score = back_result.get("score", 0)
                            if score > 0.5:
                                recs_back = [
                                    ("背部肉毒毒素（豎脊肌放鬆）", "針對背部豎脊肌過度發達，放鬆肌肉改善背部寬厚感", "每側50-100U", "2-4週改善，維持4-6個月"),
                                    ("背部溶脂針", "針對背部脂肪堆積，溶解局部脂肪讓背部線條俐落", "4-8ml/療程", "4-8週改善，效果持久"),
                                    ("背部音波拉提", "改善背部皮膚鬆弛，緊緻背部輪廓", "300-500發", "3-6月顯效，維持12-18個月"),
                                ]
                            else:
                                recs_back = [
                                    ("背部水光注射", "改善背部皮膚乾燥粗糙，提升膚質", "2-4ml", "即時保濕，維持4-6個月"),
                                    ("背部皮秒雷射", "改善背部色斑、毛孔粗大", "1-3次療程", "每4-6週一次"),
                                    ("背部肉毒毒素（局部雕塑）", "局部肌肉精細雕塑", "每側30-50U", "2-4週改善，維持4-6個月"),
                                ]
                            for name, desc, dose, effect in recs_back:
                                st.markdown(
                                    "<div style='background:#0d1a20;border-radius:8px;padding:14px;"
                                    "margin:8px 0;border-left:4px solid #e0c44a;'>"
                                    "<div style='color:#ffd700;font-weight:700;margin-bottom:6px;'>{}</div>"
                                    "<div style='color:#c0d0e0;font-size:0.88rem;margin-bottom:4px;'>{}</div>"
                                    "<div style='color:#90a0b0;font-size:0.84rem;'>劑量：{} ｜ 效果：{}</div>"
                                    "</div>".format(name, desc, dose, effect),
                                    unsafe_allow_html=True)
                        else:
                            st.warning("無法偵測背部關鍵點，請確認照片清晰，背對鏡頭站立，完整上半身入鏡。")

                    except Exception as e:
                        st.error("分析錯誤：{}".format(e))
                        st.exception(e)
        else:
            st.info("請上傳正背面照片以開始分析。")

        st.markdown("---")
        st.warning("免責聲明：本系統為輔助參考工具，不構成醫療建議，實際治療請諮詢合法執照醫師。")


if __name__ == "__main__":
    main()
