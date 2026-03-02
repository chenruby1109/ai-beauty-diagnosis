import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import base64
import io
from datetime import datetime
from PIL import Image
from scipy.spatial import distance

st.set_page_config(
    page_title="AI Beauty System",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PRODUCT_DB = {
    "botox_forehead": {"category": "botox", "brands": ["Botox A", "Botox B", "Botox C"], "dose_mid": {"mild": "12U", "moderate": "20U", "severe": "32U"}, "layer": "frontalis muscle", "method": "multi-point injection", "effect": "3-7 days, lasts 4-6 months"},
    "botox_glabella": {"category": "botox", "brands": ["Botox A", "Botox B"], "dose_mid": {"mild": "12U", "moderate": "20U", "severe": "32U"}, "layer": "corrugator muscle", "method": "5-point injection", "effect": "3-7 days, lasts 4-6 months"},
    "botox_crowsfeet": {"category": "botox", "brands": ["Botox A", "Botox B"], "dose_mid": {"mild": "7U/side", "moderate": "12U/side", "severe": "17U/side"}, "layer": "orbicularis oculi", "method": "fan-shaped injection", "effect": "5-7 days, lasts 3-5 months"},
    "botox_jawline": {"category": "botox", "brands": ["Botox A", "Botox B"], "dose_mid": {"mild": "15U", "moderate": "25U", "severe": "40U"}, "layer": "platysma", "method": "linear injection", "effect": "4-6 months"},
    "botox_masseter": {"category": "botox", "brands": ["Botox A", "Botox B"], "dose_mid": {"mild": "25U/side", "moderate": "35U/side", "severe": "50U/side"}, "layer": "masseter deep layer", "method": "fixed-point injection", "effect": "2-4 weeks, lasts 6-12 months"},
    "botox_calf": {"category": "botox", "brands": ["Botox A", "Botox B", "Botox C"], "dose_mid": {"mild": "65U/side", "moderate": "90U/side", "severe": "125U/side"}, "layer": "gastrocnemius medial head", "method": "grid multi-point injection", "effect": "4-8 weeks, lasts 6-9 months"},
    "ha_cheek": {"category": "hyaluronic acid", "brands": ["Juvederm VOLUMA"], "dose_mid": {"mild": "0.75ml/side", "moderate": "1.25ml/side", "severe": "1.75ml/side"}, "layer": "subperiosteal or deep fat", "method": "fan/linear", "effect": "lasts 18-24 months"},
    "ha_chin": {"category": "hyaluronic acid", "brands": ["Juvederm VOLUMA"], "dose_mid": {"mild": "0.75ml", "moderate": "1.25ml", "severe": "1.75ml"}, "layer": "subperiosteal", "method": "single point or fan", "effect": "lasts 12-18 months"},
    "ha_nasolabial": {"category": "hyaluronic acid", "brands": ["Juvederm VOLUMA"], "dose_mid": {"mild": "0.75ml/side", "moderate": "1.25ml/side", "severe": "1.75ml/side"}, "layer": "deep subcutaneous", "method": "retrograde linear + fan", "effect": "60-80% reduction, lasts 12-18 months"},
    "ha_jawline": {"category": "hyaluronic acid", "brands": ["Juvederm VOLUX"], "dose_mid": {"mild": "1.5ml", "moderate": "2.5ml", "severe": "3.5ml"}, "layer": "subperiosteal", "method": "linear along mandible", "effect": "lasts 18-24 months"},
    "ha_nasolabial2": {"category": "hyaluronic acid", "brands": ["Juvederm VOLIFT"], "dose_mid": {"mild": "0.9ml/side", "moderate": "1.25ml/side", "severe": "1.75ml/side"}, "layer": "deep dermis to subcutaneous", "method": "retrograde linear + fern technique", "effect": "lasts 12-15 months"},
    "ha_tear": {"category": "hyaluronic acid", "brands": ["Juvederm VOLBELLA"], "dose_mid": {"mild": "0.4ml/side", "moderate": "0.75ml/side", "severe": "1.25ml/side"}, "layer": "preseptal fat or subperiosteal", "method": "micro multi-point", "effect": "lasts 9-12 months"},
    "ha_skin": {"category": "hyaluronic acid", "brands": ["Juvederm VOLITE"], "dose_mid": {"mild": "1.5ml", "moderate": "2.5ml", "severe": "3.5ml"}, "layer": "mid dermis", "method": "multi-point or gun-assisted", "effect": "lasts 6-9 months"},
    "sculptra": {"category": "collagen stimulator", "brands": ["Sculptra"], "dose_mid": {"mild": "1 vial", "moderate": "2 vials", "severe": "3 vials"}, "layer": "deep dermis to subcutaneous", "method": "fan injection with massage", "effect": "shows in 2-3 months, lasts 18-24 months"},
    "thread": {"category": "thread lift", "brands": ["PDO Thread"], "dose_mid": {"mild": "5 threads/side", "moderate": "8 threads/side", "severe": "13 threads/side"}, "layer": "SMAS/deep subcutaneous", "method": "retrograde barbed thread", "effect": "immediate lift, lasts 12-18 months"},
    "pn": {"category": "PN therapy", "brands": ["Rejuran PN 1%"], "dose_mid": {"mild": "1ml", "moderate": "1.75ml", "severe": "2.5ml"}, "layer": "mid dermis", "method": "micro multi-point", "effect": "brightening, recommend 3-4 sessions"},
    "picosecond": {"category": "laser", "brands": ["Picosecond Laser"], "dose_mid": {"mild": "1 session", "moderate": "4 sessions", "severe": "6 sessions"}, "layer": "epidermis to dermis", "method": "full face scan", "effect": "even skin tone, every 4-6 weeks"},
    "ultherapy": {"category": "energy device", "brands": ["Ultherapy"], "dose_mid": {"mild": "300 shots", "moderate": "500 shots", "severe": "800 shots"}, "layer": "SMAS + dermis", "method": "linear scan, layered delivery", "effect": "shows in 3-6 months, lasts 12-18 months"},
    "thermage": {"category": "energy device", "brands": ["Thermage FLX"], "dose_mid": {"mild": "900 shots", "moderate": "1200 shots", "severe": "1500 shots"}, "layer": "deep dermis to subcutaneous", "method": "full face even scan", "effect": "immediate tightening, lasts 12-24 months"},
}

# Chinese display names for products
PRODUCT_NAMES_ZH = {
    "botox_forehead": "肉毒毒素（抬頭紋）",
    "botox_glabella": "肉毒毒素（眉間紋）",
    "botox_crowsfeet": "肉毒毒素（魚尾紋）",
    "botox_jawline": "肉毒毒素（下頜緣）",
    "botox_masseter": "肉毒毒素（咬肌）",
    "botox_calf": "肉毒毒素（小腿腓腸肌）",
    "ha_cheek": "玻尿酸（蘋果肌）",
    "ha_chin": "玻尿酸（下巴）",
    "ha_nasolabial": "玻尿酸（法令紋 VOLUMA）",
    "ha_jawline": "玻尿酸（下頜輪廓 VOLUX）",
    "ha_nasolabial2": "玻尿酸（法令紋 VOLIFT）",
    "ha_tear": "玻尿酸（淚溝 VOLBELLA）",
    "ha_skin": "玻尿酸（全臉保濕 VOLITE）",
    "sculptra": "舒顏萃 Sculptra",
    "thread": "鳳凰埋線",
    "pn": "麗珠蘭 PN 1%",
    "picosecond": "皮秒雷射",
    "ultherapy": "音波拉提",
    "thermage": "電波拉提",
}

PROBLEM_TO_PRODUCTS = {
    "forehead": [("botox_forehead", True), ("picosecond", False), ("pn", False)],
    "glabella": [("botox_glabella", True), ("picosecond", False)],
    "crowsfeet": [("botox_crowsfeet", True), ("thermage", False), ("ha_tear", False)],
    "tear": [("ha_tear", True), ("pn", False)],
    "nasolabial": [("ha_nasolabial", True), ("ha_nasolabial2", True), ("sculptra", False), ("thread", False)],
    "cheek": [("ha_cheek", True), ("sculptra", False), ("thread", False)],
    "jawline": [("ha_jawline", True), ("botox_jawline", True), ("ultherapy", False)],
    "chin": [("ha_chin", True)],
    "skin": [("ha_skin", True), ("picosecond", False), ("pn", False)],
    "symmetry": [("botox_glabella", True), ("ha_cheek", False)],
}

PROBLEM_NAMES_ZH = {
    "forehead": "抬頭紋",
    "glabella": "眉間紋",
    "crowsfeet": "魚尾紋",
    "tear": "淚溝",
    "nasolabial": "法令紋",
    "cheek": "蘋果肌",
    "jawline": "下頜緣",
    "chin": "下巴",
    "skin": "皮膚質地",
    "symmetry": "對稱性",
}

SCORE_THRESHOLD = 0.25

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


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


def get_landmarks(img_bgr):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
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
    return (ratio - 0.5) * 180.0


def classify_severity(value):
    if value < 0.3:
        return "mild"
    elif value < 0.7:
        return "moderate"
    else:
        return "severe"


def classify_severity_zh(value):
    if value < 0.3:
        return "輕度"
    elif value < 0.7:
        return "中度"
    else:
        return "重度"


def safe_depth_std(landmarks, indices):
    pts = landmarks[indices, 2]
    return float(np.std(pts))


def normalize(value, min_v, max_v):
    if max_v - min_v < 1e-6:
        return 0.0
    return max(0.0, min(1.0, (value - min_v) / (max_v - min_v)))


def analyze_face(landmarks, img_bgr):
    h, w = img_bgr.shape[:2]
    results = {}

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

    results["zones"] = {
        "upper_ratio": round(upper_ratio, 3),
        "middle_ratio": round(middle_ratio, 3),
        "lower_ratio": round(lower_ratio, 3),
        "dominant": (
            "upper" if upper_ratio > middle_ratio and upper_ratio > lower_ratio
            else "middle" if middle_ratio >= upper_ratio and middle_ratio > lower_ratio
            else "lower"
        ),
    }

    face_width = abs(landmarks[454, 0] - landmarks[234, 0]) + 1e-6
    left_eye_w = abs(landmarks[133, 0] - landmarks[33, 0])
    right_eye_w = abs(landmarks[263, 0] - landmarks[362, 0])
    inter_eye = abs(landmarks[362, 0] - landmarks[133, 0])
    eye_avg = (left_eye_w + right_eye_w) / 2.0 + 1e-6
    five_eye_ratio = face_width / (5 * eye_avg)
    inter_ratio = inter_eye / eye_avg

    results["five_eyes"] = {
        "face_width": round(face_width, 1),
        "eye_avg_width": round(eye_avg, 1),
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
    results["symmetry"] = {
        "score": round(asym_norm, 3),
        "severity": classify_severity(asym_norm),
    }

    tear_z_avg = np.mean(landmarks[[159, 145], 2])
    cheek_z_avg = np.mean(landmarks[[50, 280], 2])
    tear_depth = abs(tear_z_avg - cheek_z_avg)
    tear_norm = normalize(tear_depth, 0.001, 0.015)
    results["tear"] = {"score": round(tear_norm, 3), "severity": classify_severity(tear_norm)}

    nasal_z = np.mean(landmarks[[49, 279], 2])
    mouth_corner_z = np.mean(landmarks[[61, 291], 2])
    nasolabial_depth = abs(nasal_z - mouth_corner_z)
    left_len = np.linalg.norm(landmarks[49, :2] - landmarks[61, :2])
    right_len = np.linalg.norm(landmarks[279, :2] - landmarks[291, :2])
    nasolabial_len_avg = (left_len + right_len) / 2.0
    depth_norm = normalize(nasolabial_depth, 0.001, 0.015)
    len_norm = normalize(nasolabial_len_avg, face_width * 0.1, face_width * 0.25)
    nasolabial_score = 0.6 * depth_norm + 0.4 * len_norm
    results["nasolabial"] = {"score": round(nasolabial_score, 3), "severity": classify_severity(nasolabial_score)}

    cheekbone_z = np.mean(landmarks[[117, 346], 2])
    cheek_ref_z = np.mean(landmarks[[50, 280], 2])
    apple_muscle = abs(cheekbone_z - cheek_ref_z)
    apple_norm = normalize(apple_muscle, 0.001, 0.012)
    apple_score = 1.0 - apple_norm
    results["cheek"] = {"score": round(apple_score, 3), "severity": classify_severity(apple_score)}

    jaw_indices = [152, 172, 171, 170, 169, 136, 135, 134, 58, 172]
    jaw_pts = landmarks[jaw_indices, :2]
    jaw_std = np.std(jaw_pts[:, 1])
    jaw_norm = normalize(jaw_std, 2.0, 25.0)
    results["jawline"] = {"score": round(jaw_norm, 3), "severity": classify_severity(jaw_norm)}

    forehead_indices = [55, 107, 66, 105, 65, 52, 53, 46, 124, 156, 70, 63]
    forehead_std = safe_depth_std(landmarks, forehead_indices)
    forehead_norm = normalize(forehead_std, 0.001, 0.008)
    results["forehead"] = {"score": round(forehead_norm, 3), "severity": classify_severity(forehead_norm)}

    glabella_indices = [168, 6, 197, 195, 5, 4, 8, 9]
    glabella_std = safe_depth_std(landmarks, glabella_indices)
    glabella_norm = normalize(glabella_std, 0.001, 0.008)
    results["glabella"] = {"score": round(glabella_norm, 3), "severity": classify_severity(glabella_norm)}

    crow_indices = [33, 133, 246, 161, 160, 159, 263, 362, 466, 388, 387, 386]
    crow_std = safe_depth_std(landmarks, crow_indices)
    crow_norm = normalize(crow_std, 0.001, 0.01)
    results["crowsfeet"] = {"score": round(crow_norm, 3), "severity": classify_severity(crow_norm)}

    chin_tip = landmarks[152]
    chin_left = landmarks[172]
    chin_right = landmarks[397]
    chin_width = abs(chin_left[0] - chin_right[0])
    chin_length = abs(chin_tip[1] - subnasale_y)
    chin_ratio = chin_length / (chin_width + 1e-6)
    chin_score = normalize(chin_ratio, 0.3, 1.0)
    results["chin"] = {"score": round(chin_score, 3), "severity": classify_severity(chin_score)}

    skin_indices = [50, 280, 205, 425, 117, 346, 187, 411]
    skin_z = landmarks[skin_indices, 2]
    skin_cv = np.std(skin_z) / (np.mean(np.abs(skin_z)) + 1e-6)
    skin_norm = normalize(skin_cv, 0.05, 0.6)
    results["skin"] = {"score": round(skin_norm, 3), "severity": classify_severity(skin_norm)}

    return results


def analyze_calf(img_bgr):
    result = {"detected": False, "score": 0, "severity": "mild"}
    try:
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)
        if not res.pose_landmarks:
            return result
        h, w = img_bgr.shape[:2]
        lm = res.pose_landmarks.landmark
        lk = lm[mp_pose.PoseLandmark.LEFT_KNEE]
        la = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
        rk = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
        ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
        if abs(lk.y - la.y) * h < 10 or abs(rk.y - ra.y) * h < 10:
            return result
        calf_width = abs(lk.x - rk.x) * w
        calf_ratio = calf_width / (w * 0.6 + 1e-6)
        score = normalize(float(calf_ratio), 0.2, 0.6)
        result = {
            "detected": True,
            "score": round(score, 3),
            "severity": classify_severity(score),
            "calf_ratio": round(float(calf_ratio), 3),
            "left_knee": (int(lk.x * w), int(lk.y * h)),
            "left_ankle": (int(la.x * w), int(la.y * h)),
            "right_knee": (int(rk.x * w), int(rk.y * h)),
            "right_ankle": (int(ra.x * w), int(ra.y * h)),
        }
    except Exception:
        pass
    return result


def analyze_back(img_front_bgr, img_side_bgr):
    result = {"detected": False, "score": 0, "severity": "mild"}
    try:
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            img_rgb = cv2.cvtColor(img_front_bgr, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)
        if not res.pose_landmarks:
            return result
        h, w = img_front_bgr.shape[:2]
        lm = res.pose_landmarks.landmark
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lhip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        rhip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        shoulder_w = abs(ls.x - rs.x) * w
        hip_w = abs(lhip.x - rhip.x) * w
        bwr = shoulder_w / (w * 0.8 + 1e-6)
        shr = shoulder_w / (hip_w + 1e-6)
        score = normalize(bwr, 0.3, 0.7)
        result = {
            "detected": True,
            "score": round(score, 3),
            "severity": classify_severity(score),
            "back_width_ratio": round(float(bwr), 3),
            "shoulder_width_px": round(float(shoulder_w), 1),
            "hip_width_px": round(float(hip_w), 1),
            "shoulder_hip_ratio": round(float(shr), 3),
            "back_thickness_ratio": 0.0,
        }
        if img_side_bgr is not None:
            with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
                res2 = pose.process(cv2.cvtColor(img_side_bgr, cv2.COLOR_BGR2RGB))
            if res2.pose_landmarks:
                h2, w2 = img_side_bgr.shape[:2]
                lm2 = res2.pose_landmarks.landmark
                nose2 = lm2[mp_pose.PoseLandmark.NOSE]
                ls2 = lm2[mp_pose.PoseLandmark.LEFT_SHOULDER]
                result["back_thickness_ratio"] = round(abs(nose2.x - ls2.x), 3)
    except Exception:
        pass
    return result


def draw_annotations(img_bgr, landmarks, analysis):
    img = img_bgr.copy()
    h, w = img.shape[:2]
    COLOR = (0, 220, 255)
    TEXT = (255, 255, 255)
    lines_y = {
        "top": int(landmarks[10, 1]),
        "mid1": int(landmarks[8, 1]),
        "mid2": int(landmarks[94, 1]),
        "bot": int(landmarks[152, 1]),
    }
    for y in lines_y.values():
        cv2.line(img, (0, y), (w, y), COLOR, 1)
    for x in [landmarks[234, 0], landmarks[33, 0], landmarks[133, 0],
               landmarks[362, 0], landmarks[263, 0], landmarks[454, 0]]:
        cv2.line(img, (int(x), 0), (int(x), h), (200, 200, 0), 1)
    for idx in [10, 8, 94, 152, 33, 133, 362, 263, 234, 454, 1]:
        cv2.circle(img, (int(landmarks[idx, 0]), int(landmarks[idx, 1])), 3, (0, 100, 255), -1)
    zones = analysis.get("zones", {})
    labels = [
        (lines_y["top"], lines_y["mid1"], "upper {:.1%}".format(zones.get("upper_ratio", 0))),
        (lines_y["mid1"], lines_y["mid2"], "middle {:.1%}".format(zones.get("middle_ratio", 0))),
        (lines_y["mid2"], lines_y["bot"], "lower {:.1%}".format(zones.get("lower_ratio", 0))),
    ]
    for y1, y2, text in labels:
        cv2.putText(img, text, (w - 140, (y1 + y2) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR, 1, cv2.LINE_AA)
    return img


def generate_recommendations(analysis):
    rec_list = []
    for key, name_zh in PROBLEM_NAMES_ZH.items():
        data = analysis.get(key, {})
        if not data or "score" not in data:
            continue
        score = data["score"]
        if score < SCORE_THRESHOLD:
            continue
        sev = data["severity"]
        if key not in PROBLEM_TO_PRODUCTS:
            continue
        primary = []
        alternatives = []
        for pk, is_primary in PROBLEM_TO_PRODUCTS[key]:
            if pk not in PRODUCT_DB:
                continue
            prod = PRODUCT_DB[pk]
            info = {
                "key": pk,
                "name_zh": PRODUCT_NAMES_ZH.get(pk, pk),
                "brand": prod["brands"][0],
                "category": prod["category"],
                "dose": prod["dose_mid"].get(sev, "consult doctor"),
                "layer": prod["layer"],
                "method": prod["method"],
                "effect": prod["effect"],
            }
            if is_primary:
                primary.append(info)
            else:
                alternatives.append(info)
        rec_list.append({
            "key": key,
            "name_zh": name_zh,
            "score": score,
            "severity": sev,
            "severity_zh": classify_severity_zh(score),
            "primary": primary,
            "alternatives": alternatives,
        })
    return rec_list


def physiognomy_reading(analysis):
    readings = []
    zones = analysis.get("zones", {})
    dominant = zones.get("dominant", "")
    upper_r = zones.get("upper_ratio", 0.333)
    middle_r = zones.get("middle_ratio", 0.333)
    lower_r = zones.get("lower_ratio", 0.333)

    zone_info = {
        "upper": {
            "zh": "上庭",
            "good": "上庭寬闊，早年運佳，智慧過人，父母緣深厚",
            "improve": "若額頭有皺紋或過窄，可透過肉毒放鬆額肌改善抬頭紋，或以玻尿酸填充額頭弧度，讓上庭更飽滿圓潤，提升整體氣場。",
            "ratio": upper_r,
            "products_zh": ["肉毒毒素（抬頭紋）", "玻尿酸（全臉保濕 VOLITE）"],
        },
        "middle": {
            "zh": "中庭",
            "good": "中庭均衡，中年事業運旺，適合創業或擔任管理職",
            "improve": "若鼻樑較低或法令紋明顯，可透過玻尿酸墊高鼻樑、填充法令紋，讓中庭比例更完美，面相上增強事業運與威嚴感。",
            "ratio": middle_r,
            "products_zh": ["玻尿酸（法令紋 VOLUMA）", "玻尿酸（法令紋 VOLIFT）"],
        },
        "lower": {
            "zh": "下庭",
            "good": "下庭豐厚，晚年運佳，福氣深厚，子孫有緣",
            "improve": "若下巴短或下頜緣不清晰，可透過玻尿酸墊下巴或埋線提拉，讓下庭比例更協調，面相上加強晚年財運與福氣。",
            "ratio": lower_r,
            "products_zh": ["玻尿酸（下巴）", "玻尿酸（下頜輪廓 VOLUX）"],
        },
    }

    for zone_key, info in zone_info.items():
        is_dom = (dominant == zone_key)
        readings.append({
            "aspect": info["zh"] + "分析",
            "zone": info["zh"],
            "ratio": "{:.1%}".format(info["ratio"]),
            "icon": "star" if is_dom else "square",
            "reading": info["good"] if is_dom else info["zh"] + "比例偏低，建議調整",
            "beauty_tip": info["improve"],
            "products_zh": info["products_zh"],
        })

    inter_r = analysis.get("five_eyes", {}).get("inter_ratio", 1.0)
    if inter_r < 0.85:
        readings.append({
            "aspect": "眼距偏窄",
            "zone": "",
            "ratio": "{:.2f}".format(inter_r),
            "icon": "bolt",
            "reading": "眼距較窄，個性敏銳反應快，容易急躁",
            "beauty_tip": "可透過眉形調整或淚溝填充，視覺上拉寬眼距，讓面部比例更協調。",
            "products_zh": ["玻尿酸（淚溝 VOLBELLA）"],
        })
    elif inter_r > 1.2:
        readings.append({
            "aspect": "眼距寬廣",
            "zone": "",
            "ratio": "{:.2f}".format(inter_r),
            "icon": "wave",
            "reading": "眼距寬廣，心胸寬大，人緣佳，適合公關外交",
            "beauty_tip": "眼距已相當理想，可搭配眼周保養或皮秒改善眼周膚質。",
            "products_zh": ["皮秒雷射"],
        })

    nasolabial_score = analysis.get("nasolabial", {}).get("score", 0)
    if nasolabial_score > 0.5:
        readings.append({
            "aspect": "法令紋顯現",
            "zone": "",
            "ratio": "",
            "icon": "crown",
            "reading": "法令紋深象徵威嚴與領導力，主掌大局之相",
            "beauty_tip": "法令紋象徵威嚴，但視覺顯老。可透過玻尿酸填充或舒顏萃刺激膠原新生，保持威嚴命格同時外觀更年輕。",
            "products_zh": ["玻尿酸（法令紋 VOLUMA）", "舒顏萃 Sculptra"],
        })

    return readings


def apply_styles():
    st.markdown("""
<style>
.stApp { background: #0d1117; }
h1, h2, h3, h4 { color: #ffd700 !important; }
p, li { color: #e8e8e8 !important; }
.stMarkdown p { color: #e8e8e8 !important; }
label { color: #e0e0e0 !important; }
.stRadio label { color: #e8e8e8 !important; font-weight: 600; }
.stCaption { color: #a0b4c8 !important; }
.ulabel { font-size: 0.95rem; font-weight: 700; color: #ffd700; margin-bottom: 6px; display: block; }
.aok { color: #2ecc71; font-size: 0.88rem; font-weight: 600; }
.afail { color: #e74c3c; font-size: 0.88rem; font-weight: 600; }
.bmild { background: #1a3a2a; color: #2ecc71; border: 1px solid #2ecc71; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
.bmod { background: #3a2a0a; color: #ffcc00; border: 1px solid #ffcc00; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
.bsev { background: #3a1a1a; color: #ff6060; border: 1px solid #ff6060; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
.ptag { background: #1a3a2a; color: #2ecc71; border: 1px solid #2ecc71; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; }
.atag { background: #1a1a3a; color: #8888ff; border: 1px solid #8888ff; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; }
.stButton > button { background: linear-gradient(135deg, #e0c44a, #c0a030) !important; color: #000 !important; font-weight: 700 !important; border-radius: 8px !important; border: none !important; }
div[data-testid="metric-container"] label { color: #a0b4c8 !important; }
div[data-testid="metric-container"] [data-testid="metric-value"] { color: #ffd700 !important; }
div[data-testid="stInfo"] p { color: #c0d8f0 !important; }
div[data-testid="stSuccess"] p { color: #c0f0c0 !important; }
div[data-testid="stWarning"] p { color: #f0d080 !important; }
.stExpander summary p { color: #ffd700 !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def badge(severity_zh):
    cls = {"mild": "bmild", "moderate": "bmod", "severe": "bsev"}.get(
        {"mild": "mild", "moderate": "moderate", "severe": "severe",
         "輕度": "mild", "中度": "moderate", "重度": "severe"}.get(severity_zh, "mild"), "bmild"
    )
    return "<span class='{}'>{}</span>".format(cls, severity_zh)


def main():
    apply_styles()
    st.title("AI Beauty System")
    st.caption("AI Facial and Body Analysis")
    st.markdown("---")
    mode = st.radio("**Analysis Mode**", ["Face Analysis", "Calf Muscle Analysis", "Back Analysis"], horizontal=True)

    # ── FACE ANALYSIS ───────────────────────────────────────────────────────
    if mode == "Face Analysis":
        st.markdown("---")
        st.subheader("Upload Face Photos (up to 5 angles)")
        st.info("Please upload: Front (0), Left 45, Left 90, Right 45, Right 90")

        ANGLES = {
            "front":   {"expected": 0,   "tol": 15, "label": "Front (0)"},
            "left45":  {"expected": -35, "tol": 18, "label": "Left 45"},
            "left90":  {"expected": -70, "tol": 20, "label": "Left 90"},
            "right45": {"expected": 35,  "tol": 18, "label": "Right 45"},
            "right90": {"expected": 70,  "tol": 20, "label": "Right 90"},
        }

        c1, c2, c3, c4, c5 = st.columns(5)
        cols = {"front": c1, "left45": c2, "left90": c3, "right45": c4, "right90": c5}
        uploads = {}

        for key, col in cols.items():
            with col:
                st.markdown("<span class='ulabel'>{}</span>".format(ANGLES[key]["label"]), unsafe_allow_html=True)
                f = st.file_uploader("", type=["jpg", "jpeg", "png"], key=key, label_visibility="collapsed")
                if f:
                    uploads[key] = Image.open(f).convert("RGB")
                    st.image(uploads[key], use_container_width=True)

        if uploads:
            st.markdown("---")
            st.subheader("Angle Verification")
            vcols = st.columns(len(uploads))
            for i, (key, pil_img) in enumerate(uploads.items()):
                cfg = ANGLES[key]
                img_bgr = pil_to_cv2(resize_image(pil_img, 640))
                lm, _ = get_landmarks(img_bgr)
                with vcols[i]:
                    if lm is None:
                        st.markdown("<div class='afail'>No face detected</div>", unsafe_allow_html=True)
                    else:
                        yaw = estimate_yaw(lm)
                        ok = abs(yaw - cfg["expected"]) <= cfg["tol"]
                        cls = "aok" if ok else "afail"
                        icon = "OK" if ok else "CHECK"
                        st.markdown("<div class='{}'>{} {} Yaw={:.1f}</div>".format(cls, icon, cfg["label"], yaw), unsafe_allow_html=True)

        st.markdown("---")

        if "front" in uploads:
            if st.button("Start AI Analysis", use_container_width=True):
                with st.spinner("Analyzing..."):
                    try:
                        front_bgr = pil_to_cv2(resize_image(uploads["front"], 800))
                        lm_front, _ = get_landmarks(front_bgr)
                        if lm_front is None:
                            st.error("Cannot detect face. Please upload a clearer photo.")
                            return
                        analysis = analyze_face(lm_front, front_bgr)
                        img_ann = draw_annotations(front_bgr, lm_front, analysis)
                        recs = generate_recommendations(analysis)
                        physio = physiognomy_reading(analysis)

                        st.success("Analysis complete!")

                        st.subheader("Three Zones Annotation")
                        st.image(cv2_to_pil(img_ann), use_container_width=True)

                        st.subheader("Three Zones (三庭)")
                        zones = analysis["zones"]
                        zc1, zc2, zc3 = st.columns(3)
                        zc1.metric("上庭 Upper", "{:.1%}".format(zones["upper_ratio"]), delta="{:+.1%}".format(zones["upper_ratio"] - 0.333))
                        zc2.metric("中庭 Middle", "{:.1%}".format(zones["middle_ratio"]), delta="{:+.1%}".format(zones["middle_ratio"] - 0.333))
                        zc3.metric("下庭 Lower", "{:.1%}".format(zones["lower_ratio"]), delta="{:+.1%}".format(zones["lower_ratio"] - 0.333))
                        dom_zh = {"upper": "上庭", "middle": "中庭", "lower": "下庭"}.get(zones["dominant"], zones["dominant"])
                        st.info("Dominant zone: {} | Ideal each 33.3%".format(dom_zh))

                        st.subheader("Five Eyes (五眼)")
                        fe = analysis["five_eyes"]
                        fc1, fc2 = st.columns(2)
                        fc1.metric("Five-Eye Ratio", "{:.2f}".format(fe["five_eye_ratio"]), delta="{:+.2f} (ideal=1.0)".format(fe["five_eye_ratio"] - 1.0))
                        fc2.metric("Inter-Eye Ratio", "{:.2f}".format(fe["inter_ratio"]), delta="{:+.2f} (ideal=1.0)".format(fe["inter_ratio"] - 1.0))

                        st.subheader("Problem Diagnosis")
                        scored = sorted(
                            [(k, v) for k, v in analysis.items() if isinstance(v, dict) and "score" in v],
                            key=lambda x: x[1]["score"], reverse=True
                        )
                        for key, data in scored:
                            name = PROBLEM_NAMES_ZH.get(key, key)
                            sev_zh = classify_severity_zh(data["score"])
                            sev_cls = {"輕度": "bmild", "中度": "bmod", "重度": "bsev"}.get(sev_zh, "bmild")
                            st.markdown(
                                "<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>"
                                "<span style='color:#fff;font-weight:700;min-width:80px;'>{}</span>"
                                "<span class='{}'>{}</span>"
                                "<span style='color:#aaa;font-size:0.85rem;'>{:.2f}</span>"
                                "</div>".format(name, sev_cls, sev_zh, data["score"]),
                                unsafe_allow_html=True
                            )
                            st.progress(data["score"])

                        st.subheader("Treatment Recommendations")
                        st.markdown("<p style='color:#c0d8e8;font-size:0.88rem;'>Primary = best choice | Alternative = supplementary</p>", unsafe_allow_html=True)
                        if recs:
                            for rec in recs:
                                label = "{} - {} ({:.2f})".format(rec["name_zh"], rec["severity_zh"], rec["score"])
                                with st.expander(label, expanded=rec["score"] > 0.5):
                                    if rec["primary"]:
                                        st.markdown("<div style='color:#2ecc71;font-weight:700;margin-bottom:6px;'>Primary Treatment</div>", unsafe_allow_html=True)
                                        for prod in rec["primary"]:
                                            st.markdown(
                                                "<div style='background:#0a1f10;border-radius:8px;padding:14px;margin:6px 0;border-left:4px solid #2ecc71;'>"
                                                "<div style='color:#ffd700;font-weight:700;margin-bottom:6px;'>{} <span class='ptag'>Primary</span></div>"
                                                "<div style='color:#c0e8c0;font-size:0.88rem;line-height:1.8;'>"
                                                "Brand: {}<br>Dose: {}<br>Layer: {}<br>Method: {}<br>Effect: {}"
                                                "</div></div>".format(
                                                    prod["name_zh"], prod["brand"],
                                                    prod["dose"], prod["layer"], prod["method"], prod["effect"]
                                                ),
                                                unsafe_allow_html=True
                                            )
                                    if rec["alternatives"]:
                                        st.markdown("<div style='color:#8888ff;font-weight:700;margin:10px 0 6px;'>Alternative Treatments</div>", unsafe_allow_html=True)
                                        acols = st.columns(min(len(rec["alternatives"]), 3))
                                        for j, prod in enumerate(rec["alternatives"]):
                                            with acols[j % len(acols)]:
                                                st.markdown(
                                                    "<div style='background:#0a0a20;border-radius:8px;padding:10px;border-left:3px solid #8888ff;'>"
                                                    "<div style='color:#c0c8ff;font-weight:600;'>{} <span class='atag'>Alt</span></div>"
                                                    "<div style='color:#8090c0;font-size:0.82rem;'>{}<br>Dose: {}<br>{}</div>"
                                                    "</div>".format(
                                                        prod["name_zh"], prod["category"], prod["dose"], prod["effect"]
                                                    ),
                                                    unsafe_allow_html=True
                                                )
                        else:
                            st.success("No significant issues detected. Keep up with your skincare!")

                        st.subheader("Physiognomy Analysis")
                        for r in physio:
                            zone_label = " | {} {}".format(r["zone"], r["ratio"]) if r.get("zone") else ""
                            st.markdown(
                                "<div style='background:#1a1a2e;border-radius:10px;padding:16px;margin-bottom:14px;border-left:4px solid #e0c44a;'>"
                                "<div style='color:#ffd700;font-weight:700;font-size:1rem;margin-bottom:8px;'>{}{}</div>"
                                "<div style='background:#0f1520;border-radius:6px;padding:10px;margin-bottom:8px;'>"
                                "<span style='color:#ffd700;font-size:0.85rem;font-weight:600;'>Face Reading: </span>"
                                "<span style='color:#c8d8e8;font-size:0.88rem;'>{}</span>"
                                "</div>"
                                "<div style='background:#0f2015;border-radius:6px;padding:10px;'>"
                                "<span style='color:#2ecc71;font-size:0.85rem;font-weight:600;'>Beauty Tip: </span>"
                                "<span style='color:#a0d0a0;font-size:0.88rem;'>{}</span><br>"
                                "<span style='color:#7090a0;font-size:0.82rem;'>Products: {}</span>"
                                "</div></div>".format(
                                    r["aspect"], zone_label, r["reading"],
                                    r["beauty_tip"], ", ".join(r.get("products_zh", []))
                                ),
                                unsafe_allow_html=True
                            )

                        st.subheader("Download Report")
                        lines = ["AI Beauty Report", "Date: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M")), "", "=== THREE ZONES ==="]
                        for k in ("upper_ratio", "middle_ratio", "lower_ratio", "dominant"):
                            lines.append("  {}: {}".format(k, zones.get(k, "")))
                        lines += ["", "=== ISSUES ==="]
                        for key, data in scored:
                            lines.append("  {}: {} ({:.2f})".format(PROBLEM_NAMES_ZH.get(key, key), classify_severity_zh(data["score"]), data["score"]))
                        lines += ["", "=== TREATMENTS ==="]
                        for rec in recs:
                            lines.append("\n[{}] {}".format(rec["name_zh"], rec["severity_zh"]))
                            for p in rec["primary"]:
                                lines.append("  Primary: {} / {}".format(p["name_zh"], p["dose"]))
                            for p in rec["alternatives"]:
                                lines.append("  Alt: {}".format(p["name_zh"]))
                        lines += ["", "Disclaimer: For reference only. Consult a licensed physician."]
                        st.download_button(
                            "Download Report (TXT)",
                            data="\n".join(lines).encode("utf-8"),
                            file_name="face_report_{}.txt".format(datetime.now().strftime("%Y%m%d_%H%M")),
                            mime="text/plain",
                            use_container_width=True
                        )
                        st.warning("Disclaimer: AI analysis for reference only. Actual treatment requires consultation with a licensed physician.")

                    except Exception as e:
                        st.error("Error: {}".format(e))
                        st.exception(e)
        else:
            st.info("Please upload at least the front photo to begin analysis.")

    # ── CALF ANALYSIS ───────────────────────────────────────────────────────
    elif mode == "Calf Muscle Analysis":
        st.markdown("---")
        st.subheader("Calf Muscle Analysis")
        st.info("Please upload full-body front-facing photos. Ensure legs (knees to ankles) are fully visible.")

        col1, col2 = st.columns(2)
        calf_normal = None
        calf_tiptoe = None
        with col1:
            st.markdown("<span class='ulabel'>Natural Stance (front)</span>", unsafe_allow_html=True)
            f1 = st.file_uploader("", type=["jpg", "jpeg", "png"], key="calf_n", label_visibility="collapsed")
            if f1:
                calf_normal = Image.open(f1).convert("RGB")
                st.image(calf_normal, use_container_width=True)
        with col2:
            st.markdown("<span class='ulabel'>Tiptoe Stance (optional)</span>", unsafe_allow_html=True)
            f2 = st.file_uploader("", type=["jpg", "jpeg", "png"], key="calf_t", label_visibility="collapsed")
            if f2:
                calf_tiptoe = Image.open(f2).convert("RGB")
                st.image(calf_tiptoe, use_container_width=True)

        if calf_normal:
            if st.button("Start Calf Analysis", use_container_width=True):
                with st.spinner("Analyzing..."):
                    try:
                        img_bgr = pil_to_cv2(resize_image(calf_normal, 800))
                        result = analyze_calf(img_bgr)

                        # draw landmarks
                        ann = img_bgr.copy()
                        if result["detected"]:
                            COLOR = (0, 220, 255)
                            for k in ("left_knee", "left_ankle", "right_knee", "right_ankle"):
                                if k in result:
                                    cv2.circle(ann, result[k], 8, COLOR, -1)
                            if "left_knee" in result and "left_ankle" in result:
                                cv2.line(ann, result["left_knee"], result["left_ankle"], COLOR, 2)
                            if "right_knee" in result and "right_ankle" in result:
                                cv2.line(ann, result["right_knee"], result["right_ankle"], COLOR, 2)

                        st.success("Analysis complete!")
                        st.image(cv2_to_pil(ann), use_container_width=True, caption="Calf landmarks")

                        if result["detected"]:
                            sev = result["severity"]
                            sev_zh = classify_severity_zh(result["score"])
                            sev_cls = {"mild": "bmild", "moderate": "bmod", "severe": "bsev"}.get(sev, "bmild")
                            st.markdown(
                                "<div style='background:#1a1a2e;border-radius:10px;padding:16px;border-left:4px solid #e0c44a;margin:12px 0;'>"
                                "<div style='color:#ffd700;font-size:1.1rem;font-weight:700;margin-bottom:8px;'>"
                                "Calf Hypertrophy: <span class='{}'>{}</span></div>"
                                "<div style='color:#c0d0e0;'>Width ratio: {:.3f} | Score: {:.2f}</div>"
                                "</div>".format(sev_cls, sev_zh, result["calf_ratio"], result["score"]),
                                unsafe_allow_html=True
                            )
                            st.progress(result["score"])

                            prod = PRODUCT_DB["botox_calf"]
                            dose = prod["dose_mid"].get(sev, "consult doctor")
                            st.markdown("<h3 style='color:#ffd700;'>Botox Treatment Recommendation</h3>", unsafe_allow_html=True)
                            st.markdown(
                                "<div style='background:#0a1f10;border-radius:10px;padding:16px;border-left:4px solid #2ecc71;'>"
                                "<div style='color:#ffd700;font-size:1.05rem;font-weight:700;margin-bottom:10px;'>Gastrocnemius Botox Injection</div>"
                                "<div style='color:#c0e8c0;font-size:0.9rem;line-height:2;'>"
                                "Brand: {}<br>"
                                "Dose ({}): <span style='color:#ffd700;font-weight:700;'>{}</span><br>"
                                "Layer: {}<br>Method: {}<br>Effect: {}"
                                "</div></div>".format(
                                    ", ".join(prod["brands"]), sev_zh, dose,
                                    prod["layer"], prod["method"], prod["effect"]
                                ),
                                unsafe_allow_html=True
                            )

                            st.markdown("<h3 style='color:#ffd700;'>Post-Treatment Notes</h3>", unsafe_allow_html=True)
                            for tip in [
                                "Avoid intense leg exercise 2 weeks before treatment",
                                "No massage on injection site for 4-6 hours post-treatment",
                                "Avoid exercise, sauna and alcohol for 24 hours",
                                "Maintenance every 6-9 months recommended",
                                "Combine with calf stretching for better results",
                            ]:
                                st.markdown("<div style='color:#c0d0e0;padding:3px 0;'>- {}</div>".format(tip), unsafe_allow_html=True)

                            if calf_tiptoe:
                                st.markdown("<h3 style='color:#ffd700;'>Tiptoe Comparison</h3>", unsafe_allow_html=True)
                                tip_bgr = pil_to_cv2(resize_image(calf_tiptoe, 800))
                                tip_result = analyze_calf(tip_bgr)
                                tip_ann = tip_bgr.copy()
                                if tip_result["detected"]:
                                    COLOR = (0, 220, 255)
                                    for k in ("left_knee", "left_ankle", "right_knee", "right_ankle"):
                                        if k in tip_result:
                                            cv2.circle(tip_ann, tip_result[k], 8, COLOR, -1)
                                ca, cb = st.columns(2)
                                with ca:
                                    st.image(cv2_to_pil(ann), caption="Natural", use_container_width=True)
                                    st.metric("Score", "{:.2f}".format(result["score"]))
                                with cb:
                                    st.image(cv2_to_pil(tip_ann), caption="Tiptoe", use_container_width=True)
                                    if tip_result["detected"]:
                                        delta = tip_result["score"] - result["score"]
                                        st.metric("Score", "{:.2f}".format(tip_result["score"]), delta="{:+.2f}".format(delta))
                                if result["detected"] and tip_result["detected"]:
                                    if tip_result["score"] - result["score"] > 0.1:
                                        st.info("Significant muscle contraction on tiptoe. Consider slightly higher dose.")
                                    else:
                                        st.info("Similar scores between stances. Standard dose is appropriate.")
                        else:
                            st.warning("Cannot detect leg landmarks. Please ensure knees to ankles are fully visible.")

                    except Exception as e:
                        st.error("Error: {}".format(e))
                        st.exception(e)
        else:
            st.info("Please upload a natural stance photo to begin.")

    # ── BACK ANALYSIS ───────────────────────────────────────────────────────
    elif mode == "Back Analysis":
        st.markdown("---")
        st.subheader("Back Muscle and Contour Analysis")
        st.info("Please upload: Back-facing photo (required) + Side-facing photo (optional)")

        col1, col2 = st.columns(2)
        back_front = None
        back_side = None
        with col1:
            st.markdown("<span class='ulabel'>Back-facing (required)</span>", unsafe_allow_html=True)
            bf = st.file_uploader("", type=["jpg", "jpeg", "png"], key="back_f", label_visibility="collapsed")
            if bf:
                back_front = Image.open(bf).convert("RGB")
                st.image(back_front, use_container_width=True)
        with col2:
            st.markdown("<span class='ulabel'>Side-facing (optional)</span>", unsafe_allow_html=True)
            bs = st.file_uploader("", type=["jpg", "jpeg", "png"], key="back_s", label_visibility="collapsed")
            if bs:
                back_side = Image.open(bs).convert("RGB")
                st.image(back_side, use_container_width=True)

        if back_front:
            if st.button("Start Back Analysis", use_container_width=True):
                with st.spinner("Analyzing..."):
                    try:
                        front_bgr = pil_to_cv2(resize_image(back_front, 800))
                        side_bgr = pil_to_cv2(resize_image(back_side, 800)) if back_side else None
                        back_result = analyze_back(front_bgr, side_bgr)

                        st.success("Analysis complete!")

                        if back_result["detected"]:
                            sev = back_result["severity"]
                            sev_zh = classify_severity_zh(back_result["score"])
                            sev_cls = {"mild": "bmild", "moderate": "bmod", "severe": "bsev"}.get(sev, "bmild")
                            st.markdown(
                                "<div style='background:#1a1a2e;border-radius:10px;padding:16px;border-left:4px solid #e0c44a;margin:12px 0;'>"
                                "<div style='color:#ffd700;font-size:1.1rem;font-weight:700;margin-bottom:8px;'>"
                                "Back Assessment: <span class='{}'>{}</span></div>"
                                "<div style='color:#c0d0e0;'>Width ratio: {:.2f} | Shoulder-hip ratio: {:.2f}</div>"
                                "</div>".format(
                                    sev_cls, sev_zh,
                                    back_result.get("back_width_ratio", 0),
                                    back_result.get("shoulder_hip_ratio", 0)
                                ),
                                unsafe_allow_html=True
                            )
                            bc1, bc2, bc3 = st.columns(3)
                            bc1.metric("Back Width Ratio", "{:.2f}".format(back_result.get("back_width_ratio", 0)))
                            bc2.metric("Shoulder Width", "{:.0f}px".format(back_result.get("shoulder_width_px", 0)))
                            bc3.metric("Shoulder-Hip Ratio", "{:.2f}".format(back_result.get("shoulder_hip_ratio", 0)))
                            if back_result.get("back_thickness_ratio", 0) > 0:
                                st.metric("Back Thickness (Side)", "{:.2f}".format(back_result["back_thickness_ratio"]))

                            st.markdown("<h3 style='color:#ffd700;'>Back Treatment Recommendations</h3>", unsafe_allow_html=True)
                            score = back_result.get("score", 0)
                            if score > 0.5:
                                recs_back = [
                                    ("Back Botox (erector spinae relaxation)", "Relax over-developed back muscles, reduce bulky back appearance", "50-100U/side", "2-4 weeks, lasts 4-6 months"),
                                    ("Back Fat Dissolving Injection", "Dissolve localized back fat, improve back contour", "4-8ml/session", "4-8 weeks, long-lasting"),
                                    ("Back Ultherapy", "Tighten loose back skin, improve back contour", "300-500 shots", "3-6 months, lasts 12-18 months"),
                                ]
                            else:
                                recs_back = [
                                    ("Back Hydra Injection", "Improve dry/rough back skin, enhance skin quality", "2-4ml", "Immediate, lasts 4-6 months"),
                                    ("Back Picosecond Laser", "Improve back pigmentation, enlarged pores", "1-3 sessions", "Every 4-6 weeks"),
                                    ("Back Botox (sculpting)", "Fine sculpting for slightly developed muscles", "30-50U/side", "2-4 weeks, lasts 4-6 months"),
                                ]
                            for name, desc, dose, effect in recs_back:
                                st.markdown(
                                    "<div style='background:#0d1a20;border-radius:8px;padding:14px;margin:8px 0;border-left:4px solid #e0c44a;'>"
                                    "<div style='color:#ffd700;font-weight:700;'>{}</div>"
                                    "<div style='color:#c0d0e0;font-size:0.88rem;margin-top:4px;'>{}</div>"
                                    "<div style='color:#90a0b0;font-size:0.84rem;'>Dose: {} | Effect: {}</div>"
                                    "</div>".format(name, desc, dose, effect),
                                    unsafe_allow_html=True
                                )
                        else:
                            st.warning("Cannot detect back landmarks. Please ensure clear full-body back-facing photo with good lighting.")

                    except Exception as e:
                        st.error("Error: {}".format(e))
                        st.exception(e)
        else:
            st.info("Please upload a back-facing photo to begin.")

        st.markdown("---")
        st.warning("Disclaimer: AI analysis for reference only. Consult a licensed physician for actual treatment.")


if __name__ == "__main__":
    main()
