import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import base64
import io
from datetime import datetime
from PIL import Image

st.set_page_config(
    page_title="AI Beauty System",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PRODUCT_DB = {
    "botox_forehead": {
        "name_zh": "frown_forehead",
        "brands_zh": ["botoxA", "botoxB", "botoxC"],
        "dose_mid": {"mild": "12U", "moderate": "20U", "severe": "32U"},
        "layer_zh": "forehead_muscle",
        "method_zh": "multi_point",
        "effect_zh": "3to7days_4to6months",
    },
    "botox_glabella": {
        "name_zh": "frown_glabella",
        "brands_zh": ["botoxA", "botoxB"],
        "dose_mid": {"mild": "12U", "moderate": "20U", "severe": "32U"},
        "layer_zh": "corrugator",
        "method_zh": "5point",
        "effect_zh": "3to7days_4to6months",
    },
    "botox_crowsfeet": {
        "name_zh": "frown_crowsfeet",
        "brands_zh": ["botoxA", "botoxB"],
        "dose_mid": {"mild": "7U", "moderate": "12U", "severe": "17U"},
        "layer_zh": "orbicularis",
        "method_zh": "fan_injection",
        "effect_zh": "5to7days_3to5months",
    },
    "botox_jawline": {
        "name_zh": "frown_jawline",
        "brands_zh": ["botoxA", "botoxB"],
        "dose_mid": {"mild": "15U", "moderate": "25U", "severe": "40U"},
        "layer_zh": "platysma",
        "method_zh": "linear",
        "effect_zh": "4to6months",
    },
    "botox_calf": {
        "name_zh": "frown_calf",
        "brands_zh": ["botoxA", "botoxB", "botoxC"],
        "dose_mid": {"mild": "65U/side", "moderate": "90U/side", "severe": "125U/side"},
        "layer_zh": "gastrocnemius_medial",
        "method_zh": "grid_multipoint",
        "effect_zh": "4to8weeks_6to9months",
    },
    "ha_cheek": {
        "name_zh": "ha_cheek",
        "brands_zh": ["Juvederm VOLUMA"],
        "dose_mid": {"mild": "0.75ml/side", "moderate": "1.25ml/side", "severe": "1.75ml/side"},
        "layer_zh": "subperiosteal",
        "method_zh": "fan_linear",
        "effect_zh": "18to24months",
    },
    "ha_chin": {
        "name_zh": "ha_chin",
        "brands_zh": ["Juvederm VOLUMA"],
        "dose_mid": {"mild": "0.75ml", "moderate": "1.25ml", "severe": "1.75ml"},
        "layer_zh": "subperiosteal",
        "method_zh": "single_fan",
        "effect_zh": "12to18months",
    },
    "ha_nasolabial": {
        "name_zh": "ha_nasolabial",
        "brands_zh": ["Juvederm VOLUMA"],
        "dose_mid": {"mild": "0.75ml/side", "moderate": "1.25ml/side", "severe": "1.75ml/side"},
        "layer_zh": "deep_subcutaneous",
        "method_zh": "retrograde_fan",
        "effect_zh": "12to18months",
    },
    "ha_jawline": {
        "name_zh": "ha_jawline",
        "brands_zh": ["Juvederm VOLUX"],
        "dose_mid": {"mild": "1.5ml", "moderate": "2.5ml", "severe": "3.5ml"},
        "layer_zh": "subperiosteal",
        "method_zh": "linear_mandible",
        "effect_zh": "18to24months",
    },
    "ha_nasolabial2": {
        "name_zh": "ha_nasolabial2",
        "brands_zh": ["Juvederm VOLIFT"],
        "dose_mid": {"mild": "0.9ml/side", "moderate": "1.25ml/side", "severe": "1.75ml/side"},
        "layer_zh": "deep_dermis",
        "method_zh": "retrograde_fern",
        "effect_zh": "12to15months",
    },
    "ha_tear": {
        "name_zh": "ha_tear",
        "brands_zh": ["Juvederm VOLBELLA"],
        "dose_mid": {"mild": "0.4ml/side", "moderate": "0.75ml/side", "severe": "1.25ml/side"},
        "layer_zh": "preseptal_fat",
        "method_zh": "micro_multipoint",
        "effect_zh": "9to12months",
    },
    "ha_skin": {
        "name_zh": "ha_skin",
        "brands_zh": ["Juvederm VOLITE"],
        "dose_mid": {"mild": "1.5ml", "moderate": "2.5ml", "severe": "3.5ml"},
        "layer_zh": "mid_dermis",
        "method_zh": "multipoint_gun",
        "effect_zh": "6to9months",
    },
    "sculptra": {
        "name_zh": "sculptra",
        "brands_zh": ["Sculptra"],
        "dose_mid": {"mild": "1vial", "moderate": "2vials", "severe": "3vials"},
        "layer_zh": "deep_dermis",
        "method_zh": "fan_massage",
        "effect_zh": "2to3months_18to24months",
    },
    "thread": {
        "name_zh": "thread",
        "brands_zh": ["PDO Thread"],
        "dose_mid": {"mild": "5threads/side", "moderate": "8threads/side", "severe": "13threads/side"},
        "layer_zh": "SMAS",
        "method_zh": "retrograde_barb",
        "effect_zh": "immediate_12to18months",
    },
    "pn": {
        "name_zh": "pn",
        "brands_zh": ["Rejuran PN"],
        "dose_mid": {"mild": "1ml", "moderate": "1.75ml", "severe": "2.5ml"},
        "layer_zh": "mid_dermis",
        "method_zh": "micropoint",
        "effect_zh": "3to4sessions",
    },
    "picosecond": {
        "name_zh": "picosecond",
        "brands_zh": ["Picosecond"],
        "dose_mid": {"mild": "1session", "moderate": "4sessions", "severe": "6sessions"},
        "layer_zh": "epidermis_dermis",
        "method_zh": "full_face",
        "effect_zh": "every4to6weeks",
    },
    "ultherapy": {
        "name_zh": "ultherapy",
        "brands_zh": ["Ultherapy"],
        "dose_mid": {"mild": "300shots", "moderate": "500shots", "severe": "800shots"},
        "layer_zh": "SMAS_dermis",
        "method_zh": "linear_scan",
        "effect_zh": "3to6months_12to18months",
    },
    "thermage": {
        "name_zh": "thermage",
        "brands_zh": ["Thermage FLX"],
        "dose_mid": {"mild": "900shots", "moderate": "1200shots", "severe": "1500shots"},
        "layer_zh": "deep_dermis",
        "method_zh": "even_scan",
        "effect_zh": "immediate_12to24months",
    },
}

PRODUCT_ZH = {
    "botox_forehead": ("肉毒毒素（抬頭紋）", "奇蹟肉毒 / 天使肉毒", "額肌（肌肉層）", "多點注射，間距約1.5cm", "3-7天見效，維持4-6個月"),
    "botox_glabella": ("肉毒毒素（眉間紋）", "奇蹟肉毒 / 天使肉毒", "皺眉肌", "5點標準注射法", "3-7天改善，維持4-6個月"),
    "botox_crowsfeet": ("肉毒毒素（魚尾紋）", "奇蹟肉毒 / 天使肉毒", "眼輪匝肌", "眼外角扇形多點注射", "5-7天平滑，維持3-5個月"),
    "botox_jawline": ("肉毒毒素（下頜緣）", "奇蹟肉毒 / 天使肉毒", "頸闊肌", "沿頸闊肌線性注射", "輪廓提升，維持4-6個月"),
    "botox_calf": ("肉毒毒素（小腿腓腸肌）", "奇蹟肉毒 / 天使肉毒 / 寶提拉", "腓腸肌內側頭（深層）", "多點格狀注射，每點5-10U", "4-8週線條改善，維持6-9個月"),
    "ha_cheek": ("玻尿酸（蘋果肌 VOLUMA）", "喬雅登 VOLUMA", "骨膜上層或深層皮下脂肪", "扇形注射 / 線性推注", "蘋果肌圓潤，維持18-24個月"),
    "ha_chin": ("玻尿酸（下巴 VOLUMA）", "喬雅登 VOLUMA", "骨膜上層", "單點或扇形注射", "下巴延長翹挺，維持12-18個月"),
    "ha_nasolabial": ("玻尿酸（法令紋 VOLUMA）", "喬雅登 VOLUMA", "深層皮下 / 骨膜上層", "逆行線性 + 扇形", "法令紋減少60-80%，維持12-18個月"),
    "ha_jawline": ("玻尿酸（下頜輪廓 VOLUX）", "喬雅登 VOLUX", "骨膜上層", "線性注射，沿下頜骨緣推注", "下頜輪廓清晰，維持18-24個月"),
    "ha_nasolabial2": ("玻尿酸（法令紋 VOLIFT）", "喬雅登 VOLIFT（豐麗緹）", "真皮深層至皮下層", "逆行線性 + 蕨葉技術", "法令紋平滑，維持12-15個月"),
    "ha_tear": ("玻尿酸（淚溝 VOLBELLA）", "喬雅登 VOLBELLA（夢蓓菈）", "眶隔前脂肪層 / 骨膜上層", "微量多點 / 線性注射", "淚溝填補，維持9-12個月"),
    "ha_skin": ("玻尿酸（全臉保濕 VOLITE）", "喬雅登 VOLITE（芙潤）", "真皮中層", "多點均勻注射 / 水光槍輔助", "膚質細緻，維持6-9個月"),
    "sculptra": ("舒顏萃 Sculptra", "舒顏萃 Sculptra", "真皮深層至皮下層", "扇形大範圍注射，按摩分散", "刺激膠原新生，2-3月顯現，維持18-24個月"),
    "thread": ("鳳凰埋線（大V線）", "鳳凰埋線", "SMAS筋膜層 / 深層皮下", "逆行進針，雙向倒鉤提拉", "即時提拉，維持12-18個月"),
    "pn": ("麗珠蘭 PN 1%", "麗珠蘭 PN 1%", "真皮淺中層", "水光槍 / 多點注射", "膚色提亮，建議3-4療程"),
    "picosecond": ("皮秒雷射", "皮秒雷射", "表皮至真皮層", "全臉掃描，依斑點加強", "膚色均勻，每4-6週一次"),
    "ultherapy": ("音波拉提 Ultherapy", "音波拉提", "SMAS筋膜層 + 真皮層", "線性掃描，分層施打", "3-6月顯效，維持12-18個月"),
    "thermage": ("電波拉提 Thermage", "電波拉提 Thermage FLX", "真皮深層至皮下層", "全臉均勻掃描", "即時緊緻，維持12-24個月"),
}

PROBLEM_ZH = {
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

PROBLEM_TO_PRODUCTS = {
    "forehead":   [("botox_forehead", True), ("picosecond", False), ("pn", False)],
    "glabella":   [("botox_glabella", True), ("picosecond", False)],
    "crowsfeet":  [("botox_crowsfeet", True), ("thermage", False), ("ha_tear", False)],
    "tear":       [("ha_tear", True), ("pn", False)],
    "nasolabial": [("ha_nasolabial", True), ("ha_nasolabial2", True), ("sculptra", False), ("thread", False)],
    "cheek":      [("ha_cheek", True), ("sculptra", False), ("thread", False)],
    "jawline":    [("ha_jawline", True), ("botox_jawline", True), ("ultherapy", False)],
    "chin":       [("ha_chin", True)],
    "skin":       [("ha_skin", True), ("picosecond", False), ("pn", False)],
    "symmetry":   [("botox_glabella", True), ("ha_cheek", False)],
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
        min_detection_confidence=0.5,
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
    return (left_dist / total - 0.5) * 180.0


def classify_sev(value):
    if value < 0.3:
        return "mild"
    elif value < 0.7:
        return "moderate"
    return "severe"


def sev_zh(sev):
    return {"mild": "輕度", "moderate": "中度", "severe": "重度"}.get(sev, "輕度")


def normalize(value, min_v, max_v):
    if max_v - min_v < 1e-6:
        return 0.0
    return max(0.0, min(1.0, (value - min_v) / (max_v - min_v)))


def safe_std(landmarks, indices):
    return float(np.std(landmarks[indices, 2]))


def analyze_face(landmarks, img_bgr):
    h, w = img_bgr.shape[:2]
    R = {}

    hairline_y = landmarks[10, 1]
    glabella_y = landmarks[8, 1]
    subnasale_y = landmarks[94, 1]
    chin_y = landmarks[152, 1]
    upper = abs(glabella_y - hairline_y)
    middle = abs(subnasale_y - glabella_y)
    lower = abs(chin_y - subnasale_y)
    total = upper + middle + lower + 1e-6
    R["zones"] = {
        "upper_ratio": round(upper / total, 3),
        "middle_ratio": round(middle / total, 3),
        "lower_ratio": round(lower / total, 3),
        "dominant": (
            "upper" if upper > middle and upper > lower
            else "middle" if middle >= upper and middle > lower
            else "lower"
        ),
    }

    face_w = abs(landmarks[454, 0] - landmarks[234, 0]) + 1e-6
    lew = abs(landmarks[133, 0] - landmarks[33, 0])
    rew = abs(landmarks[263, 0] - landmarks[362, 0])
    inter = abs(landmarks[362, 0] - landmarks[133, 0])
    eye_avg = (lew + rew) / 2.0 + 1e-6
    R["five_eyes"] = {
        "face_width": round(face_w, 1),
        "eye_avg_width": round(eye_avg, 1),
        "five_eye_ratio": round(face_w / (5 * eye_avg), 3),
        "inter_ratio": round(inter / eye_avg, 3),
    }

    nose_x = landmarks[1, 0]
    sym_pairs = [(33, 263), (133, 362), (234, 454), (61, 291), (50, 280), (149, 378)]
    asym = sum(
        abs(abs(landmarks[li, 0] - nose_x) - abs(landmarks[ri, 0] - nose_x)) /
        max(abs(landmarks[li, 0] - nose_x), abs(landmarks[ri, 0] - nose_x), 1e-6)
        for li, ri in sym_pairs
    ) / len(sym_pairs)
    asym_n = normalize(asym, 0.0, 0.3)
    R["symmetry"] = {"score": round(asym_n, 3), "severity": classify_sev(asym_n)}

    tear_z = np.mean(landmarks[[159, 145], 2])
    cheek_z = np.mean(landmarks[[50, 280], 2])
    tear_n = normalize(abs(tear_z - cheek_z), 0.001, 0.015)
    R["tear"] = {"score": round(tear_n, 3), "severity": classify_sev(tear_n)}

    nasal_z = np.mean(landmarks[[49, 279], 2])
    mouth_z = np.mean(landmarks[[61, 291], 2])
    nasolab_d = normalize(abs(nasal_z - mouth_z), 0.001, 0.015)
    ll = np.linalg.norm(landmarks[49, :2] - landmarks[61, :2])
    rl = np.linalg.norm(landmarks[279, :2] - landmarks[291, :2])
    nasolab_l = normalize((ll + rl) / 2.0, face_w * 0.1, face_w * 0.25)
    nasolab_s = 0.6 * nasolab_d + 0.4 * nasolab_l
    R["nasolabial"] = {"score": round(nasolab_s, 3), "severity": classify_sev(nasolab_s)}

    cb_z = np.mean(landmarks[[117, 346], 2])
    ck_z = np.mean(landmarks[[50, 280], 2])
    apple_s = 1.0 - normalize(abs(cb_z - ck_z), 0.001, 0.012)
    R["cheek"] = {"score": round(apple_s, 3), "severity": classify_sev(apple_s)}

    jaw_pts = landmarks[[152, 172, 171, 170, 169, 136, 135, 134, 58, 172], :2]
    jaw_n = normalize(np.std(jaw_pts[:, 1]), 2.0, 25.0)
    R["jawline"] = {"score": round(jaw_n, 3), "severity": classify_sev(jaw_n)}

    fh_std = safe_std(landmarks, [55, 107, 66, 105, 65, 52, 53, 46, 124, 156, 70, 63])
    R["forehead"] = {"score": round(normalize(fh_std, 0.001, 0.008), 3), "severity": classify_sev(normalize(fh_std, 0.001, 0.008))}

    gl_std = safe_std(landmarks, [168, 6, 197, 195, 5, 4, 8, 9])
    R["glabella"] = {"score": round(normalize(gl_std, 0.001, 0.008), 3), "severity": classify_sev(normalize(gl_std, 0.001, 0.008))}

    cr_std = safe_std(landmarks, [33, 133, 246, 161, 160, 159, 263, 362, 466, 388, 387, 386])
    R["crowsfeet"] = {"score": round(normalize(cr_std, 0.001, 0.01), 3), "severity": classify_sev(normalize(cr_std, 0.001, 0.01))}

    chin_w = abs(landmarks[172, 0] - landmarks[397, 0])
    chin_l = abs(landmarks[152, 1] - subnasale_y)
    chin_s = normalize(chin_l / (chin_w + 1e-6), 0.3, 1.0)
    R["chin"] = {"score": round(chin_s, 3), "severity": classify_sev(chin_s)}

    sk_z = landmarks[[50, 280, 205, 425, 117, 346, 187, 411], 2]
    sk_cv = np.std(sk_z) / (np.mean(np.abs(sk_z)) + 1e-6)
    sk_n = normalize(sk_cv, 0.05, 0.6)
    R["skin"] = {"score": round(sk_n, 3), "severity": classify_sev(sk_n)}

    return R


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
        score = normalize(float(calf_ratio), 0.2, 0.6)
        R = {
            "detected": True,
            "score": round(score, 3),
            "severity": classify_sev(score),
            "calf_ratio": round(float(calf_ratio), 3),
            "left_knee": (int(lk.x * w), int(lk.y * h)),
            "left_ankle": (int(la.x * w), int(la.y * h)),
            "right_knee": (int(rk.x * w), int(rk.y * h)),
            "right_ankle": (int(ra.x * w), int(ra.y * h)),
        }
    except Exception:
        pass
    return R


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
        score = normalize(bwr, 0.3, 0.7)
        R = {
            "detected": True,
            "score": round(score, 3),
            "severity": classify_sev(score),
            "back_width_ratio": round(float(bwr), 3),
            "shoulder_width_px": round(float(sw), 1),
            "hip_width_px": round(float(hw), 1),
            "shoulder_hip_ratio": round(float(sw / (hw + 1e-6)), 3),
            "back_thickness_ratio": 0.0,
        }
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


def draw_face_annotations(img_bgr, landmarks, analysis):
    img = img_bgr.copy()
    h, w = img.shape[:2]
    COLOR = (0, 220, 255)
    TEXT = (255, 255, 255)
    zones = analysis.get("zones", {})
    ys = {
        "top": int(landmarks[10, 1]),
        "m1": int(landmarks[8, 1]),
        "m2": int(landmarks[94, 1]),
        "bot": int(landmarks[152, 1]),
    }
    labels_zh = {
        "top": "髮際",
        "m1": "眉間",
        "m2": "鼻下",
        "bot": "下巴",
    }
    for k, y in ys.items():
        cv2.line(img, (0, y), (w, y), COLOR, 1)
        cv2.putText(img, labels_zh[k], (5, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
    for x in [landmarks[234, 0], landmarks[33, 0], landmarks[133, 0],
               landmarks[362, 0], landmarks[263, 0], landmarks[454, 0]]:
        cv2.line(img, (int(x), 0), (int(x), h), (200, 200, 0), 1)
    for idx in [10, 8, 94, 152, 33, 133, 362, 263, 234, 454, 1]:
        cv2.circle(img, (int(landmarks[idx, 0]), int(landmarks[idx, 1])), 3, (0, 100, 255), -1)
    zone_labels = [
        (ys["top"], ys["m1"], "上庭 {:.1%}".format(zones.get("upper_ratio", 0))),
        (ys["m1"], ys["m2"], "中庭 {:.1%}".format(zones.get("middle_ratio", 0))),
        (ys["m2"], ys["bot"], "下庭 {:.1%}".format(zones.get("lower_ratio", 0))),
    ]
    for y1, y2, text in zone_labels:
        cv2.putText(img, text, (w - 140, (y1 + y2) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR, 1, cv2.LINE_AA)
    return img


def generate_recommendations(analysis):
    recs = []
    for prob_key, prob_name in PROBLEM_ZH.items():
        data = analysis.get(prob_key, {})
        if not data or "score" not in data:
            continue
        score = data["score"]
        if score < SCORE_THRESHOLD:
            continue
        sev = data["severity"]
        if prob_key not in PROBLEM_TO_PRODUCTS:
            continue
        primary = []
        alternatives = []
        for pk, is_primary in PROBLEM_TO_PRODUCTS[prob_key]:
            if pk not in PRODUCT_ZH:
                continue
            name, brand, layer, method, effect = PRODUCT_ZH[pk]
            dose = PRODUCT_DB[pk]["dose_mid"].get(sev, "依醫師評估")
            info = {
                "key": pk,
                "name": name,
                "brand": brand,
                "dose": dose,
                "layer": layer,
                "method": method,
                "effect": effect,
            }
            if is_primary:
                primary.append(info)
            else:
                alternatives.append(info)
        recs.append({
            "key": prob_key,
            "name": prob_name,
            "score": score,
            "severity": sev,
            "primary": primary,
            "alternatives": alternatives,
        })
    return recs


def physio_analysis(analysis):
    readings = []
    zones = analysis.get("zones", {})
    dominant = zones.get("dominant", "")
    zone_data = [
        ("upper", "上庭", zones.get("upper_ratio", 0.333),
         "上庭寬闊，早年運佳，智慧過人，父母緣深厚",
         "若額頭有皺紋或過窄，可透過肉毒放鬆額肌改善抬頭紋，或以玻尿酸填充額頭弧度，讓上庭更飽滿圓潤，提升整體氣場。",
         ["肉毒毒素（抬頭紋）", "玻尿酸（全臉保濕 VOLITE）"]),
        ("middle", "中庭", zones.get("middle_ratio", 0.333),
         "中庭均衡，中年事業運旺，適合創業或擔任管理職",
         "若鼻樑較低或法令紋明顯，可透過玻尿酸墊高鼻樑、填充法令紋，讓中庭比例更完美，面相上增強事業運與威嚴感。",
         ["玻尿酸（法令紋 VOLUMA）", "玻尿酸（法令紋 VOLIFT）"]),
        ("lower", "下庭", zones.get("lower_ratio", 0.333),
         "下庭豐厚，晚年運佳，福氣深厚，子孫有緣",
         "若下巴短或下頜緣不清晰，可透過玻尿酸墊下巴或埋線提拉，讓下庭比例更協調，面相上加強晚年財運與福氣。",
         ["玻尿酸（下巴）", "玻尿酸（下頜輪廓 VOLUX）"]),
    ]
    for zone_key, zone_name, ratio, good, improve, products in zone_data:
        is_dom = (dominant == zone_key)
        readings.append({
            "aspect": zone_name + "分析",
            "zone": zone_name,
            "ratio": "{:.1%}".format(ratio),
            "reading": good if is_dom else zone_name + "比例偏低，建議調整",
            "improve": improve,
            "products": products,
            "is_dominant": is_dom,
        })

    inter_r = analysis.get("five_eyes", {}).get("inter_ratio", 1.0)
    if inter_r < 0.85:
        readings.append({
            "aspect": "眼距偏窄",
            "zone": "",
            "ratio": "{:.2f}".format(inter_r),
            "reading": "眼距較窄，個性敏銳反應快，容易急躁",
            "improve": "可透過眉形調整或淚溝填充，視覺上拉寬眼距，讓面部比例更協調。",
            "products": ["玻尿酸（淚溝 VOLBELLA）"],
            "is_dominant": False,
        })
    elif inter_r > 1.2:
        readings.append({
            "aspect": "眼距寬廣",
            "zone": "",
            "ratio": "{:.2f}".format(inter_r),
            "reading": "眼距寬廣，心胸寬大，人緣佳，適合公關外交",
            "improve": "眼距已相當理想，可搭配眼周保養或皮秒改善眼周膚質。",
            "products": ["皮秒雷射"],
            "is_dominant": True,
        })

    nasolab_s = analysis.get("nasolabial", {}).get("score", 0)
    if nasolab_s > 0.5:
        readings.append({
            "aspect": "法令紋顯現",
            "zone": "",
            "ratio": "",
            "reading": "法令紋深象徵威嚴與領導力，主掌大局之相",
            "improve": "法令紋象徵威嚴，但視覺顯老。可透過玻尿酸填充或舒顏萃刺激膠原新生，保持威嚴命格同時外觀更年輕。",
            "products": ["玻尿酸（法令紋 VOLUMA）", "舒顏萃 Sculptra"],
            "is_dominant": False,
        })
    return readings


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
.ptag { background:#1a3a2a; color:#2ecc71; border:1px solid #2ecc71; padding:2px 8px; border-radius:12px; font-size:0.78rem; }
.atag { background:#1a1a3a; color:#8888ff; border:1px solid #8888ff; padding:2px 8px; border-radius:12px; font-size:0.78rem; }
.stButton > button { background: linear-gradient(135deg,#e0c44a,#c0a030) !important; color:#000 !important; font-weight:700 !important; border-radius:8px !important; border:none !important; }
div[data-testid="metric-container"] label { color:#a0b4c8 !important; }
div[data-testid="metric-container"] [data-testid="metric-value"] { color:#ffd700 !important; }
div[data-testid="stInfo"] p { color:#c0d8f0 !important; }
div[data-testid="stSuccess"] p { color:#c0f0c0 !important; }
div[data-testid="stWarning"] p { color:#f0d080 !important; }
.stExpander summary p { color:#ffd700 !important; font-weight:600; }
</style>
""", unsafe_allow_html=True)


def badge_html(sev):
    cls = {"mild": "bmild", "moderate": "bmod", "severe": "bsev"}.get(sev, "bmild")
    zh = sev_zh(sev)
    return "<span class='{}'>{}</span>".format(cls, zh)


def main():
    apply_styles()
    st.title("AI 智能醫美面診輔助系統")
    st.caption("上傳照片，AI 自動分析並生成個人化治療建議")
    st.markdown("---")

    mode = st.radio(
        "**選擇分析模式**",
        ["臉部分析", "小腿肌肉分析", "背部分析"],
        horizontal=True,
    )

    # ========== 臉部分析 ==========
    if mode == "臉部分析":
        st.markdown("---")
        st.subheader("上傳臉部照片（最多5個角度）")
        st.info("請上傳：正面、左側45度、左側90度、右側45度、右側90度（至少需要正面）")

        ANGLES = {
            "front":   {"expected": 0,   "tol": 15, "label": "① 正面（0度）"},
            "left45":  {"expected": -35, "tol": 18, "label": "② 左側 45度"},
            "left90":  {"expected": -70, "tol": 20, "label": "③ 左側 90度"},
            "right45": {"expected": 35,  "tol": 18, "label": "④ 右側 45度"},
            "right90": {"expected": 70,  "tol": 20, "label": "⑤ 右側 90度"},
        }

        c1, c2, c3, c4, c5 = st.columns(5)
        cols = {"front": c1, "left45": c2, "left90": c3, "right45": c4, "right90": c5}
        uploads = {}

        for key, col in cols.items():
            with col:
                st.markdown(
                    "<span class='ulabel'>{}</span>".format(ANGLES[key]["label"]),
                    unsafe_allow_html=True,
                )
                f = st.file_uploader(
                    "", type=["jpg", "jpeg", "png"], key=key, label_visibility="collapsed"
                )
                if f:
                    uploads[key] = Image.open(f).convert("RGB")
                    st.image(uploads[key], use_container_width=True)

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
                        st.markdown(
                            "<div class='afail'>無法偵測人臉</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        yaw = estimate_yaw(lm)
                        ok = abs(yaw - cfg["expected"]) <= cfg["tol"]
                        cls = "aok" if ok else "afail"
                        icon = "OK" if ok else "角度偏差"
                        st.markdown(
                            "<div class='{}'>{} Yaw={:.1f}</div>".format(cls, icon, yaw),
                            unsafe_allow_html=True,
                        )

        st.markdown("---")

        if "front" in uploads:
            if st.button("開始 AI 面診分析", use_container_width=True):
                with st.spinner("AI 分析中，請稍候..."):
                    try:
                        front_bgr = pil_to_cv2(resize_image(uploads["front"], 800))
                        lm_front, _ = get_landmarks(front_bgr)
                        if lm_front is None:
                            st.error("無法偵測正面人臉，請上傳更清晰的照片。")
                            return
                        analysis = analyze_face(lm_front, front_bgr)
                        img_ann = draw_face_annotations(front_bgr, lm_front, analysis)
                        recs = generate_recommendations(analysis)
                        physio = physio_analysis(analysis)

                        st.success("分析完成！")

                        st.subheader("三庭五眼標註圖")
                        st.image(cv2_to_pil(img_ann), use_container_width=True)

                        st.subheader("三庭比例")
                        zones = analysis["zones"]
                        zc1, zc2, zc3 = st.columns(3)
                        dom_map = {"upper": "上庭", "middle": "中庭", "lower": "下庭"}
                        zc1.metric("上庭", "{:.1%}".format(zones["upper_ratio"]),
                                   delta="{:+.1%}".format(zones["upper_ratio"] - 0.333))
                        zc2.metric("中庭", "{:.1%}".format(zones["middle_ratio"]),
                                   delta="{:+.1%}".format(zones["middle_ratio"] - 0.333))
                        zc3.metric("下庭", "{:.1%}".format(zones["lower_ratio"]),
                                   delta="{:+.1%}".format(zones["lower_ratio"] - 0.333))
                        st.info("主導分區：{} ｜ 理想各區均為 33.3%".format(dom_map.get(zones["dominant"], zones["dominant"])))

                        st.subheader("五眼比例")
                        fe = analysis["five_eyes"]
                        fc1, fc2 = st.columns(2)
                        fc1.metric("五眼比例", "{:.2f}".format(fe["five_eye_ratio"]),
                                   delta="{:+.2f}（理想=1.0）".format(fe["five_eye_ratio"] - 1.0))
                        fc2.metric("眼距比", "{:.2f}".format(fe["inter_ratio"]),
                                   delta="{:+.2f}（理想=1.0）".format(fe["inter_ratio"] - 1.0))

                        st.subheader("問題診斷")
                        scored = sorted(
                            [(k, v) for k, v in analysis.items()
                             if isinstance(v, dict) and "score" in v],
                            key=lambda x: x[1]["score"], reverse=True,
                        )
                        for prob_key, data in scored:
                            name = PROBLEM_ZH.get(prob_key, prob_key)
                            sev = data["severity"]
                            sev_cls = {"mild": "bmild", "moderate": "bmod", "severe": "bsev"}.get(sev, "bmild")
                            st.markdown(
                                "<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>"
                                "<span style='color:#fff;font-weight:700;min-width:80px;'>{}</span>"
                                "<span class='{}'>{}</span>"
                                "<span style='color:#aaa;font-size:0.85rem;'>{:.2f}</span>"
                                "</div>".format(name, sev_cls, sev_zh(sev), data["score"]),
                                unsafe_allow_html=True,
                            )
                            st.progress(data["score"])

                        st.subheader("個人化治療建議")
                        st.markdown(
                            "<p style='color:#c0d8e8;font-size:0.88rem;'>"
                            "首選 = 最推薦療程 ｜ 備選 = 輔助或替代療程</p>",
                            unsafe_allow_html=True,
                        )
                        if recs:
                            for rec in recs:
                                label = "{} — {} （{:.2f}）".format(
                                    rec["name"], sev_zh(rec["severity"]), rec["score"])
                                with st.expander(label, expanded=rec["score"] > 0.5):
                                    if rec["primary"]:
                                        st.markdown(
                                            "<div style='color:#2ecc71;font-weight:700;margin-bottom:6px;'>"
                                            "首選治療方案</div>",
                                            unsafe_allow_html=True,
                                        )
                                        for prod in rec["primary"]:
                                            st.markdown(
                                                "<div style='background:#0a1f10;border-radius:8px;"
                                                "padding:14px;margin:6px 0;border-left:4px solid #2ecc71;'>"
                                                "<div style='color:#ffd700;font-weight:700;margin-bottom:6px;'>"
                                                "{} <span class='ptag'>首選</span></div>"
                                                "<div style='color:#c0e8c0;font-size:0.88rem;line-height:1.8;'>"
                                                "品牌：{}<br>劑量：{}<br>注射層次：{}<br>注射方式：{}<br>預估效果：{}"
                                                "</div></div>".format(
                                                    prod["name"], prod["brand"], prod["dose"],
                                                    prod["layer"], prod["method"], prod["effect"],
                                                ),
                                                unsafe_allow_html=True,
                                            )
                                    if rec["alternatives"]:
                                        st.markdown(
                                            "<div style='color:#8888ff;font-weight:700;margin:10px 0 6px;'>"
                                            "備選治療方案</div>",
                                            unsafe_allow_html=True,
                                        )
                                        acols = st.columns(min(len(rec["alternatives"]), 3))
                                        for j, prod in enumerate(rec["alternatives"]):
                                            with acols[j % len(acols)]:
                                                st.markdown(
                                                    "<div style='background:#0a0a20;border-radius:8px;"
                                                    "padding:10px;border-left:3px solid #8888ff;'>"
                                                    "<div style='color:#c0c8ff;font-weight:600;'>"
                                                    "{} <span class='atag'>備選</span></div>"
                                                    "<div style='color:#8090c0;font-size:0.82rem;'>"
                                                    "劑量：{}<br>{}"
                                                    "</div></div>".format(
                                                        prod["name"], prod["dose"], prod["effect"],
                                                    ),
                                                    unsafe_allow_html=True,
                                                )
                        else:
                            st.success("未偵測到明顯問題，繼續維持良好保養即可！")

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
                                    r["improve"], "、".join(r.get("products", [])),
                                ),
                                unsafe_allow_html=True,
                            )

                        st.subheader("下載分析報告")
                        lines = [
                            "AI 智能醫美面診報告",
                            "日期：{}".format(datetime.now().strftime("%Y-%m-%d %H:%M")),
                            "",
                            "=== 三庭比例 ===",
                        ]
                        for k in ("upper_ratio", "middle_ratio", "lower_ratio", "dominant"):
                            lines.append("  {}: {}".format(k, zones.get(k, "")))
                        lines += ["", "=== 問題診斷 ==="]
                        for prob_key, data in scored:
                            lines.append("  {}: {} ({:.2f})".format(
                                PROBLEM_ZH.get(prob_key, prob_key), sev_zh(data["severity"]), data["score"]))
                        lines += ["", "=== 治療建議 ==="]
                        for rec in recs:
                            lines.append("\n[{}] {}".format(rec["name"], sev_zh(rec["severity"])))
                            for p in rec["primary"]:
                                lines.append("  首選：{} / {}".format(p["name"], p["dose"]))
                            for p in rec["alternatives"]:
                                lines.append("  備選：{}".format(p["name"]))
                        lines += ["", "免責聲明：本報告為AI輔助參考，不構成醫療建議，請諮詢合法執照醫師。"]
                        st.download_button(
                            "下載完整分析報告（TXT）",
                            data="\n".join(lines).encode("utf-8"),
                            file_name="face_report_{}.txt".format(
                                datetime.now().strftime("%Y%m%d_%H%M")),
                            mime="text/plain",
                            use_container_width=True,
                        )
                        st.warning("免責聲明：本系統為輔助參考工具，不構成醫療建議，實際治療請諮詢合法執照醫師。")

                    except Exception as e:
                        st.error("分析錯誤：{}".format(e))
                        st.exception(e)
        else:
            st.info("請先上傳正面照片後開始分析。")

    # ========== 小腿肌肉分析 ==========
    elif mode == "小腿肌肉分析":
        st.markdown("---")
        st.subheader("小腿肌肉肥大分析")
        st.info("拍攝要求：全身正面站立，膝蓋至腳踝完整入鏡，光線充足。")

        col1, col2 = st.columns(2)
        calf_normal = None
        calf_tiptoe = None
        with col1:
            st.markdown("<span class='ulabel'>① 自然站姿（正面）</span>", unsafe_allow_html=True)
            f1 = st.file_uploader(
                "", type=["jpg", "jpeg", "png"], key="calf_n", label_visibility="collapsed"
            )
            if f1:
                calf_normal = Image.open(f1).convert("RGB")
                st.image(calf_normal, use_container_width=True, caption="自然站姿")
        with col2:
            st.markdown("<span class='ulabel'>② 墊腳尖（選填）</span>", unsafe_allow_html=True)
            f2 = st.file_uploader(
                "", type=["jpg", "jpeg", "png"], key="calf_t", label_visibility="collapsed"
            )
            if f2:
                calf_tiptoe = Image.open(f2).convert("RGB")
                st.image(calf_tiptoe, use_container_width=True, caption="墊腳尖")

        if calf_normal:
            if st.button("開始小腿分析", use_container_width=True):
                with st.spinner("分析中..."):
                    try:
                        img_bgr = pil_to_cv2(resize_image(calf_normal, 800))
                        result = analyze_calf(img_bgr)
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

                        st.success("分析完成！")
                        st.image(cv2_to_pil(ann), use_container_width=True, caption="小腿關鍵點標註")

                        if result["detected"]:
                            sev = result["severity"]
                            sev_cls = {"mild": "bmild", "moderate": "bmod", "severe": "bsev"}.get(sev, "bmild")
                            st.markdown(
                                "<div style='background:#1a1a2e;border-radius:10px;padding:16px;"
                                "border-left:4px solid #e0c44a;margin:12px 0;'>"
                                "<div style='color:#ffd700;font-size:1.1rem;font-weight:700;margin-bottom:8px;'>"
                                "小腿肌肉肥大程度：<span class='{}'>{}</span></div>"
                                "<div style='color:#c0d0e0;'>寬度比例：{:.3f} ｜ AI 評分：{:.2f}/1.00</div>"
                                "</div>".format(sev_cls, sev_zh(sev), result["calf_ratio"], result["score"]),
                                unsafe_allow_html=True,
                            )
                            st.progress(result["score"])

                            prod_key = "botox_calf"
                            name, brand, layer, method, effect = PRODUCT_ZH[prod_key]
                            dose = PRODUCT_DB[prod_key]["dose_mid"].get(sev, "依醫師評估")
                            st.subheader("小腿肉毒毒素治療建議")
                            st.markdown(
                                "<div style='background:#0a1f10;border-radius:10px;padding:16px;"
                                "border-left:4px solid #2ecc71;'>"
                                "<div style='color:#ffd700;font-size:1.05rem;font-weight:700;margin-bottom:10px;'>"
                                "腓腸肌肉毒毒素注射</div>"
                                "<div style='color:#c0e8c0;font-size:0.9rem;line-height:2;'>"
                                "品牌：{}<br>"
                                "建議劑量（{}）：<span style='color:#ffd700;font-weight:700;'>{}</span><br>"
                                "注射層次：{}<br>注射方式：{}<br>預估效果：{}"
                                "</div></div>".format(brand, sev_zh(sev), dose, layer, method, effect),
                                unsafe_allow_html=True,
                            )

                            st.subheader("治療注意事項")
                            tips = [
                                "治療前2週停止高強度腿部訓練，避免肌肉過度充血影響藥效",
                                "注射後4-6小時內請勿按摩注射部位",
                                "注射後24小時內避免劇烈運動、三溫暖及飲酒",
                                "建議每6-9個月維持一次療程以保持效果",
                                "若有長期穿高跟鞋習慣，建議同時調整姿勢以配合治療",
                                "可搭配小腿拉伸伸展運動，加速肌肉放鬆效果更佳",
                            ]
                            for tip in tips:
                                st.markdown(
                                    "<div style='color:#c0d0e0;padding:3px 0;'>• {}</div>".format(tip),
                                    unsafe_allow_html=True,
                                )

                            if calf_tiptoe:
                                st.subheader("墊腳尖對比分析")
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
                                    st.image(cv2_to_pil(ann), caption="自然站姿", use_container_width=True)
                                    st.metric("肌肉評分", "{:.2f}".format(result["score"]))
                                with cb:
                                    st.image(cv2_to_pil(tip_ann), caption="墊腳尖", use_container_width=True)
                                    if tip_result["detected"]:
                                        delta = tip_result["score"] - result["score"]
                                        st.metric("肌肉評分", "{:.2f}".format(tip_result["score"]),
                                                  delta="{:+.2f}".format(delta))
                                if result["detected"] and tip_result.get("detected"):
                                    if tip_result["score"] - result["score"] > 0.1:
                                        st.info("墊腳尖時肌肉明顯收縮，腓腸肌活躍度高，建議治療劑量可略微增加。")
                                    else:
                                        st.info("兩姿勢評分差異不大，肌肉肥大主要為靜態增大型，標準劑量即可。")
                        else:
                            st.warning("無法偵測小腿關鍵點，請確認照片包含膝蓋至腳踝完整範圍，並保持光線充足。")

                    except Exception as e:
                        st.error("分析錯誤：{}".format(e))
                        st.exception(e)
        else:
            st.info("請上傳自然站姿照片以開始分析。")

    # ========== 背部分析 ==========
    elif mode == "背部分析":
        st.markdown("---")
        st.subheader("背部肌肉與輪廓分析")
        st.info("請上傳：正背面（背對鏡頭站立）及側背面（側面站立，選填）")

        col1, col2 = st.columns(2)
        back_front = None
        back_side = None
        with col1:
            st.markdown("<span class='ulabel'>① 正背面（背對鏡頭）</span>", unsafe_allow_html=True)
            bf = st.file_uploader(
                "", type=["jpg", "jpeg", "png"], key="back_f", label_visibility="collapsed"
            )
            if bf:
                back_front = Image.open(bf).convert("RGB")
                st.image(back_front, use_container_width=True, caption="正背面")
        with col2:
            st.markdown("<span class='ulabel'>② 側背面（選填）</span>", unsafe_allow_html=True)
            bs = st.file_uploader(
                "", type=["jpg", "jpeg", "png"], key="back_s", label_visibility="collapsed"
            )
            if bs:
                back_side = Image.open(bs).convert("RGB")
                st.image(back_side, use_container_width=True, caption="側背面")

        if back_front:
            if st.button("開始背部分析", use_container_width=True):
                with st.spinner("分析中..."):
                    try:
                        front_bgr = pil_to_cv2(resize_image(back_front, 800))
                        side_bgr = pil_to_cv2(resize_image(back_side, 800)) if back_side else None
                        back_result = analyze_back(front_bgr, side_bgr)

                        st.success("分析完成！")

                        if back_result["detected"]:
                            sev = back_result["severity"]
                            sev_cls = {"mild": "bmild", "moderate": "bmod", "severe": "bsev"}.get(sev, "bmild")
                            st.markdown(
                                "<div style='background:#1a1a2e;border-radius:10px;padding:16px;"
                                "border-left:4px solid #e0c44a;margin:12px 0;'>"
                                "<div style='color:#ffd700;font-size:1.1rem;font-weight:700;margin-bottom:8px;'>"
                                "背部評估：<span class='{}'>{}</span></div>"
                                "<div style='color:#c0d0e0;'>"
                                "背部寬度比例：{:.2f} ｜ 肩臀比：{:.2f}</div>"
                                "</div>".format(
                                    sev_cls, sev_zh(sev),
                                    back_result.get("back_width_ratio", 0),
                                    back_result.get("shoulder_hip_ratio", 0),
                                ),
                                unsafe_allow_html=True,
                            )
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
                                    ("背部肉毒毒素（豎脊肌放鬆）", "針對背部豎脊肌過度發達或肌肉緊繃，放鬆肌肉改善背部寬厚感", "每側 50-100U", "2-4週改善，維持4-6個月"),
                                    ("背部溶脂針", "針對背部脂肪堆積，溶解局部脂肪讓背部線條更俐落", "4-8ml/療程", "4-8週改善，效果持久"),
                                    ("背部音波拉提", "改善背部皮膚鬆弛，緊緻背部輪廓", "300-500發", "3-6月顯效，維持12-18個月"),
                                ]
                            else:
                                recs_back = [
                                    ("背部水光注射（保濕）", "改善背部皮膚乾燥粗糙，提升膚質與光澤", "2-4ml", "即時保濕，維持4-6個月"),
                                    ("背部皮秒雷射", "改善背部色斑、毛孔粗大、膚色不均", "1-3次療程", "每4-6週一次"),
                                    ("背部肉毒毒素（局部雕塑）", "針對局部肌肉輕度發達做精細雕塑", "每側 30-50U", "2-4週改善，維持4-6個月"),
                                ]
                            for name, desc, dose, effect in recs_back:
                                st.markdown(
                                    "<div style='background:#0d1a20;border-radius:8px;padding:14px;"
                                    "margin:8px 0;border-left:4px solid #e0c44a;'>"
                                    "<div style='color:#ffd700;font-weight:700;margin-bottom:6px;'>{}</div>"
                                    "<div style='color:#c0d0e0;font-size:0.88rem;margin-bottom:4px;'>{}</div>"
                                    "<div style='color:#90a0b0;font-size:0.84rem;'>劑量：{} ｜ 效果：{}</div>"
                                    "</div>".format(name, desc, dose, effect),
                                    unsafe_allow_html=True,
                                )
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
