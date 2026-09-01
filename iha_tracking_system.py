#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
TEKNOFEST SAVAŞAN İHA - TAM TAKİP VE KONTROL SİSTEMİ
═══════════════════════════════════════════════════════════════════════════════════

Bileşenler:
├── TESPİT    : YOLOv11-s + TensorRT (yüksek FPS, düşük gecikme)
├── TAKİP     : SORT (Kalman Filter + Hungarian Algorithm)
├── KONTROL   : PID Kontrolcüsü (Yaw, Pitch, Roll, Hız)
└── DONANIM   : Jetson Nano optimize

Özellikler:
- 4 saniyelik kesintisiz kilitlenme takibi
- Ekran merkezi - hedef merkezi hata hesaplama
- Seri port üzerinden uçuş kontrolcüsüne komut gönderme
- Gerçek zamanlı performans izleme

Kullanım:
    python iha_tracking_system.py

Çıkış için 'q' tuşuna basın.

═══════════════════════════════════════════════════════════════════════════════════
"""

import cv2
import torch
import numpy as np
import time
import os
import sys
import traceback
import threading
import queue
from collections import deque
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union
from enum import Enum

# Opsiyonel: Seri haberleşme
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠️  pyserial kurulu değil, seri haberleşme devre dışı")

# YOLO
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════════════════════════
# YAPILANDIRMA
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass
class SystemConfig:
    """Sistem yapılandırması"""
    # YOLO Ayarları
    yolo_model_path: str = ""  # Otomatik belirlenir
    yolo_img_size: int = 256   # Düşük = Hızlı (160, 192, 256, 320)
    yolo_conf_threshold: float = 0.4
    yolo_skip_frames: int = 2  # Her N frame'de bir YOLO çalıştır (2-3 önerilir)
    yolo_half: bool = True     # FP16 kullan (hız artışı)
    
    # Takip Ayarları (ByteTrack + Kalman)
    sort_max_age: int = 45        # Frame skip için artırıldı
    sort_min_hits: int = 2        # Daha hızlı onay
    sort_iou_threshold: float = 0.25
    max_objects: int = 10

    # ByteTrack iki asamali eslestirme esikleri
    #   >= high            : yeni iz baslatabilir, 1. turda eslestirilir
    #   low <= s < high    : yeni iz BASLATMAZ, yalniz 2. turda mevcut izleri
    #                        ayakta tutar (kapanma anini kurtaran adim)
    #   < low              : tamamen yoksayilir
    # yolo_conf_threshold bu yuzden DUSUK tutulur; asil eleme burada yapilir.
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    track_low_iou_thresh: float = 0.15   # 2. turda esik gevsetilir
    
    # Kilitlenme Ayarları
    lock_duration_required: float = 4.0  # Saniye cinsinden (yarışma kuralı)
    lock_lost_timeout: float = 0.8       # Frame skip için tolerans artırıldı
    
    # PID Kontrolcü Ayarları
    pid_yaw_kp: float = 0.5
    pid_yaw_ki: float = 0.01
    pid_yaw_kd: float = 0.1
    
    pid_pitch_kp: float = 0.5
    pid_pitch_ki: float = 0.01
    pid_pitch_kd: float = 0.1
    
    pid_roll_kp: float = 0.3
    pid_roll_ki: float = 0.005
    pid_roll_kd: float = 0.05
    
    pid_speed_kp: float = 0.4
    pid_speed_ki: float = 0.02
    pid_speed_kd: float = 0.08
    
    # Kontrol Limitleri (derece/saniye veya m/s)
    max_yaw_rate: float = 90.0
    max_pitch_rate: float = 45.0
    max_roll_rate: float = 30.0
    max_speed: float = 15.0
    
    # Seri Port Ayarları
    serial_port: str = "/dev/ttyTHS1"  # Jetson Nano UART
    serial_baudrate: int = 115200
    serial_enabled: bool = False
    
    # Görselleştirme
    show_display: bool = True
    show_trails: bool = True
    trail_length: int = 20     # Azaltıldı (hız için)
    
    # Kamera / görüntü kaynağı
    # camera_id: kamera indeksi (0, 1, ...) VEYA bir video dosyası yolu.
    # Dosya verildiginde kare atlanmaz ve akis sonunda program temiz kapanir.
    camera_id: Union[int, str] = 0
    camera_width: int = 640
    camera_height: int = 480

    # Threading
    # Yalniz CANLI kamerada anlamli. Dosya kaynaginda VideoSource bunu zaten
    # yok sayar; dosyada threadli okuma kare atlatir ve CPU'yu bosuna doldurur.
    use_threading: bool = True

    # Kayit
    record_path: str = ""   # bos degilse islenmis goruntu buraya yazilir

    # Tespit Filtreleri
    yolo_classes: tuple = (0,)    # Sadece bu class ID'leri takip et
    min_box_area: int = 50        # Çok küçük parazitleri yoksay
    max_box_area_percent: float = 0.8  # Ekranı kaplayan hatalı bbox'ları yoksay
    min_aspect_ratio: float = 0.2 # Aşırı ince/uzun bbox'ları yoksay
    max_aspect_ratio: float = 5.0


# ═══════════════════════════════════════════════════════════════════════════════════
# THREADED KAMERA OKUYUCU (FPS Artışı)
# ═══════════════════════════════════════════════════════════════════════════════════

class VideoSource:
    """
    Görüntü kaynağı. Canlı kamera ile video dosyasını AYRI ele alır.

    NEDEN AYRI:
      Canlı kamerada amaç HER ZAMAN EN GÜNCEL kareyi almaktır — YOLO bir kareyi
      işlerken gelen ara kareler atılmalı, yoksa görüntü gerçeğin gerisine düşer.
      Bunun için ayrı bir thread sürekli okuyup en sonuncuyu tutar.

      Video dosyasında ise tam tersi: hiçbir kare atlanmamalı ve oynatma kendi
      hızında ilerlemeli. Dosyada cv2.read() anında döndüğü için threadli okuma
      bir çekirdeği tam doldurup videoyu sonuna kadar koştururdu.

    Dosya kaynağında bu yüzden thread kullanılmaz; okuma doğrudan yapılır.
    Akış bittiğinde eof işaretlenir — çağıran taraf döngüden çıkabilsin diye.
    """

    def __init__(self, source, width: int = 640, height: int = 480):
        # "0" gibi bir dizge geldiyse kamera indeksine çevir
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self.is_file = isinstance(source, str)
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Görüntü kaynağı açılamadı: {source}")

        if not self.is_file:
            # Yalnız canlı kamerada anlamlı: küçük tampon = düşük gecikme
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.eof = False
        self.frame = None
        self.ret = False
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = None

        if not self.is_file:
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        """Canlı kamera: sürekli oku, en sonuncuyu tut."""
        basarisiz = 0
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                # Kamera çekildi ya da sürücü hata verdi. Sonsuz hızda dönmemek
                # icin nefes al; arka arkaya çok olursa akışı bitmiş say.
                basarisiz += 1
                if basarisiz > 100:
                    with self.lock:
                        self.eof = True
                    return
                time.sleep(0.01)
                continue
            basarisiz = 0
            with self.lock:
                self.ret = True
                self.frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.is_file:
            ret, frame = self.cap.read()
            if not ret:
                self.eof = True
                return False, None
            return True, frame

        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def is_finished(self) -> bool:
        """Akış kalıcı olarak bitti mi (dosya sonu / kamera koptu)."""
        with self.lock:
            return self.eof

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def release(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.cap.release()


# Geriye dönük uyumluluk: eski ad hâlâ çalışsın
ThreadedCamera = VideoSource


class LockState(Enum):
    """Kilitlenme durumu"""
    SEARCHING = "ARAMA"       # Hedef aranıyor
    TRACKING = "TAKİP"        # Hedef takip ediliyor (kilit yok)
    LOCKING = "KİLİTLENİYOR"  # Kilitlenme süresi dolmadı
    LOCKED = "KİLİTLİ"        # 4 saniye tamamlandı, hedef kilitli


# ═══════════════════════════════════════════════════════════════════════════════════
# KALMAN FİLTRESİ (SORT için - filterpy bağımlılığı olmadan)
# ═══════════════════════════════════════════════════════════════════════════════════

class SimpleKalmanFilter:
    """
    Basit Kalman Filter implementasyonu (filterpy bağımlılığı olmadan).
    """
    def __init__(self, dim_x: int, dim_z: int):
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # State vector
        self.x = np.zeros((dim_x, 1))
        
        # State covariance matrix
        self.P = np.eye(dim_x)
        
        # Process noise covariance
        self.Q = np.eye(dim_x)
        
        # Measurement noise covariance
        self.R = np.eye(dim_z)
        
        # State transition matrix
        self.F = np.eye(dim_x)
        
        # Measurement matrix
        self.H = np.zeros((dim_z, dim_x))
    
    def predict(self):
        """Tahmin adımı"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z: np.ndarray):
        """Güncelleme adımı"""
        z = z.reshape((self.dim_z, 1))
        
        # Innovation (measurement residual)
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P


class KalmanBoxTracker:
    """
    Kalman Filter ile tek nesne takibi.
    State: [x, y, s, r, vx, vy, vs]
    - x, y: bbox merkezi
    - s: alan (width * height)
    - r: aspect ratio (width / height)
    - vx, vy, vs: hızlar
    """
    count = 0
    
    def __init__(self, bbox: np.ndarray):
        """
        Args:
            bbox: [x1, y1, x2, y2] formatında bounding box
        """
        # 7 state, 4 measurement
        self.kf = SimpleKalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float64)
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float64)
        
        # Measurement noise
        self.kf.R = np.eye(4, dtype=np.float64)
        self.kf.R[2:, 2:] *= 10.0
        
        # Process noise
        self.kf.P = np.eye(7, dtype=np.float64)
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        
        self.kf.Q = np.eye(7, dtype=np.float64)
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = self._bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def _bbox_to_z(self, bbox: np.ndarray) -> np.ndarray:
        """[x1, y1, x2, y2] -> [cx, cy, s, r]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        return np.array([cx, cy, s, r], dtype=np.float64).reshape((4, 1))
    
    def _z_to_bbox(self, z: np.ndarray) -> np.ndarray:
        """[cx, cy, s, r] -> [x1, y1, x2, y2]"""
        # Skaler değerleri düzgün çıkar
        cx = float(z[0].item()) if hasattr(z[0], 'item') else float(z[0])
        cy = float(z[1].item()) if hasattr(z[1], 'item') else float(z[1])
        s = max(float(z[2].item()) if hasattr(z[2], 'item') else float(z[2]), 1.0)
        r = max(float(z[3].item()) if hasattr(z[3], 'item') else float(z[3]), 0.01)
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 1.0
        return np.array([
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0
        ])
    
    def update(self, bbox: np.ndarray):
        """Kalman filter güncelle"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))
    
    def predict(self, yaslandir: bool = True) -> np.ndarray:
        """Sonraki pozisyonu tahmin et.

        yaslandir=False: hareket tahmini yapilir ama iz YASLANDIRILMAZ.

        NEDEN: YOLO'yu her karede degil her N karede bir calistiriyoruz
        (yolo_skip_frames). Ara karelerde tespit YOK ama bu bir KAYIP degil,
        bilincli bir tasarruf. Bu kareleri de kayip saymak iki seyi bozuyordu:

          1. hit_streak her ara karede sifirlaniyordu, dolayisiyla
             hit_streak >= sort_min_hits kosulu ASLA saglanmiyordu
             (varsayilan skip=2, min_hits=2 ile hicbir iz bildirilmiyordu).
          2. time_since_update artiyor, iz "aktif degil" sayilip
             takipcinin cikti kapisindan eleniyordu.

        Gercek kayiplar (YOLO calisti ama hedefi bulamadi) yine yaslandirilir.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if yaslandir:
            if self.time_since_update > 0:
                self.hit_streak = 0
            self.time_since_update += 1

        self.history.append(self._z_to_bbox(self.kf.x))
        return self.history[-1]
    
    def get_state(self) -> np.ndarray:
        """Mevcut bbox [x1, y1, x2, y2]"""
        return self._z_to_bbox(self.kf.x)
    
    def get_center(self) -> Tuple[float, float]:
        """Merkez koordinatları (cx, cy)"""
        bbox = self.get_state()
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


# ═══════════════════════════════════════════════════════════════════════════════════
# SORT TRACKER
# ═══════════════════════════════════════════════════════════════════════════════════

class ByteTrackTracker:
    """
    ByteTrack + Kalman Filter ile çoklu nesne takibi.

    Çalışma Prensibi:
      1. Kalman Filter ile her izin sonraki pozisyonu tahmin edilir
      2. IKI ASAMALI eslestirme (Hungarian, maliyet = 1 - IoU):
           1. tur — yuksek guvenli tespitler  -> tum izler
           2. tur — dusuk guvenli tespitler   -> eslesmeden kalan izler
      3. Eslesmeyen YUKSEK guvenli tespit -> yeni iz
      4. Eslesmeyen iz -> yaslanir, max_age sonra silinir

    2. turun anlami: hedef kismen kapandiginda tespit guveni duser ama nesne
    hala oradadir. Tek asamali bir eslestirici o tespiti atar ve kimlik kopar.
    Ikinci tur onu yakalayip izi ayakta tutar — kilit bu sayede bozulmuyor.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
    
    def _iou_batch(self, bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
        """
        Batch IoU hesaplama (vektörize).
        
        Args:
            bb_test: (N, 4) test bboxes
            bb_gt: (M, 4) ground truth bboxes
        
        Returns:
            (N, M) IoU matrix
        """
        if len(bb_test) == 0 or len(bb_gt) == 0:
            return np.zeros((len(bb_test), len(bb_gt)))
        
        bb_test = np.expand_dims(bb_test, 1)  # (N, 1, 4)
        bb_gt = np.expand_dims(bb_gt, 0)      # (1, M, 4)
        
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        union = area_test + area_gt - intersection
        
        return intersection / np.maximum(union, 1e-6)
    
    def _associate_detections_to_trackers(
        self, 
        detections: np.ndarray, 
        trackers: np.ndarray, 
        iou_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Hungarian algorithm ile optimal eşleştirme.
        
        Returns:
            matches: (K, 2) eşleşen [detection_idx, tracker_idx] çiftleri
            unmatched_detections: eşleşmeyen detection indeksleri
            unmatched_trackers: eşleşmeyen tracker indeksleri
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty(0, dtype=int)
        
        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.empty(0, dtype=int), np.arange(len(trackers))
        
        iou_matrix = self._iou_batch(detections, trackers)
        
        # Hungarian algorithm (cost = 1 - IoU)
        row_indices, col_indices = linear_sum_assignment(1 - iou_matrix)
        
        # Eşleşmeleri filtrele (IoU < threshold olanları ayır)
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(trackers)))
        
        for r, c in zip(row_indices, col_indices):
            if iou_matrix[r, c] >= iou_threshold:
                matches.append([r, c])
                if r in unmatched_detections:
                    unmatched_detections.remove(r)
                if c in unmatched_trackers:
                    unmatched_trackers.remove(c)
        
        return (
            np.array(matches).reshape(-1, 2) if matches else np.empty((0, 2), dtype=int),
            np.array(unmatched_detections),
            np.array(unmatched_trackers)
        )
    
    def update(self, detections: np.ndarray,
               tespit_karesi: bool = True) -> List[Dict]:
        """
        SORT güncelleme.

        Args:
            detections: (N, 5) [x1, y1, x2, y2, score] formatında
            tespit_karesi: Bu karede YOLO calisti mi. False ise izler
                hareket tahmini alir ama yaslandirilmaz — kare atlama
                bilincli bir tasarruf, kayip degil.

        Returns:
            Aktif track'lerin listesi: [{'id': int, 'bbox': [x1,y1,x2,y2], 'score': float}, ...]
        """
        self.frame_count += 1

        # 1. Mevcut tracker'ların tahminini al
        predicted_boxes = np.zeros((len(self.trackers), 4))
        to_delete = []

        for i, trk in enumerate(self.trackers):
            pos = trk.predict(yaslandir=tespit_karesi)
            predicted_boxes[i] = pos
            # Geçersiz tahmin kontrolü
            if np.any(np.isnan(pos)):
                to_delete.append(i)
        
        # Geçersiz tracker'ları sil
        for i in reversed(to_delete):
            self.trackers.pop(i)
            predicted_boxes = np.delete(predicted_boxes, i, axis=0)
        
        # ─── 2. IKI ASAMALI ESLESTIRME (ByteTrack) ──────────────────────────
        #
        # Tek asamali eslestirici, esigin altinda kalan tespitleri atar.
        # Sorun su: hedef kismen kapandiginda ya da hizli hareket ettiginde
        # guven skoru duser — nesne HALA ORADA ama tespit cope gider ve iz
        # kopar. Kimlik degisir, kilit sifirlanir.
        #
        # ByteTrack bunu iki turda cozer:
        #   1. tur — YUKSEK guvenli tespitler tum izlerle eslestirilir
        #   2. tur — eslesmeden KALAN izler, DUSUK guvenli tespitlerle
        #            eslestirilir  (kapanma anini kurtaran adim budur)
        #
        # Yeni iz YALNIZ yuksek guvenli tespitten dogar; dusuk guvenliler
        # sadece mevcut izleri ayakta tutar. Yoksa gurultu iz uretirdi.

        if len(detections) > 0:
            skorlar = detections[:, 4]
            yuksek = np.where(skorlar >= self.config.track_high_thresh)[0]
            dusuk = np.where(
                (skorlar < self.config.track_high_thresh) &
                (skorlar >= self.config.track_low_thresh)
            )[0]
        else:
            yuksek = np.empty(0, dtype=int)
            dusuk = np.empty(0, dtype=int)

        det_yuksek = detections[yuksek, :4] if len(yuksek) else np.empty((0, 4))
        det_dusuk = detections[dusuk, :4] if len(dusuk) else np.empty((0, 4))

        # ── 1. tur: yuksek guven -> tum izler
        m1, eslesmeyen_yuksek, eslesmeyen_izler = \
            self._associate_detections_to_trackers(
                det_yuksek, predicted_boxes, self.config.sort_iou_threshold
            )

        for d, t in m1:
            self.trackers[t].update(det_yuksek[d])

        # ── 2. tur: dusuk guven -> 1. turda eslesmeden kalan izler
        #
        # Kapanma sirasinda kutu bozulup IoU dustugu icin bu turda esik
        # GEVSETILIR. Aday havuzu zaten "kaybedilmek uzere olan izler" ile
        # sinirli oldugu icin yanlis eslestirme riski dusuk.
        if len(det_dusuk) > 0 and len(eslesmeyen_izler) > 0:
            kalan_kutular = predicted_boxes[eslesmeyen_izler]
            m2, _, hala_eslesmeyen = self._associate_detections_to_trackers(
                det_dusuk, kalan_kutular, self.config.track_low_iou_thresh
            )
            for d, t_yerel in m2:
                gercek = eslesmeyen_izler[t_yerel]
                self.trackers[gercek].update(det_dusuk[d])

            eslesmeyen_izler = eslesmeyen_izler[hala_eslesmeyen] \
                if len(hala_eslesmeyen) else np.empty(0, dtype=int)

        # ── 3. Yeni iz: YALNIZ yuksek guvenli, eslesmemis tespitlerden
        for i in eslesmeyen_yuksek:
            if len(self.trackers) < self.config.max_objects:
                self.trackers.append(KalmanBoxTracker(det_yuksek[i]))
        
        # 5. Kayıp tracker'ları temizle
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.config.sort_max_age:
                self.trackers.pop(i)
        
        # 6. Aktif track'leri döndür
        results = []
        for trk in self.trackers:
            if trk.time_since_update < 1 and (trk.hit_streak >= self.config.sort_min_hits or self.frame_count <= self.config.sort_min_hits):
                bbox = trk.get_state()
                results.append({
                    'id': trk.id,
                    'bbox': bbox,
                    'score': 1.0 if trk.hit_streak > 0 else 0.5,
                    'center': trk.get_center()
                })
        
        return results
    
    def get_primary_target(self) -> Optional[Dict]:
        """En büyük veya en yakın hedefi döndür"""
        active_tracks = [t for t in self.trackers if t.time_since_update < 1]
        if not active_tracks:
            return None
        
        # En büyük alana sahip hedef (veya başka bir kriter)
        best_track = max(active_tracks, key=lambda t: (t.get_state()[2] - t.get_state()[0]) * (t.get_state()[3] - t.get_state()[1]))
        bbox = best_track.get_state()
        return {
            'id': best_track.id,
            'bbox': bbox,
            'center': best_track.get_center(),
            'age': best_track.age,
            'hits': best_track.hits
        }


# Geriye donuk uyumluluk: eski ad hala calissin
SortTracker = ByteTrackTracker


# ═══════════════════════════════════════════════════════════════════════════════════
# PID KONTROLCÜ
# ═══════════════════════════════════════════════════════════════════════════════════

class PIDController:
    """
    PID (Proportional-Integral-Derivative) Kontrolcü
    
    Kullanım:
        pid = PIDController(kp=0.5, ki=0.01, kd=0.1)
        output = pid.compute(error, dt)
    """
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_min: float = -1.0, output_max: float = 1.0,
                 integral_limit: float = 100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_limit = integral_limit
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None
    
    def reset(self):
        """PID durumunu sıfırla"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None
    
    def compute(self, error: float, dt: Optional[float] = None) -> float:
        """
        PID çıkışını hesapla.
        
        Args:
            error: Hata (setpoint - actual)
            dt: Zaman farkı (None ise otomatik hesaplanır)
        
        Returns:
            Kontrol çıkışı [-output_max, output_max] aralığında
        """
        current_time = time.time()
        
        if dt is None:
            dt = current_time - self.last_time if self.last_time else 0.01
        
        self.last_time = current_time
        
        # Proportional
        p_term = self.kp * error
        
        # Integral (anti-windup ile)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivative
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative
        self.prev_error = error
        
        # Toplam çıkış
        output = p_term + i_term + d_term
        return np.clip(output, self.output_min, self.output_max)


# ═══════════════════════════════════════════════════════════════════════════════════
# UÇUŞ KONTROL SİSTEMİ
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass
class FlightCommand:
    """Uçuş kontrolcüsüne gönderilecek komut"""
    yaw_rate: float = 0.0      # Derece/saniye
    pitch_rate: float = 0.0    # Derece/saniye
    roll_rate: float = 0.0     # Derece/saniye
    throttle: float = 0.0      # -1.0 ile 1.0 arası
    timestamp: float = field(default_factory=time.time)


class FlightController:
    """
    PID tabanlı uçuş kontrolcüsü.
    
    Hedef merkezini ekran merkezine hizalamak için:
    - Yaw: Yatay sapma kontrolü
    - Pitch: Dikey sapma kontrolü
    - Roll: Yardımcı stabilizasyon
    - Throttle/Speed: Mesafe kontrolü
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # PID kontrolcüleri
        self.yaw_pid = PIDController(
            config.pid_yaw_kp, config.pid_yaw_ki, config.pid_yaw_kd,
            -config.max_yaw_rate, config.max_yaw_rate
        )
        self.pitch_pid = PIDController(
            config.pid_pitch_kp, config.pid_pitch_ki, config.pid_pitch_kd,
            -config.max_pitch_rate, config.max_pitch_rate
        )
        self.roll_pid = PIDController(
            config.pid_roll_kp, config.pid_roll_ki, config.pid_roll_kd,
            -config.max_roll_rate, config.max_roll_rate
        )
        self.speed_pid = PIDController(
            config.pid_speed_kp, config.pid_speed_ki, config.pid_speed_kd,
            -1.0, 1.0
        )
        
        # Ekran merkezi
        self.screen_center_x = config.camera_width / 2
        self.screen_center_y = config.camera_height / 2
        
        # Seri port
        self.serial_port = None
        if config.serial_enabled and SERIAL_AVAILABLE:
            try:
                self.serial_port = serial.Serial(
                    port=config.serial_port,
                    baudrate=config.serial_baudrate,
                    timeout=0.1
                )
                print(f"✅ Seri port açıldı: {config.serial_port}")
            except Exception as e:
                print(f"⚠️  Seri port açılamadı: {e}")
    
    def compute(self, target_center: Optional[Tuple[float, float]], 
                target_area: float = 0.0,
                target_speed: float = 0.0) -> FlightCommand:
        """
        Hedef pozisyonuna göre uçuş komutları hesapla.
        
        Args:
            target_center: (x, y) hedef merkezi (None ise hedef yok)
            target_area: Hedefin piksel alanı (mesafe tahmini için)
            target_speed: Hedef hızı (sunucudan gelirse)
        
        Returns:
            FlightCommand
        """
        cmd = FlightCommand()
        
        if target_center is None:
            # Hedef yok, tüm PID'leri sıfırla
            self.yaw_pid.reset()
            self.pitch_pid.reset()
            self.roll_pid.reset()
            self.speed_pid.reset()
            return cmd
        
        tx, ty = target_center
        
        # Hata hesaplama (normalize edilmiş: -1.0 ile 1.0 arası)
        error_x = (tx - self.screen_center_x) / self.screen_center_x
        error_y = (ty - self.screen_center_y) / self.screen_center_y
        
        # Yaw kontrolü (yatay)
        cmd.yaw_rate = self.yaw_pid.compute(error_x)
        
        # Pitch kontrolü (dikey)
        cmd.pitch_rate = self.pitch_pid.compute(-error_y)  # Eksen ters
        
        # Roll kontrolü (yatay sapma yardımcı)
        cmd.roll_rate = self.roll_pid.compute(error_x * 0.3)
        
        # Throttle/Speed kontrolü (alan bazlı mesafe)
        # Büyük alan = yakın = yavaşla, küçük alan = uzak = hızlan
        optimal_area = (self.config.camera_width * self.config.camera_height) * 0.1  # %10
        area_error = (optimal_area - target_area) / optimal_area
        cmd.throttle = self.speed_pid.compute(area_error)
        
        return cmd
    
    def send_command(self, cmd: FlightCommand):
        """Komutu seri port üzerinden gönder"""
        if self.serial_port and self.serial_port.is_open:
            # Basit protokol: "YAW,PITCH,ROLL,THR\n"
            msg = f"{cmd.yaw_rate:.2f},{cmd.pitch_rate:.2f},{cmd.roll_rate:.2f},{cmd.throttle:.2f}\n"
            self.serial_port.write(msg.encode())
    
    def close(self):
        """Kaynakları temizle"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()


# ═══════════════════════════════════════════════════════════════════════════════════
# KİLİTLENME TAKİP SİSTEMİ
# ═══════════════════════════════════════════════════════════════════════════════════

class LockTracker:
    """
    4 saniyelik kesintisiz kilitlenme takibi.
    
    Yarışma Kuralı:
    - Hedef 4 saniye boyunca kesintisiz takip edilmeli
    - Kısa kayıplar (< 0.5 sn) tolere edilebilir
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.state = LockState.SEARCHING
        
        self.locked_target_id: Optional[int] = None
        self.lock_start_time: Optional[float] = None
        self.last_seen_time: Optional[float] = None
        self.total_lock_time: float = 0.0
        
        # İstatistikler
        self.lock_count = 0
        self.lock_history: List[Dict] = []
    
    def update(self, tracked_objects: List[Dict]) -> Tuple[LockState, Optional[Dict], float]:
        """
        Kilitlenme durumunu güncelle.
        
        Args:
            tracked_objects: SORT'tan gelen track listesi
        
        Returns:
            (state, primary_target, lock_progress)
        """
        current_time = time.time()
        primary_target = None
        lock_progress = 0.0
        
        # Hedef var mı?
        if tracked_objects:
            # Kilitli hedefi bul
            if self.locked_target_id is not None:
                for obj in tracked_objects:
                    if obj['id'] == self.locked_target_id:
                        primary_target = obj
                        break
            
            # Kilitli hedef bulunamadı ama başka hedefler var
            if primary_target is None:
                # En büyük hedefi seç
                primary_target = max(
                    tracked_objects,
                    key=lambda o: (o['bbox'][2] - o['bbox'][0]) * (o['bbox'][3] - o['bbox'][1])
                )
                
                # ÖNEMLİ: Hedef ID değişse bile kilitlenmeye devam et!
                # (Tracker bazen ID yeniler ama aynı hedef)
                if self.state == LockState.SEARCHING:
                    # İlk kez hedef bulundu
                    self.locked_target_id = primary_target['id']
                    self.lock_start_time = current_time
                    self.state = LockState.TRACKING
                else:
                    # Hedef ID değişti ama kilitlenmeye devam et
                    # (aynı nesne, farklı ID olabilir)
                    self.locked_target_id = primary_target['id']
                    # lock_start_time'ı SIFIRLAMIYORUZ - devam!
        
        # Durum güncellemesi
        if primary_target:
            self.last_seen_time = current_time
            
            if self.lock_start_time is None:
                self.lock_start_time = current_time
            
            elapsed = current_time - self.lock_start_time
            lock_progress = min(1.0, elapsed / self.config.lock_duration_required)
            
            if elapsed >= self.config.lock_duration_required:
                if self.state != LockState.LOCKED:
                    self.lock_count += 1
                    self.lock_history.append({
                        'target_id': self.locked_target_id,
                        'time': current_time,
                        'duration': elapsed
                    })
                self.state = LockState.LOCKED
                self.total_lock_time = elapsed
            else:
                self.state = LockState.LOCKING
        else:
            # Hedef kayıp - tolerans süresi içinde bekle
            if self.last_seen_time:
                lost_duration = current_time - self.last_seen_time
                
                # Tolerans süresince progress'i koru
                if self.lock_start_time:
                    elapsed = self.last_seen_time - self.lock_start_time
                    lock_progress = min(1.0, elapsed / self.config.lock_duration_required)
                
                if lost_duration > self.config.lock_lost_timeout:
                    # Tolerans aşıldı - kilit bozuldu
                    self.state = LockState.SEARCHING
                    self.locked_target_id = None
                    self.lock_start_time = None
                    lock_progress = 0.0
        
        return self.state, primary_target, lock_progress
    
    def reset(self):
        """Kilidi sıfırla"""
        self.state = LockState.SEARCHING
        self.locked_target_id = None
        self.lock_start_time = None
        self.last_seen_time = None


# ═══════════════════════════════════════════════════════════════════════════════════
# ANA SİSTEM
# ═══════════════════════════════════════════════════════════════════════════════════

class IHATrackingSystem:
    """
    Tam entegre İHA Takip ve Kontrol Sistemi.
    
    Pipeline:
    1. YOLO ile tespit
    2. SORT ile takip
    3. Kilitlenme takibi (4 sn)
    4. PID ile kontrol komutları
    5. Görselleştirme
    """
    
    # Egitilmis agirlik bulunamazsa kullanilacak hazir model.
    # Ultralytics bunu ilk kullanimda kendisi indirir (~6 MB).
    FALLBACK_MODEL = "yolo11n.pt"

    def __init__(self, config: SystemConfig):
        self.config = config

        # Kullanici acikca bir model verdiyse arama yapma
        if not config.yolo_model_path:
            self._setup_yolo_path()
        
        # Bileşenleri başlat
        print("\n" + "═" * 70)
        print("   TEKNOFEST SAVAŞAN İHA - TAKİP VE KONTROL SİSTEMİ")
        print("═" * 70)
        
        print("\n📦 Model yükleniyor...")
        self.yolo = YOLO(config.yolo_model_path, task='detect')
        print(f"   ✅ YOLO: {os.path.basename(config.yolo_model_path)}")
        
        self.tracker = ByteTrackTracker(config)
        print("   ✅ ByteTrack + Kalman")
        
        self.lock_tracker = LockTracker(config)
        print("   ✅ Kilitlenme Takibi")
        
        self.flight_controller = FlightController(config)
        print("   ✅ PID Kontrolcü")
        
        # Görselleştirme
        self.colors = self._generate_colors(100)
        self.track_trails: Dict[int, deque] = {}
        
        # İstatistikler
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.session_start = time.time()
        self.current_fps = 0
        
        print("\n" + "═" * 70)
        print("   SİSTEM HAZIR")
        print("═" * 70 + "\n")
    
    def _setup_yolo_path(self):
        """YOLO model yolunu belirle"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # TensorRT kurulu mu kontrol et
        tensorrt_available = False
        try:
            import tensorrt
            tensorrt_available = True
            print("   ✓ TensorRT kurulu, .engine dosyaları kullanılabilir")
        except ImportError:
            print("   ⚠ TensorRT kurulu değil, PyTorch modeli kullanılacak")
        
        # Model öncelik sırası
        if tensorrt_available:
            candidates = [
                os.path.join(current_dir, "yolo", "best_fp16_640.engine"),
                os.path.join(current_dir, "yolo", "best.engine"),
                os.path.join(current_dir, "yolo", "best.pt"),
                os.path.join(current_dir, "yolo", "best.onnx"),
            ]
        else:
            # TensorRT yoksa .engine dosyalarını atla
            candidates = [
                os.path.join(current_dir, "yolo", "best.pt"),
                os.path.join(current_dir, "yolo", "best.onnx"),
            ]
        
        for path in candidates:
            if os.path.exists(path):
                self.config.yolo_model_path = path
                if path.endswith(".engine"):
                    model_type = "TensorRT"
                elif path.endswith(".pt"):
                    model_type = "PyTorch"
                elif path.endswith(".onnx"):
                    model_type = "ONNX"
                else:
                    model_type = "Unknown"
                print(f"   🔍 {model_type} model bulundu: {os.path.basename(path)}")
                return
        
        # Egitilmis agirliklar depoda YOK (boyut nedeniyle). Bulunamazsa
        # ultralytics'in hazir modeline dus: ilk calistirmada kendisi indirir.
        # Boylece depoyu klonlayan biri hicbir sey indirmeden sistemi
        # calisir halde gorebiliyor.
        #
        # NOT: hazir model COCO ile egitilmistir; sinif 0 = "insan".
        # Kendi agirliklarinla calistirmak icin yolo/best.pt koymak yeterli.
        self.config.yolo_model_path = self.FALLBACK_MODEL
        print("   ⚠ Eğitilmiş ağırlık bulunamadı — hazır modele düşülüyor")
        print(f"     Aranan: {os.path.join(current_dir, 'yolo')}/")
        print(f"     Kullanılan: {self.FALLBACK_MODEL} (COCO, sınıf 0 = insan)")
        print("     Kendi modelin için: yolo/best.pt")
    
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Renk paleti oluştur"""
        np.random.seed(42)
        return [(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)) for _ in range(n)]
    
    def _draw_frame(self, frame: np.ndarray, tracked_objects: List[Dict],
                    lock_state: LockState, primary_target: Optional[Dict],
                    lock_progress: float, flight_cmd: FlightCommand) -> np.ndarray:
        """Görselleştirme"""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Ekran merkezi çizgisi
        cv2.line(display, (w // 2 - 30, h // 2), (w // 2 + 30, h // 2), (255, 255, 255), 1)
        cv2.line(display, (w // 2, h // 2 - 30), (w // 2, h // 2 + 30), (255, 255, 255), 1)
        
        # Track trails
        if self.config.show_trails:
            for obj in tracked_objects:
                obj_id = obj['id']
                cx, cy = int(obj['center'][0]), int(obj['center'][1])
                
                if obj_id not in self.track_trails:
                    self.track_trails[obj_id] = deque(maxlen=self.config.trail_length)
                self.track_trails[obj_id].append((cx, cy))
                
                if len(self.track_trails[obj_id]) > 1:
                    color = self.colors[obj_id % len(self.colors)]
                    points = np.array(self.track_trails[obj_id], dtype=np.int32)
                    cv2.polylines(display, [points], False, color, 2)
        
        # Tracked objects
        for obj in tracked_objects:
            bbox = obj['bbox']
            obj_id = obj['id']
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = self.colors[obj_id % len(self.colors)]
            
            # Primary target vurgulama
            if primary_target and obj_id == primary_target['id']:
                thickness = 3
                cv2.rectangle(display, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 255), 2)
            else:
                thickness = 2
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            label = f"ID:{obj_id}"
            cv2.putText(display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Merkez noktası
            cx, cy = int(obj['center'][0]), int(obj['center'][1])
            cv2.circle(display, (cx, cy), 4, color, -1)
        
        # --- INFO PANEL ---
        panel_h = 200
        cv2.rectangle(display, (5, 5), (300, panel_h), (0, 0, 0), -1)
        cv2.rectangle(display, (5, 5), (300, panel_h), (50, 50, 50), 2)
        
        y_offset = 25
        line_height = 22
        
        # FPS
        cv2.putText(display, f"FPS: {self.current_fps}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        
        # Tracked count
        cv2.putText(display, f"Tracked: {len(tracked_objects)}/{self.config.max_objects}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += line_height
        
        # Lock state
        state_colors = {
            LockState.SEARCHING: (0, 0, 255),
            LockState.TRACKING: (0, 165, 255),
            LockState.LOCKING: (0, 255, 255),
            LockState.LOCKED: (0, 255, 0)
        }
        cv2.putText(display, f"Durum: {lock_state.value}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_colors[lock_state], 1)
        y_offset += line_height
        
        # Lock progress bar
        cv2.putText(display, "Kilit:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        bar_x, bar_y, bar_w, bar_h = 60, y_offset - 12, 150, 14
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
        fill_w = int(bar_w * lock_progress)
        bar_color = (0, 255, 0) if lock_state == LockState.LOCKED else (0, 255, 255)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.putText(display, f"{lock_progress * 100:.0f}%", (bar_x + bar_w + 5, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += line_height
        
        # PID outputs
        cv2.putText(display, f"Yaw: {flight_cmd.yaw_rate:+.1f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(display, f"Pitch: {flight_cmd.pitch_rate:+.1f}", (100, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(display, f"Roll: {flight_cmd.roll_rate:+.1f}", (200, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += line_height
        
        # Lock count
        cv2.putText(display, f"Toplam Kilit: {self.lock_tracker.lock_count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Hedef işaretçi (büyük)
        if lock_state == LockState.LOCKED and primary_target:
            cv2.putText(display, "HEDEF KİLİTLİ", (w // 2 - 80, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return display
    
    def run(self):
        """Ana döngü - Optimize edilmiş versiyon"""
        
        # Görüntü kaynağını aç. VideoSource canli kamera ile dosyayi ayirir:
        # canlida en guncel kareyi tutar, dosyada hicbir kareyi atlamaz.
        try:
            cap = VideoSource(
                self.config.camera_id,
                self.config.camera_width,
                self.config.camera_height,
            )
        except RuntimeError as e:
            print(f"❌ {e}")
            return

        if cap.is_file:
            toplam = int(cap.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            print(f"🎬 Video dosyası açıldı: {self.config.camera_id}"
                  + (f"  ({toplam} kare)" if toplam > 0 else ""))
        else:
            print("📷 Kamera açıldı (canlı, en güncel kare)")

        # İstege bagli kayit
        writer = None
        if self.config.record_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps_out = cap.cap.get(cv2.CAP_PROP_FPS) or 30.0
            if fps_out <= 1 or fps_out > 240:
                fps_out = 30.0
            writer = cv2.VideoWriter(
                self.config.record_path, fourcc, fps_out,
                (self.config.camera_width, self.config.camera_height),
            )
            print(f"💾 Kayıt: {self.config.record_path}")
        
        print(f"   • YOLO imgsz: {self.config.yolo_img_size}")
        print(f"   • Frame skip: {self.config.yolo_skip_frames}")
        print(f"   • Half precision: {self.config.yolo_half}")
        print("   • 'q' ile çıkış\n")
        
        # YOLO cihaz ve ayarlar.
        # predict() argumanlari SABIT bir sozlukte tutuluyor: her karede yeniden
        # kurulmuyor, ve 'half' YALNIZ gercekten kullanilacaksa ekleniyor.
        # Yeni ultralytics surumlerinde 'half' kullanimdan kalkti; False olsa
        # bile gecildiginde her karede uyari basiyordu.
        device = 0 if torch.cuda.is_available() else 'cpu'
        half = self.config.yolo_half and torch.cuda.is_available()

        # ByteTrack'in 2. turu DUSUK guvenli tespitlere ihtiyac duyar; o yuzden
        # YOLO'yu dusuk esikle calistiriyoruz. Asil eleme takipcide yapiliyor:
        #   >= track_high_thresh -> yeni iz baslatabilir
        #   >= track_low_thresh  -> yalniz mevcut izi ayakta tutar
        # Bu esik YOLO'ya verilseydi kapanma anindaki tespit hic gelmezdi.
        yolo_conf = min(self.config.yolo_conf_threshold,
                        self.config.track_low_thresh)

        predict_kwargs = dict(
            imgsz=self.config.yolo_img_size,
            conf=yolo_conf,
            device=device,
            verbose=False,
            classes=list(self.config.yolo_classes),
        )
        if half:
            predict_kwargs["half"] = True
        
        # Son tespitleri sakla
        last_detections = np.empty((0, 5))
        detection_count = 0  # Debug için
        
        try:
            bos_okuma = 0
            while True:
                ret, frame = cap.read()

                if not ret or frame is None:
                    # Akis kalici olarak bittiyse cik. Eski surumde burada
                    # 'continue' vardi: video bitince ya da kamera cekilince
                    # program %100 CPU'da sonsuza kadar donuyordu.
                    if cap.is_finished():
                        print("\n🏁 Görüntü akışı bitti.")
                        break
                    # Canli kamerada ilk kareler gelene kadar kisa bekleme
                    bos_okuma += 1
                    if bos_okuma > 300:
                        print("\n⚠ Görüntü alınamıyor, çıkılıyor.")
                        break
                    time.sleep(0.005)
                    continue

                bos_okuma = 0
                self.frame_count += 1
                
                # 1. YOLO Tespit (sadece belirli frame'lerde)
                run_yolo = (self.frame_count % self.config.yolo_skip_frames == 0)
                
                if run_yolo:
                    results = self.yolo.predict(frame, **predict_kwargs)
                    
                    # Tespitleri hazırla [x1, y1, x2, y2, score]
                    detections = []
                    if len(results[0].boxes) > 0:
                        for det in results[0].boxes:
                            bbox = det.xyxy[0].cpu().numpy()
                            score = det.conf[0].item()
                            
                            # Geometrik Filtreleme
                            w = bbox[2] - bbox[0]
                            h = bbox[3] - bbox[1]
                            area = w * h
                            if h > 0:
                                aspect_ratio = w / h
                            else:
                                aspect_ratio = 0
                            
                            # 1. Alan kontrolü (çok küçük veya devasa)
                            if area < self.config.min_box_area:
                                continue
                            
                            image_area = self.config.camera_width * self.config.camera_height
                            if area > (image_area * self.config.max_box_area_percent):
                                continue
                                
                            # 2. Aspect Ratio kontrolü (şeritmsi hatalar)
                            if aspect_ratio < self.config.min_aspect_ratio or aspect_ratio > self.config.max_aspect_ratio:
                                continue
                            
                            detections.append([*bbox, score])
                    
                    if detections:
                        detections = np.array(detections)
                        last_detections = detections.copy()
                        detection_count = len(detections)
                    else:
                        detections = np.empty((0, 5))
                        detection_count = 0
                else:
                    # YOLO çalışmadığında boş gönder - Kalman tahmin devam eder
                    detections = np.empty((0, 5))
                
                # 2. SORT Takip (her frame'de çalışır - Kalman tahmin yapar).
                # run_yolo bilgisi tracker'a GECILIYOR: atlanan karelerde izler
                # hareket tahmini alir ama kayip sayilmaz.
                tracked_objects = self.tracker.update(detections, tespit_karesi=run_yolo)
                
                # 3. Kilitlenme Takibi
                lock_state, primary_target, lock_progress = self.lock_tracker.update(tracked_objects)
                
                # 4. PID Kontrol
                target_center = primary_target['center'] if primary_target else None
                target_area = 0.0
                if primary_target:
                    bbox = primary_target['bbox']
                    target_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                
                flight_cmd = self.flight_controller.compute(target_center, target_area)
                
                # Komutu gönder (seri port aktifse)
                if self.config.serial_enabled:
                    self.flight_controller.send_command(flight_cmd)
                
                # 5. Görselleştirme
                # Cizim, PENCERE ACIK OLMASA DA gerekebilir: --no-display ile
                # --record birlikte kullanildiginda kayda islenmis goruntu
                # yazilmali. Bu yuzden kosul "ikisinden biri".
                ciz = self.config.show_display or writer is not None

                if ciz:
                    # Ham tespitleri çiz (yeşil - debug için)
                    if len(last_detections) > 0:
                        for det in last_detections:
                            x1, y1, x2, y2 = [int(v) for v in det[:4]]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                            cv2.putText(frame, f"{det[4]:.2f}", (x1, y1-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                    display = self._draw_frame(
                        frame, tracked_objects, lock_state,
                        primary_target, lock_progress, flight_cmd
                    )

                    # Debug info
                    cv2.putText(display, f"Det: {detection_count}", (220, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Kilit debug - süre göster
                    if self.lock_tracker.lock_start_time:
                        elapsed = time.time() - self.lock_tracker.lock_start_time
                        cv2.putText(display, f"Sure: {elapsed:.1f}s", (10, 160),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        cv2.putText(display, f"ID: {self.lock_tracker.locked_target_id}", (120, 160),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    if writer is not None:
                        h, w = display.shape[:2]
                        if (w, h) != (self.config.camera_width, self.config.camera_height):
                            display_kayit = cv2.resize(
                                display,
                                (self.config.camera_width, self.config.camera_height))
                        else:
                            display_kayit = display
                        writer.write(display_kayit)

                if self.config.show_display:
                    cv2.imshow("IHA Tracking System", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # FPS sayaci GORSELLESTIRMEDEN BAGIMSIZ.
                # Eski surumde bu blok imshow ile ayni if icindeydi; --no-display
                # ile calistirildiginda FPS hic guncellenmiyordu.
                self.fps_counter += 1
                if time.time() - self.fps_start_time > 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_start_time = time.time()

        except KeyboardInterrupt:
            print("\n👋 Kullanıcı çıkışı")
        except Exception as e:
            print(f"\n❌ Hata: {e}")
            traceback.print_exc()
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            self.flight_controller.close()
            cv2.destroyAllWindows()

            # Istatistikler. "Ortalama" gercekten ortalama: toplam kare /
            # toplam sure. Eski surumde son saniyenin sayisi yazdiriliyordu.
            gecen = max(time.time() - self.session_start, 1e-6)
            ortalama = self.frame_count / gecen

            print("\n" + "═" * 50)
            print("📊 OTURUM İSTATİSTİKLERİ")
            print("═" * 50)
            print(f"   Toplam frame      : {self.frame_count}")
            print(f"   Süre              : {gecen:.1f} sn")
            print(f"   Ortalama FPS      : {ortalama:.1f}")
            print(f"   Toplam kilit      : {self.lock_tracker.lock_count}")
            print("═" * 50 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════════
# GİRİŞ NOKTASI
# ═══════════════════════════════════════════════════════════════════════════════════

def parse_args():
    """Komut satiri arayuzu.

    Onceki surumde butun ayarlar __main__ icine gomuluydu; kamerayi degistirmek
    ya da bir video dosyasi denemek icin kaynak kodu duzenlemek gerekiyordu.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="iha_tracking_system.py",
        description="TEKNOFEST Savasan IHA - takip ve kontrol sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ornekler:
  python iha_tracking_system.py                      # varsayilan kamera
  python iha_tracking_system.py --source 1           # ikinci kamera
  python iha_tracking_system.py --source ucus.mp4    # video dosyasi
  python iha_tracking_system.py --source ucus.mp4 --record cikti.mp4
  python iha_tracking_system.py --no-display         # basliksiz (sunucu/SSH)
  python iha_tracking_system.py --model yolo/best.pt --classes 0 1
""",
    )

    g = p.add_argument_group("goruntu kaynagi")
    g.add_argument("--source", default="0",
                   help="kamera indeksi (0, 1, ...) veya video dosyasi yolu")
    g.add_argument("--width", type=int, default=640, help="kamera genisligi")
    g.add_argument("--height", type=int, default=480, help="kamera yuksekligi")

    g = p.add_argument_group("model")
    g.add_argument("--model", default="",
                   help="model yolu. bos birakilirsa yolo/ altinda aranir, "
                        "bulunamazsa hazir modele dusulur")
    g.add_argument("--imgsz", type=int, default=320,
                   help="YOLO giris boyutu (dusuk = hizli)")
    g.add_argument("--conf", type=float, default=0.55,
                   help="yuksek guven esigi — bunun ustundeki tespitler yeni iz "
                        "baslatir (ByteTrack 1. tur)")
    g.add_argument("--conf-low", type=float, default=0.1,
                   help="dusuk guven esigi — bu araliktakiler yeni iz baslatmaz, "
                        "yalniz mevcut izleri ayakta tutar (ByteTrack 2. tur)")
    g.add_argument("--skip", type=int, default=2,
                   help="her N karede bir tespit calistir (aradakiler Kalman)")
    g.add_argument("--classes", type=int, nargs="+", default=[0],
                   help="takip edilecek sinif kimlikleri")
    g.add_argument("--half", action="store_true",
                   help="FP16 kullan (yalniz CUDA'da anlamli)")

    g = p.add_argument_group("takip ve kilit")
    g.add_argument("--max-objects", type=int, default=10)
    g.add_argument("--lock-seconds", type=float, default=4.0,
                   help="kesintisiz kilit suresi (yarisma kurali)")

    g = p.add_argument_group("cikis")
    g.add_argument("--no-display", action="store_true",
                   help="pencere acma (baslikli ortam yoksa)")
    g.add_argument("--record", default="",
                   help="islenmis goruntuyu bu dosyaya yaz")
    g.add_argument("--serial", action="store_true",
                   help="ucus kontrolcusune seri porttan komut gonder")
    g.add_argument("--serial-port", default="/dev/ttyTHS1")

    return p.parse_args()


def main():
    a = parse_args()

    config = SystemConfig(
        # Model
        yolo_model_path=a.model,
        yolo_img_size=a.imgsz,
        yolo_conf_threshold=a.conf,
        track_high_thresh=a.conf,
        track_low_thresh=a.conf_low,
        yolo_skip_frames=max(1, a.skip),
        yolo_half=a.half and torch.cuda.is_available(),
        yolo_classes=tuple(a.classes),

        # Takip
        max_objects=a.max_objects,
        lock_duration_required=a.lock_seconds,

        # Kaynak
        camera_id=a.source,
        camera_width=a.width,
        camera_height=a.height,

        # Cikis
        show_display=not a.no_display,
        record_path=a.record,
        serial_enabled=a.serial,
        serial_port=a.serial_port,
    )

    cihaz = "CUDA" if torch.cuda.is_available() else "CPU"
    print("\n" + "=" * 62)
    print("   TEKNOFEST SAVASAN IHA - TAKIP VE KONTROL SISTEMI")
    print("=" * 62)
    print(f"   Kaynak     : {a.source}")
    print(f"   Cihaz      : {cihaz}")
    print(f"   YOLO boyutu: {a.imgsz}px   guven: {a.conf}   kare atlama: {a.skip}")
    print(f"   Goruntu    : {'kapali' if a.no_display else 'acik'}")
    print("=" * 62)

    try:
        IHATrackingSystem(config).run()
    except KeyboardInterrupt:
        print("\nKullanici cikisi")
        return 130
    except Exception as e:
        print(f"\nHATA: {e}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
