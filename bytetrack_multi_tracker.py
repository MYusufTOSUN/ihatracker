"""
Optimize edilmiş çoklu nesne tracking sistemi.
ByteTrack + Kalman Filter tabanlı, tamamen YOLO tespitlerine dayalı çalışır.

Özellikler:
- Kalman filter ile hareket tahmini
- ByteTrack tarzı iki aşamalı matching algoritması
- ID persistence için iz takibi ve istatistikler
- Optimize edilmiş performans
"""

import cv2
import torch
from ultralytics import YOLO 
import time
import numpy as np
import os
import sys
import yaml
import traceback
from collections import OrderedDict, deque
from scipy.optimize import linear_sum_assignment


current_dir = os.path.dirname(os.path.abspath(__file__))


class KalmanBoxTracker:
    """
    Kalman Filter ile bbox tracking
    Hareket tahminleri yapar, ID persistence'ı güçlendirir
    """
    count = 0
    
    def __init__(self, bbox):
        """bbox: [x1, y1, x2, y2]"""
        from filterpy.kalman import KalmanFilter
        
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # State: [x, y, s, r, vx, vy, vs]
        # x, y: merkez, s: alan, r: aspect ratio
        # vx, vy, vs: hızlar
        
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def _convert_bbox_to_z(self, bbox):
        """[x1,y1,x2,y2] -> [x,y,s,r]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h) if h != 0 else 1
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _convert_x_to_bbox(self, x):
        """[x,y,s,r] -> [x1,y1,x2,y2]"""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w if w != 0 else 1
        return np.array([
            x[0]-w/2., x[1]-h/2.,
            x[0]+w/2., x[1]-h/2.
        ]).reshape((1,4))
    
    def update(self, bbox):
        """Yeni detection ile güncelle"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
    
    def predict(self):
        """Bir sonraki pozisyonu tahmin et"""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1][0]
    
    def get_state(self):
        """Mevcut bbox'ı al"""
        return self._convert_x_to_bbox(self.kf.x)[0]


class ByteTrackMultiTracker:
    """
    ByteTrack algoritması ile optimize edilmiş çoklu nesne tracking.
    Tamamen YOLO tespitleri + Kalman Filter ile çalışır, Siam tabanlı hiçbir bileşen kullanmaz.
    """
    
    def __init__(self, device, max_objects=10):
        self.max_objects = max_objects
        self.device = device
        self.frame_id = 0
        
        # Tracked objects
        self.tracked_objects = []
        self.lost_stracks = []
        self.removed_stracks = []
        
        # Parametreler (optimize edilmiş)
        self.track_high_thresh = 0.6
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.7
        self.match_thresh = 0.8
        self.max_time_lost = 30
        
        # Görselleştirme
        self.colors = self._generate_colors(100)
        
        # Track trail (iz takibi)
        self.track_trails = {}  # {obj_id: deque([(x,y), ...])}
        self.max_trail_length = 30
        
        # Event logging
        self.events = deque(maxlen=5)  # Son 5 event
        
        # İstatistikler
        self.stats = {
            'total_tracks_created': 0,
            'total_tracks_lost': 0,
            'total_recoveries': 0
        }
        
        print("✅ ByteTrack Multi-Tracker hazır (Optimize + Enhanced)")
    
    def _generate_colors(self, n):
        """Renk paleti oluştur"""
        np.random.seed(42)
        colors = []
        for i in range(n):
            colors.append((
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            ))
        return colors
    
    def _iou_batch(self, bboxes1, bboxes2):
        """Batch IoU hesaplama (optimize edilmiş)"""
        if len(bboxes1) == 0 or len(bboxes2) == 0:
            return np.zeros((len(bboxes1), len(bboxes2)))
        
        bboxes1 = np.asarray(bboxes1)
        bboxes2 = np.asarray(bboxes2)
        
        xx1 = np.maximum(bboxes1[:, None, 0], bboxes2[:, 0])
        yy1 = np.maximum(bboxes1[:, None, 1], bboxes2[:, 1])
        xx2 = np.minimum(bboxes1[:, None, 2], bboxes2[:, 2])
        yy2 = np.minimum(bboxes1[:, None, 3], bboxes2[:, 3])
        
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        
        intersection = w * h
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        union = area1[:, None] + area2 - intersection
        
        return intersection / np.maximum(union, 1e-6)
    
    def _linear_assignment(self, cost_matrix, thresh):
        """Hungarian algorithm ile optimal matching"""
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        
        matches, unmatched_a, unmatched_b = [], [], []
        cost_matrix = np.where(cost_matrix > thresh, 0, cost_matrix)
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for i in range(cost_matrix.shape[0]):
            if i not in row_ind:
                unmatched_a.append(i)
        
        for j in range(cost_matrix.shape[1]):
            if j not in col_ind:
                unmatched_b.append(j)
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < thresh:
                matches.append([i, j])
            else:
                unmatched_a.append(i)
                unmatched_b.append(j)
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.array(matches)
        
        return matches, tuple(unmatched_a), tuple(unmatched_b)
    
    def update(self, frame, detections):
        """
        ByteTrack algoritması ile güncelle
        detections: list of {'bbox': [x1,y1,x2,y2], 'score': float}
        """
        self.frame_id += 1
        
        # 1. Kalman prediction
        for track in self.tracked_objects:
            track.predict()
        
        # 2. Detections'ları skorlarına göre ayır
        if detections:
            scores = np.array([d['score'] for d in detections])
            bboxes = np.array([d['bbox'] for d in detections])
            
            remain_inds = scores > self.track_low_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            
            inds_high = scores_keep > self.track_high_thresh
            inds_low = scores_keep <= self.track_high_thresh
            
            dets_high = dets[inds_high]
            dets_low = dets[inds_low]
            scores_high = scores_keep[inds_high]
            scores_low = scores_keep[inds_low]
        else:
            dets_high = np.empty((0, 4))
            dets_low = np.empty((0, 4))
            scores_high = np.empty((0,))
            scores_low = np.empty((0,))
        
        # 3. İlk matching: high confidence detections
        if len(dets_high) > 0 and len(self.tracked_objects) > 0:
            track_bboxes = np.array([t.get_state() for t in self.tracked_objects])
            iou_matrix = self._iou_batch(track_bboxes, dets_high)
            cost_matrix = 1 - iou_matrix
            
            matches, unmatched_tracks, unmatched_dets = self._linear_assignment(
                cost_matrix, self.match_thresh
            )
            
            # Matched tracks'i güncelle
            for itracked, idet in matches:
                self.tracked_objects[itracked].update(dets_high[idet])
            
            # Unmatched tracks
            unmatched_tracks = list(unmatched_tracks)
        else:
            unmatched_tracks = list(range(len(self.tracked_objects)))
            unmatched_dets = list(range(len(dets_high)))
        
        # 4. İkinci matching: low confidence detections ile unmatched tracks
        if len(dets_low) > 0 and len(unmatched_tracks) > 0:
            track_bboxes = np.array([self.tracked_objects[i].get_state() for i in unmatched_tracks])
            iou_matrix = self._iou_batch(track_bboxes, dets_low)
            cost_matrix = 1 - iou_matrix
            
            matches_low, unmatched_tracks_low, _ = self._linear_assignment(
                cost_matrix, 0.5
            )
            
            for itracked, idet in matches_low:
                track_idx = unmatched_tracks[itracked]
                self.tracked_objects[track_idx].update(dets_low[idet])
            
            unmatched_tracks = [unmatched_tracks[i] for i in unmatched_tracks_low]
        
        # 5. Yeni tracks oluştur (high confidence detections)
        for idet in unmatched_dets:
            if scores_high[idet] > self.new_track_thresh:
                if len(self.tracked_objects) < self.max_objects:
                    new_track = KalmanBoxTracker(dets_high[idet])
                    self.tracked_objects.append(new_track)
                    self.stats['total_tracks_created'] += 1
                    self.events.append(f"✨ ID:{new_track.id} oluşturuldu")
                    print(f"✨ Yeni nesne: ID #{new_track.id} (skor: {scores_high[idet]:.2f})")
        
        # 6. Kayıp tracks'i kaldır
        i = len(self.tracked_objects)
        for track in reversed(self.tracked_objects):
            i -= 1
            if track.time_since_update > self.max_time_lost:
                removed_track = self.tracked_objects.pop(i)
                self.removed_stracks.append(removed_track)
                self.stats['total_tracks_lost'] += 1
                self.events.append(f"🗑️ ID:{removed_track.id} kayıp")
                print(f"🗑️ Nesne kayıp: ID #{removed_track.id}")
                # Trail'i de sil
                if removed_track.id in self.track_trails:
                    del self.track_trails[removed_track.id]
        
        # 7. Track trails güncelle
        for track in self.tracked_objects:
            if track.time_since_update == 0:  # Güncellendi
                bbox = track.get_state()
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                
                if track.id not in self.track_trails:
                    self.track_trails[track.id] = deque(maxlen=self.max_trail_length)
                
                self.track_trails[track.id].append((cx, cy))
    
    def get_tracked_objects(self):
        """Aktif tracked objects'i al"""
        results = []
        for track in self.tracked_objects:
            if track.time_since_update < 1:
                bbox = track.get_state()
                results.append({
                    'id': track.id,
                    'bbox': bbox,
                    'score': 1.0 if track.hit_streak > 0 else 0.5
                })
        return results
    
    def draw(self, frame, show_raw_detections=False, raw_detections=None, show_trails=True):
        """
        Tracked objects'i çiz (gelişmiş görselleştirme)
        
        Args:
            frame: görüntü
            show_raw_detections: YOLO ham detection'larını göster
            raw_detections: YOLO'dan gelen ham detections
            show_trails: Track izlerini göster
        """
        # 1. Ham YOLO detections'larını çiz (GRİ ÇERÇEVE)
        if show_raw_detections and raw_detections:
            for det in raw_detections:
                bbox = det['bbox']
                score = det['score']
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                # Gri kesikli çerçeve - Ham detection
                self._draw_dashed_rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                
                # "DET" etiketi (küçük, gri)
                label = f"DET:{score:.2f}"
                cv2.putText(frame, label, (x1, y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        # 2. Track trails çiz (iz çizgileri)
        if show_trails:
            for obj_id, trail in self.track_trails.items():
                if len(trail) > 1:
                    color = self.colors[obj_id % len(self.colors)]
                    points = np.array(trail, dtype=np.int32)
                    
                    # İz çizgisi (fade effect)
                    for i in range(1, len(points)):
                        alpha = i / len(points)  # 0 -> 1 (fade)
                        thickness = max(1, int(2 * alpha))
                        cv2.line(frame, tuple(points[i-1]), tuple(points[i]), 
                                color, thickness)
        
        # 3. Tracked objects'i çiz (KALIN RENKLİ ÇERÇEVE)
        for obj in self.get_tracked_objects():
            bbox = obj['bbox']
            obj_id = obj['id']
            score = obj['score']
            
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # ID'ye göre ana renk
            color = self.colors[obj_id % len(self.colors)]
            
            # Confidence'a göre renk modifikasyonu
            if score > 0.8:
                border_color = color  # Yüksek güven - parlak
                thickness = 3
            elif score > 0.5:
                border_color = tuple(int(c * 0.8) for c in color)  # Orta - biraz soluk
                thickness = 2
            else:
                border_color = tuple(int(c * 0.6) for c in color)  # Düşük - daha soluk
                thickness = 2
            
            # Bbox (kalın çerçeve)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, thickness)
            
            # Köşe işaretleri (fancy)
            corner_len = 15
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)
            
            # Label arka planı (gradient effect)
            label = f"ID:{obj_id} ({score:.2f})"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Arka plan (shadow effect)
            cv2.rectangle(frame, (x1+2, y1 - label_h - 8), (x1 + label_w + 7, y1+2), (0,0,0), -1)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)
            
            # Label text (beyaz, kalın)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Merkez nokta (büyük daire)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)
        
        return frame
    
    def _draw_dashed_rectangle(self, frame, pt1, pt2, color, thickness):
        """Kesikli dikdörtgen çiz"""
        x1, y1 = pt1
        x2, y2 = pt2
        dash_length = 10
        
        # Üst kenar
        for x in range(x1, x2, dash_length * 2):
            cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        # Alt kenar
        for x in range(x1, x2, dash_length * 2):
            cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        # Sol kenar
        for y in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        # Sağ kenar
        for y in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)


# --- MAIN ---

# ═══════════════════════════════════════════════════════════════════
# GIRIS NOKTASI
# ═══════════════════════════════════════════════════════════════════

def main():
    """Uygulamayi calistir.

    NEDEN FONKSIYON ICINDE: onceden bu blok modul seviyesindeydi, yani
    dosyayi import etmek KAMERAYI ACIP sistemi baslatiyordu. Icindeki
    siniflari baska bir yerde yeniden kullanmak imkansizdi.
    """
    YOLO_ENGINE_PATH = os.path.join(current_dir, "yolo", "best.engine")
    YOLO_PT_PATH = os.path.join(current_dir, "yolo", "best.pt")
 
    # ÖNCE best.engine'i kontrol et (TensorRT öncelikli!)
    if os.path.exists(YOLO_ENGINE_PATH):
        YOLO_MODEL_PATH = YOLO_ENGINE_PATH
        YOLO_MODEL_TYPE = "TensorRT Engine"
        print("✅ best.engine bulundu - TensorRT kullanılacak")
    elif os.path.exists(YOLO_PT_PATH):
        YOLO_MODEL_PATH = YOLO_PT_PATH
        YOLO_MODEL_TYPE = "PyTorch"
        print("⚠️ best.engine YOK - best.pt kullanılacak (daha yavaş)")
    else:
        print("❌ HATA: YOLO model dosyası bulunamadı!")
        print(f"   Aranan: {YOLO_ENGINE_PATH}")
        print(f"   Veya: {YOLO_PT_PATH}")
        sys.exit(1)

    YOLO_IMG_SIZE = 320  # TensorRT engine boyutu (sabit, değiştirilemez!)
    YOLO_CONF_THRESHOLD = 0.5  # Düşük threshold (ByteTrack için)
    MAX_OBJECTS = 20

    print(f"📦 YOLO Model: {YOLO_MODEL_TYPE}")
    print(f"📂 Dosya: {os.path.basename(YOLO_MODEL_PATH)}")

    # Model yükleme
    try:
        print("\nModel yükleme başlıyor...")
    
        # YOLO - TensorRT Engine garantisi
        if YOLO_MODEL_TYPE == "TensorRT Engine":
            yolo_model = YOLO(YOLO_MODEL_PATH, task='detect')
            print("✓ YOLO yüklendi (TensorRT Engine - MAKSIMUM HIZ)")
        else:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print("✓ YOLO yüklendi (PyTorch - standart hız)")
            print("  ℹ️  Daha hızlı için best.engine kullanın")
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # ByteTrack Multi-Tracker (yalnızca YOLO + Kalman)
        tracker = ByteTrackMultiTracker(
            device=device,
            max_objects=MAX_OBJECTS
        )
    
        print("✅ ByteTrack Sistemi Hazır\n")

    except Exception as e:
        print(f"❌ Hata: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera açılamadı")
        sys.exit(1)

    print("📷 Kamera açıldı - ByteTrack Multi-Object Tracking")
    print("   • Kalman filter ile hareket tahmini")
    print("   • Robust ID persistence")
    print("   • Optimize edilmiş performans")
    print("   • 'q' ile çıkış\n")

    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    total_detections = 0
    total_tracks = 0

    # Görselleştirme ayarları
    SHOW_RAW_DETECTIONS = True  # YOLO ham detection'larını göster

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
        
            # YOLO detection (TensorRT Engine garantisi ile)
            if YOLO_MODEL_TYPE == "TensorRT Engine":
                results = yolo_model.predict(
                    frame, 
                    imgsz=YOLO_IMG_SIZE,  # TensorRT engine boyutu
                    conf=YOLO_CONF_THRESHOLD,
                    device=0 if torch.cuda.is_available() else 'cpu',      # GPU zorunlu (TensorRT)
                    verbose=False, 
                    half=True      # FP16 (TensorRT optimize)
                )
            else:
                results = yolo_model.predict(
                    frame, 
                    imgsz=YOLO_IMG_SIZE, 
                    conf=YOLO_CONF_THRESHOLD,
                    device=0 if torch.cuda.is_available() else 'cpu',
                    verbose=False
                )
        
            # Detections'ları hazırla
            detections = []
            if len(results[0].boxes) > 0:
                for det in results[0].boxes:
                    bbox_xyxy = det.xyxy[0].cpu().numpy()
                    conf = det.conf[0].item()
                    detections.append({
                        'bbox': bbox_xyxy,
                        'score': conf
                    })
                total_detections += len(detections)
        
            # ByteTrack güncelle
            tracker.update(frame, detections)
            total_tracks = max(total_tracks, len(tracker.get_tracked_objects()))
        
            # Çiz (ham detections ile birlikte)
            display_frame = tracker.draw(
                frame.copy(), 
                show_raw_detections=SHOW_RAW_DETECTIONS,
                raw_detections=detections
            )
        
            # FPS
            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
        
            # Gelişmiş Info panel
            tracked_count = len(tracker.get_tracked_objects())
            det_count = len(detections)
        
            # Panel arka planı (biraz daha büyük)
            cv2.rectangle(display_frame, (5, 5), (320, 130), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (5, 5), (320, 130), (50, 50, 50), 2)
        
            # FPS (yeşil)
            cv2.putText(display_frame, f"FPS: {current_fps}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
            # Tracked count (sarı)
            cv2.putText(display_frame, f"Tracked: {tracked_count}/{MAX_OBJECTS}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
            # Detection count (mavi)
            cv2.putText(display_frame, f"Detections: {det_count}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)
        
            # Model tipi
            model_text = "TensorRT" if YOLO_MODEL_TYPE == "TensorRT Engine" else "PyTorch"
            model_color = (0, 255, 0) if YOLO_MODEL_TYPE == "TensorRT Engine" else (0, 165, 255)
            cv2.putText(display_frame, f"Engine: {model_text}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, model_color, 1)
        
            # Algoritma
            cv2.putText(display_frame, "ByteTrack + Kalman", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
            cv2.imshow("ByteTrack Multi-Object Tracker", display_frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n👋 Çıkış yapıldı")
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        traceback.print_exc()
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("✅ Temizlendi")



if __name__ == "__main__":
    main()
