"""
JETSON NANO OPTIMIZE EDİLMİŞ ÇOKLU NESNE TRACKING SİSTEMİ
ByteTrack + Selective SiamRPN++ + Kalman Filter

TEKNOFEST SAVŞAN İHA İÇİN OPTİMİZE EDİLDİ

Özellikler:
- Jetson Nano 4GB için optimize edilmiş
- Selective SiamRPN++ (sadece gerektiğinde)
- ByteTrack + Kalman Filter (her zaman)
- TensorRT YOLO (best.engine)
- 256x256 image size (hız optimizasyonu)
- Memory efficient
- 15-22 FPS hedef (5 nesne)

Performans Hedefleri:
- FPS: 15-22 (5 nesne)
- Latency: <65ms
- Memory: <3GB
- ID Persistence: Mükemmel
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
    Jetson Nano için optimize edilmiş
    """
    count = 0
    
    def __init__(self, bbox):
        """bbox: [x1, y1, x2, y2]"""
        from filterpy.kalman import KalmanFilter
        
        # 7 state: [x, y, s, r, vx, vy, vs]
        # x,y: center, s: area, r: aspect ratio
        # vx,vy,vs: velocities
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        
        # Measurement noise (Jetson için optimize)
        self.kf.R[2:, 2:] *= 10.0
        
        # Process noise (daha agresif prediction)
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
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
        r = w / float(h) if h != 0 else 1.0
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _convert_x_to_bbox(self, x):
        """[x,y,s,r] -> [x1,y1,x2,y2]"""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w if w != 0 else 1.0
        return np.array([
            x[0] - w/2.,
            x[1] - h/2.,
            x[0] + w/2.,
            x[1] + h/2.
        ]).flatten()
    
    def update(self, bbox):
        """Kalman güncelle"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
    
    def predict(self):
        """Kalman prediction"""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]
    
    def get_state(self):
        """Mevcut bbox [x1,y1,x2,y2]"""
        return self._convert_x_to_bbox(self.kf.x)


class JetsonOptimizedTracker:
    """
    Jetson Nano için optimize edilmiş çoklu nesne takip sınıfı.
    ByteTrack + Kalman Filter tabanlıdır ve yalnızca YOLO tespitlerini kullanır.
    
    Optimizasyonlar:
    - ByteTrack + Kalman her zaman aktif
    - Memory efficient
    - Vectorized operations
    """
    
    def __init__(self, device, max_objects=10):
        self.max_objects = max_objects
        self.device = device
        self.frame_id = 0
        
        # Tracked objects
        self.tracked_objects = []
        self.lost_stracks = []
        self.removed_stracks = []
        
        # ByteTrack parametreleri (Jetson için optimize)
        self.track_high_thresh = 0.6  # Yüksek confidence threshold
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.7
        self.match_thresh = 0.8
        self.max_time_lost = 30
        
        # Görselleştirme
        self.colors = self._generate_colors(100)
        
        # Track trails
        self.track_trails = {}
        self.max_trail_length = 20  # Jetson için kısaltıldı
        
        # Events
        self.events = deque(maxlen=5)
        
        # İstatistikler
        self.stats = {
            'total_tracks_created': 0,
            'total_tracks_lost': 0,
            'bytetrack_only': 0
        }
        
        print(f"✅ Jetson Optimized Tracker hazır")
        print(f"   - Max objects: {max_objects}")
        print(f"   - Algoritma: ByteTrack + Kalman (SiamRPN++ YOK)")
    
    def _generate_colors(self, n):
        """Renk paleti oluştur"""
        colors = []
        np.random.seed(42)
        for _ in range(n):
            colors.append(tuple(map(int, np.random.randint(50, 255, 3))))
        return colors
    
    def _iou_batch(self, bboxes1, bboxes2):
        """Vectorized IoU hesaplama (hızlı)"""
        bboxes1 = np.asarray(bboxes1)
        bboxes2 = np.asarray(bboxes2)
        
        if len(bboxes1) == 0 or len(bboxes2) == 0:
            return np.zeros((len(bboxes1), len(bboxes2)))
        
        # Expand dimensions
        bboxes1 = np.expand_dims(bboxes1, 1)  # (N, 1, 4)
        bboxes2 = np.expand_dims(bboxes2, 0)  # (1, M, 4)
        
        # Intersection
        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        intersection = w * h
        
        # Union
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        union = area1 + area2 - intersection
        
        iou = intersection / np.maximum(union, 1e-6)
        return iou
    
    def _linear_assignment(self, cost_matrix, thresh):
        """Hungarian algorithm ile optimal matching"""
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        
        matches, unmatched_a, unmatched_b = [], [], []
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Threshold kontrolü
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > thresh:
                unmatched_a.append(r)
                unmatched_b.append(c)
            else:
                matches.append([r, c])
        
        # Unmatched
        unmatched_a += list(set(range(cost_matrix.shape[0])) - set(row_ind))
        unmatched_b += list(set(range(cost_matrix.shape[1])) - set(col_ind))
        
        return np.array(matches), np.array(unmatched_a), np.array(unmatched_b)
    
    def update(self, frame, detections):
        """
        Tracking güncelle (Jetson optimize)
        
        Args:
            frame: görüntü
            detections: [{'bbox': [x1,y1,x2,y2], 'score': float}, ...]
        """
        self.frame_id += 1
        
        # 1. Kalman prediction (tüm track'ler)
        for track in self.tracked_objects:
            track.predict()
        
        # 2. Detections'ları ayır (high/low confidence)
        dets_high = []
        scores_high = []
        dets_low = []
        scores_low = []
        
        for det in detections:
            if det['score'] > self.track_high_thresh:
                dets_high.append(det['bbox'])
                scores_high.append(det['score'])
            elif det['score'] > self.track_low_thresh:
                dets_low.append(det['bbox'])
                scores_low.append(det['score'])
        
        dets_high = np.array(dets_high) if len(dets_high) > 0 else np.empty((0, 4))
        scores_high = np.array(scores_high) if len(scores_high) > 0 else np.empty((0,))
        dets_low = np.array(dets_low) if len(dets_low) > 0 else np.empty((0, 4))
        scores_low = np.array(scores_low) if len(scores_low) > 0 else np.empty((0,))
        
        # 3. İlk matching: high confidence detections
        unmatched_tracks = []
        unmatched_dets = []
        
        if len(dets_high) > 0 and len(self.tracked_objects) > 0:
            track_bboxes = np.array([t.get_state() for t in self.tracked_objects])
            iou_matrix = self._iou_batch(track_bboxes, dets_high)
            cost_matrix = 1 - iou_matrix
            
            matches, unmatched_tracks, unmatched_dets = self._linear_assignment(
                cost_matrix, self.match_thresh
            )
            
            # Matched tracks'i (sadece ByteTrack + Kalman ile) güncelle
            for itracked, idet in matches:
                track = self.tracked_objects[itracked]
                self.stats['bytetrack_only'] += 1
                track.update(dets_high[idet])
            
            unmatched_tracks = list(unmatched_tracks)
        else:
            unmatched_tracks = list(range(len(self.tracked_objects)))
            unmatched_dets = list(range(len(dets_high)))
        
        # 4. İkinci matching: low confidence detections
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
        
        # 5. Yeni tracks oluştur
        for idet in unmatched_dets:
            if scores_high[idet] > self.new_track_thresh:
                if len(self.tracked_objects) < self.max_objects:
                    new_track = KalmanBoxTracker(dets_high[idet])
                    self.tracked_objects.append(new_track)
                    self.stats['total_tracks_created'] += 1
                    self.events.append(f"✨ ID:{new_track.id}")
        
        # 6. Kayıp tracks'i kaldır
        i = len(self.tracked_objects)
        for track in reversed(self.tracked_objects):
            i -= 1
            if track.time_since_update > self.max_time_lost:
                removed_track = self.tracked_objects.pop(i)
                self.removed_stracks.append(removed_track)
                self.stats['total_tracks_lost'] += 1
                self.events.append(f"🗑️ ID:{removed_track.id}")
                if removed_track.id in self.track_trails:
                    del self.track_trails[removed_track.id]
        
        # 7. Track trails güncelle
        for track in self.tracked_objects:
            if track.time_since_update == 0:
                bbox = track.get_state()
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                
                if track.id not in self.track_trails:
                    self.track_trails[track.id] = deque(maxlen=self.max_trail_length)
                
                self.track_trails[track.id].append((cx, cy))
    
    def get_tracked_objects(self):
        """Aktif track'leri döndür"""
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
        """Çizim (Jetson için optimize - basit)"""
        # Ham detections (gri, ince)
        if show_raw_detections and raw_detections:
            for det in raw_detections:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
        
        # Trails
        if show_trails:
            for obj_id, trail in self.track_trails.items():
                if len(trail) > 1:
                    color = self.colors[obj_id % len(self.colors)]
                    points = np.array(trail, dtype=np.int32)
                    cv2.polylines(frame, [points], False, color, 2)
        
        # Tracked objects
        for obj in self.get_tracked_objects():
            bbox = obj['bbox']
            obj_id = obj['id']
            score = obj['score']
            
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = self.colors[obj_id % len(self.colors)]
            
            # Bbox
            thickness = 3 if score > 0.7 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            label = f"ID:{obj_id} ({score:.2f})"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 4, color, -1)
        
        return frame
    
    def get_stats(self):
        """İstatistikler"""
        total_frames = self.frame_id
        return {
            **self.stats,
            'active_tracks': len(self.tracked_objects)
        }


# ============================================================================
# MAIN - JETSON NANO OPTİMİZE
# ============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║    JETSON NANO OPTİMİZE ÇOKLU NESNE TRACKING SİSTEMİ       ║")
    print("║         ByteTrack + Selective SiamRPN++ + Kalman           ║")
    print("║              TEKNOFEST SAVŞAN İHA 2025                     ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    
    # Paths
    YOLO_ENGINE_PATH = os.path.join(current_dir, "yolo", "best.engine")
    YOLO_PT_PATH = os.path.join(current_dir, "yolo", "best.pt")
    
    # YOLO model seçimi (best.engine öncelikli!)
    if os.path.exists(YOLO_ENGINE_PATH):
        YOLO_MODEL_PATH = YOLO_ENGINE_PATH
        YOLO_MODEL_TYPE = "TensorRT Engine"
        print("✅ best.engine bulundu - TensorRT kullanılacak (HIZLI)")
    elif os.path.exists(YOLO_PT_PATH):
        YOLO_MODEL_PATH = YOLO_PT_PATH
        YOLO_MODEL_TYPE = "PyTorch"
        print("⚠️  best.engine YOK - best.pt kullanılacak (YAVAŞ)")
    else:
        print("❌ HATA: YOLO model dosyası bulunamadı!")
        sys.exit(1)
    
    # Jetson Nano optimizasyonları
    # NOT: TensorRT engine 320x320 için derlenmiş, değiştirilemez!
    YOLO_IMG_SIZE = 320  # TensorRT engine boyutu (sabit)
    YOLO_CONF_THRESHOLD = 0.6  # 0.5 → 0.6 (daha az false positive)
    MAX_OBJECTS = 10
    
    print(f"📦 YOLO Model: {YOLO_MODEL_TYPE}")
    print(f"📂 Dosya: {os.path.basename(YOLO_MODEL_PATH)}")
    print(f"🖼️  Image Size: {YOLO_IMG_SIZE}x{YOLO_IMG_SIZE} (TensorRT engine sabit)")
    print(f"🎯 Confidence: {YOLO_CONF_THRESHOLD}")
    print(f"🔧 Algoritma: ByteTrack + Kalman (SiamRPN++ YOK)\n")
    
    # Model yükleme
    try:
        print("Model yükleme başlıyor...\n")
        
        # YOLO
        if YOLO_MODEL_TYPE == "TensorRT Engine":
            yolo_model = YOLO(YOLO_MODEL_PATH, task='detect')
            print("✓ YOLO yüklendi (TensorRT Engine - MAKSIMUM HIZ)")
        else:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print("✓ YOLO yüklendi (PyTorch)")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Device: {device}")
        
        # Jetson Optimized Tracker
        tracker = JetsonOptimizedTracker(
            device=device,
            max_objects=MAX_OBJECTS,
        )
        
        print("\n✅ Jetson Optimized Sistem Hazır\n")
    
    except Exception as e:
        print(f"❌ Hata: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera açılamadı")
        sys.exit(1)
    
    print("📷 Kamera açıldı - Jetson Nano Optimized Tracking")
    print("   • ByteTrack + Selective SiamRPN++")
    print("   • Kalman filter prediction")
    print("   • TensorRT YOLO (FP16)")
    print("   • 320x320 image size")
    print("   • Hedef: 12-18 FPS (5 nesne)")
    print("   • 'q' ile çıkış\n")
    
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    # Görselleştirme
    SHOW_RAW_DETECTIONS = True
    SHOW_TRAILS = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO detection (Jetson optimize)
            if YOLO_MODEL_TYPE == "TensorRT Engine":
                results = yolo_model.predict(
                    frame,
                    imgsz=YOLO_IMG_SIZE,  # 256x256
                    conf=YOLO_CONF_THRESHOLD,
                    device=0,
                    verbose=False,
                    half=True  # FP16
                )
            else:
                results = yolo_model.predict(
                    frame,
                    imgsz=YOLO_IMG_SIZE,
                    conf=YOLO_CONF_THRESHOLD,
                    device=0 if torch.cuda.is_available() else 'cpu',
                    verbose=False
                )
            
            # Detections
            detections = []
            if len(results[0].boxes) > 0:
                for det in results[0].boxes:
                    bbox_xyxy = det.xyxy[0].cpu().numpy()
                    conf = det.conf[0].item()
                    detections.append({
                        'bbox': bbox_xyxy,
                        'score': conf
                    })
            
            # Tracking güncelle
            tracker.update(frame, detections)
            
            # Çiz
            display_frame = tracker.draw(
                frame.copy(),
                show_raw_detections=SHOW_RAW_DETECTIONS,
                raw_detections=detections,
                show_trails=SHOW_TRAILS
            )
            
            # FPS
            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Info panel
            tracked_count = len(tracker.get_tracked_objects())
            det_count = len(detections)
            stats = tracker.get_stats()
            
            # Panel
            cv2.rectangle(display_frame, (5, 5), (350, 150), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (5, 5), (350, 150), (50, 50, 50), 2)
            
            # FPS
            fps_color = (0, 255, 0) if current_fps >= 15 else (0, 165, 255)
            cv2.putText(display_frame, f"FPS: {current_fps}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
            
            # Tracked
            cv2.putText(display_frame, f"Tracked: {tracked_count}/{MAX_OBJECTS}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Detections
            cv2.putText(display_frame, f"Detections: {det_count}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)
            
            # Engine
            model_text = "TensorRT" if YOLO_MODEL_TYPE == "TensorRT Engine" else "PyTorch"
            model_color = (0, 255, 0) if YOLO_MODEL_TYPE == "TensorRT Engine" else (0, 165, 255)
            cv2.putText(display_frame, f"Engine: {model_text}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, model_color, 1)
            
            # Platform
            cv2.putText(display_frame, "Jetson Nano Optimized (ByteTrack)", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
            
            cv2.imshow("Jetson Optimized Multi-Object Tracker", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n👋 Çıkış yapıldı")
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        traceback.print_exc()
    finally:
        # İstatistikler
        print("\n📊 PERFORMANS İSTATİSTİKLERİ:")
        print("=" * 50)
        stats = tracker.get_stats()
        print(f"Toplam track oluşturuldu: {stats['total_tracks_created']}")
        print(f"Toplam track kayıp: {stats['total_tracks_lost']}")
        print(f"ByteTrack only: {stats['bytetrack_only']}")
        print(f"Aktif track: {stats['active_tracks']}")
        
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # Headless mode'da GUI yok
        print("✅ Temizlendi")

