"""
YOLO + SORT Çoklu Nesne Takip Scripti
-------------------------------------

Bu script, Siam/SiamRPN++ bileşenleri olmadan yalnızca:
- YOLO (tespit)
- SORT (Kalman Filter + IoU matching)
ile çoklu nesne takibi yapar.
"""

import cv2
import torch
from ultralytics import YOLO
import time
import numpy as np
import os
import sys
import traceback
from collections import deque
from scipy.optimize import linear_sum_assignment


current_dir = os.path.dirname(os.path.abspath(__file__))


class KalmanBoxTracker:
    """
    SORT için kullanılan basit Kalman Filter tabanlı bbox takipçisi.
    State: [x, y, s, r, vx, vy, vs]
    """
    count = 0

    def __init__(self, bbox):
        """bbox: [x1, y1, x2, y2]"""
        from filterpy.kalman import KalmanFilter

        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # Geçiş matrisi
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        # Ölçüm matrisi
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        # Gürültü ayarları
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

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
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h) if h != 0 else 1.0
        return np.array([x, y, s, r]).reshape((4, 1))

    def _convert_x_to_bbox(self, x):
        """[x,y,s,r] -> [x1,y1,x2,y2]"""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w if w != 0 else 1.0
        return np.array([
            x[0] - w / 2.0,
            x[1] - h / 2.0,
            x[0] + w / 2.0,
            x[1] + h / 2.0,
        ]).flatten()

    def update(self, bbox):
        """Yeni ölçüm ile güncelle"""
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
        return self.history[-1]

    def get_state(self):
        """Mevcut bbox'ı [x1,y1,x2,y2] olarak döndür"""
        return self._convert_x_to_bbox(self.kf.x)


class SortMultiTracker:
    """
    Klasik SORT algoritmasına yakın çalışan çoklu nesne takipçisi.
    - KalmanBoxTracker ile state tahmini
    - IoU + Hungarian (linear_sum_assignment) ile eşleme
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, max_trail_length=30):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers = []
        self.frame_count = 0

        self.colors = self._generate_colors(100)
        self.track_trails = {}  # id -> deque[(x,y)]
        self.max_trail_length = max_trail_length

        self.stats = {
            "total_tracks_created": 0,
        }

    def _generate_colors(self, n):
        np.random.seed(42)
        colors = []
        for _ in range(n):
            colors.append(tuple(map(int, np.random.randint(50, 255, 3))))
        return colors

    def _iou_batch(self, bboxes1, bboxes2):
        """IoU matrisi (N x M)"""
        bboxes1 = np.asarray(bboxes1)
        bboxes2 = np.asarray(bboxes2)

        if len(bboxes1) == 0 or len(bboxes2) == 0:
            return np.zeros((len(bboxes1), len(bboxes2)))

        b1 = np.expand_dims(bboxes1, 1)  # (N,1,4)
        b2 = np.expand_dims(bboxes2, 0)  # (1,M,4)

        xx1 = np.maximum(b1[..., 0], b2[..., 0])
        yy1 = np.maximum(b1[..., 1], b2[..., 1])
        xx2 = np.minimum(b1[..., 2], b2[..., 2])
        yy2 = np.minimum(b1[..., 3], b2[..., 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
        area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
        union = area1 + area2 - inter

        return inter / np.maximum(union, 1e-6)

    def update(self, detections):
        """
        SORT güncellemesi.
        Args:
            detections: [{'bbox': [x1,y1,x2,y2], 'score': float}, ...]
        """
        self.frame_count += 1

        if detections:
            dets = np.array([d["bbox"] for d in detections], dtype=float)
            scores = np.array([d["score"] for d in detections], dtype=float)
        else:
            dets = np.empty((0, 4), dtype=float)
            scores = np.empty((0,), dtype=float)

        # 1) Mevcut tracker'ların prediction'larını al
        trks = np.empty((len(self.trackers), 4), dtype=float)
        for t, trk in enumerate(self.trackers):
            trks[t, :] = trk.predict()

        # 2) Matching
        matched, unmatched_trks, unmatched_dets = [], [], []

        if len(trks) > 0 and len(dets) > 0:
            iou_matrix = self._iou_batch(trks, dets)
            cost_matrix = 1.0 - iou_matrix

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for t in range(len(self.trackers)):
                if t not in row_ind:
                    unmatched_trks.append(t)

            for d in range(len(dets)):
                if d not in col_ind:
                    unmatched_dets.append(d)

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched.append((r, c))
                else:
                    unmatched_trks.append(r)
                    unmatched_dets.append(c)
        else:
            unmatched_trks = list(range(len(self.trackers)))
            unmatched_dets = list(range(len(dets)))

        # 3) Eşleşen tracker'ları güncelle
        for t_idx, d_idx in matched:
            self.trackers[t_idx].update(dets[d_idx])

        # 4) Eşleşmeyen detection'lar için yeni tracker oluştur
        for d_idx in unmatched_dets:
            new_trk = KalmanBoxTracker(dets[d_idx])
            self.trackers.append(new_trk)
            self.stats["total_tracks_created"] += 1

        # 5) Yaşı çok büyüyen tracker'ları sil
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                trk_id = trk.id
                self.trackers.pop(i)
                if trk_id in self.track_trails:
                    del self.track_trails[trk_id]

        # 6) Trail'leri güncelle
        for trk in self.trackers:
            if trk.time_since_update == 0:
                bbox = trk.get_state()
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                if trk.id not in self.track_trails:
                    self.track_trails[trk.id] = deque(maxlen=self.max_trail_length)
                self.track_trails[trk.id].append((cx, cy))

    def get_tracked_objects(self):
        """Aktif tracker'ları döndür"""
        results = []
        for trk in self.trackers:
            if trk.time_since_update < 1 and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                bbox = trk.get_state()
                results.append(
                    {
                        "id": trk.id,
                        "bbox": bbox,
                        "score": 1.0,
                    }
                )
        return results

    def draw(self, frame, show_trails=True):
        """
        Takip edilen nesneleri ve istenirse izlerini çiz.
        """
        # Trails
        if show_trails:
            for obj_id, trail in self.track_trails.items():
                if len(trail) > 1:
                    color = self.colors[obj_id % len(self.colors)]
                    for i in range(1, len(trail)):
                        cv2.line(frame, trail[i - 1], trail[i], color, 2)

        # Bbox'lar
        for obj in self.get_tracked_objects():
            bbox = obj["bbox"]
            obj_id = obj["id"]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = self.colors[obj_id % len(self.colors)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{obj_id}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 3, color, -1)

        return frame


# ---------------------------------------------------------------------------
# YOLO Model Seçimi
# ---------------------------------------------------------------------------


# ═══════════════════════════════════════════════════════════════════
# GIRIS NOKTASI
# ═══════════════════════════════════════════════════════════════════

def main():
    """Uygulamayi calistir.

    NEDEN FONKSIYON ICINDE: onceden bu blok modul seviyesindeydi, yani
    dosyayi import etmek MODEL YUKLEYIP kamerayi aciyordu.
    """
    BEST_FP16_ENGINE_PATH = os.path.join(current_dir, "yolo", "best_fp16_640.engine")
    YOLO_ENGINE_PATH = os.path.join(current_dir, "yolo", "best.engine")
    YOLO_PT_PATH = os.path.join(current_dir, "yolo", "best.pt")

    if os.path.exists(BEST_FP16_ENGINE_PATH):
        YOLO_MODEL_PATH = BEST_FP16_ENGINE_PATH
        YOLO_MODEL_TYPE = "TensorRT Engine (640 FP16)"
        YOLO_IMG_SIZE = 640
    elif os.path.exists(YOLO_ENGINE_PATH):
        YOLO_MODEL_PATH = YOLO_ENGINE_PATH
        YOLO_MODEL_TYPE = "TensorRT Engine"
        YOLO_IMG_SIZE = 320
    elif os.path.exists(YOLO_PT_PATH):
        YOLO_MODEL_PATH = YOLO_PT_PATH
        YOLO_MODEL_TYPE = "PyTorch"
        YOLO_IMG_SIZE = 640
    else:
        print("❌ HATA: YOLO model dosyası bulunamadı!")
        print(f"   Aranan: {BEST_FP16_ENGINE_PATH}")
        print(f"   Veya:  {YOLO_ENGINE_PATH}")
        print(f"   Veya:  {YOLO_PT_PATH}")
        sys.exit(1)

    YOLO_CONF_THRESHOLD = 0.5

    print(f"📦 YOLO Model Tipi: {YOLO_MODEL_TYPE}")
    print(f"📂 Dosya: {os.path.basename(YOLO_MODEL_PATH)}")


    # ---------------------------------------------------------------------------
    # Model Yükleme
    # ---------------------------------------------------------------------------
    try:
        print("\nModel yükleme başlıyor...")

        if "TensorRT" in YOLO_MODEL_TYPE:
            yolo_model = YOLO(YOLO_MODEL_PATH, task="detect")
            print("✓ YOLO yüklendi (TensorRT Engine)")
        else:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print("✓ YOLO yüklendi (PyTorch)")

        use_gpu = torch.cuda.is_available()
        print(f"✓ CUDA: {'Var' if use_gpu else 'Yok'}")

        sort_tracker = SortMultiTracker(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            max_trail_length=30,
        )

        print("✅ YOLO + SORT sistemi hazır\n")

    except Exception as e:
        print(f"❌ Hata: {e}")
        traceback.print_exc()
        sys.exit(1)


    # ---------------------------------------------------------------------------
    # Kamera ve Ana Döngü
    # ---------------------------------------------------------------------------

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera açılamadı")
        sys.exit(1)



    print("📷 Kamera açıldı - YOLO + SORT Çoklu Nesne Takibi")
    print("   • Siam/SiamRPN++ YOK")
    print("   • YOLO tespit + SORT (Kalman + IoU)")
    print("   • 'q' ile çıkış\n")

    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0

    SHOW_TRAILS = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO tespit
            if "TensorRT" in YOLO_MODEL_TYPE:
                results = yolo_model.predict(
                    frame,
                    imgsz=YOLO_IMG_SIZE,
                    conf=YOLO_CONF_THRESHOLD,
                    device=0 if torch.cuda.is_available() else 'cpu',
                    verbose=False,
                    half=True,
                )
            else:
                results = yolo_model.predict(
                    frame,
                    imgsz=YOLO_IMG_SIZE,
                    conf=YOLO_CONF_THRESHOLD,
                    device=0 if torch.cuda.is_available() else "cpu",
                    verbose=False,
                )

            detections = []
            if len(results[0].boxes) > 0:
                for det in results[0].boxes:
                    bbox_xyxy = det.xyxy[0].cpu().numpy()
                    conf = det.conf[0].item()
                    detections.append({"bbox": bbox_xyxy, "score": conf})

            # SORT güncelle
            sort_tracker.update(detections)

            # Çizim
            display_frame = sort_tracker.draw(frame.copy(), show_trails=SHOW_TRAILS)

            # FPS
            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()

            # Bilgi paneli
            tracked_count = len(sort_tracker.get_tracked_objects())
            det_count = len(detections)

            cv2.rectangle(display_frame, (5, 5), (320, 120), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (5, 5), (320, 120), (50, 50, 50), 2)

            cv2.putText(
                display_frame,
                f"FPS: {current_fps}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                display_frame,
                f"Tracked: {tracked_count}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"Detections: {det_count}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 150, 0),
                1,
            )

            cv2.putText(
                display_frame,
                "YOLO + SORT",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            cv2.imshow("YOLO + SORT Multi-Object Tracker", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
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
