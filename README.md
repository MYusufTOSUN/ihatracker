# İHA Takip ve Kontrol Sistemi

TEKNOFEST Savaşan İHA yarışması için gerçek zamanlı nesne takip ve uçuş kontrol
sistemi. Kameradan gelen görüntüde hedefi bulur, kesintisiz takip eder ve uçuş
kontrolcüsüne yönelme komutları gönderir.

Jetson Nano üzerinde çalışacak şekilde optimize edildi.

---

## Çalışma zinciri

```
kamera
  │
  ▼
TESPİT        YOLOv11-s + TensorRT
              320×320 giriş, FP16
  │           çıktı: [x1, y1, x2, y2, skor]
  ▼
TAKİP         ByteTrack
              Kalman filtresi ile hareket tahmini
              Hungarian algoritması ile eşleştirme
  │           çıktı: [id, kutu, merkez]
  ▼
KONTROL       dört ayrı PID döngüsü
              hata = ekran merkezi − hedef merkezi
  │           yaw · pitch · roll · throttle
  ▼
ÇIKIŞ         UART / MAVLink  +  OpenCV görselleştirme
```

## Neden ByteTrack

İlk sürüm her nesne için ayrı bir model örneği çalıştırıyor ve eşleştirmeyi düz
IoU ile yapıyordu. İki sorunu vardı: yavaştı ve nesneler birbirini kestiğinde
kimlikler karışıyordu.

ByteTrack tek modeli bütün nesneler için paylaşıyor ve eşleştirmeyi **iki
aşamada** yapıyor:

| aşama | hangi tespitler | neye eşleştirilir |
|---|---|---|
| 1 | yüksek güven (> 0,6) | mevcut takipçiler |
| 2 | düşük güven (0,1 – 0,6) | eşleşmeden kalan takipçiler |

İkinci aşama önemli: hedef kısmen kapandığında güven skoru düşüyor ve tek
aşamalı bir eşleştirici o tespiti çöpe atıp kimliği kaybediyor. ByteTrack onu
ikinci turda yakalıyor.

Kalman filtresi de aradaki karelerde nesnenin nerede olacağını tahmin ettiği
için, tespit bir kare atlasa bile takip kopmuyor.

## Dosyalar

| dosya | ne yapar |
|---|---|
| `iha_tracking_system.py` | tam sistem — tespit, takip, PID, seri port, görselleştirme |
| `bytetrack_multi_tracker.py` | ByteTrack + Kalman çoklu nesne takibi |
| `bytetrack_jetson_optimized.py` | Jetson Nano için optimize edilmiş sürüm |
| `hybrid_tracker.py` | YOLO + SORT ile daha hafif takip |
| `test_kurulum.py` | kurulum doğrulama — kütüphaneler, CUDA, TensorRT, kamera |

## Model ağırlıkları depoda değil

Eğitilmiş ağırlıklar boyutları nedeniyle depoya alınmadı:

```
yolo/best.pt                 45 MB
yolo/best.onnx               89 MB
yolo/best.onnx.data          90 MB
yolo/best_fp16_640.engine    48 MB
```

Kendi ağırlıklarınla çalıştırmak için `yolo/` klasörünü oluşturup `best.pt`
dosyasını içine koyman yeterli. TensorRT motorunu (`.engine`) **çalıştıracağın
makinede üretmelisin** — motor dosyası GPU mimarisine ve TensorRT sürümüne bağlı,
başka makinede üretilmiş bir motor açılmıyor.

## Kurulum

Gereksinim: Python 3.8+, NVIDIA GPU. Jetson için JetPack 4.6+, CUDA 10.2+,
TensorRT 8.0+.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch'u platformuna göre ayrıca kur — Jetson Nano için NVIDIA'nın kendi
tekerlekleri gerekiyor, `pip install torch` doğru sürümü getirmiyor.

Kurulumu doğrula:

```bash
python test_kurulum.py
```

Kütüphaneleri, CUDA erişimini, TensorRT'yi ve kamerayı sırayla kontrol eder.

## Çalıştırma

```bash
python iha_tracking_system.py
```

Yalnız takip tarafını denemek için:

```bash
python bytetrack_multi_tracker.py
```

## Lisans

MIT — bkz. [LICENSE](LICENSE).
