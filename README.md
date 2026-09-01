# İHA Takip ve Kontrol Sistemi

Gerçek zamanlı nesne takip ve uçuş kontrol sistemi. Kameradan gelen görüntüde
hedefi bulur, kesintisiz takip eder ve uçuş kontrolcüsüne yönelme komutu
gönderir. TEKNOFEST Savaşan İHA yarışması için yazıldı, Jetson Nano üzerinde
çalışacak şekilde optimize edildi.

**Modelden bağımsızdır** — hangi YOLO modelini verirsen onun tespit ettiği
nesneyi takip eder. İHA'ya özel bir şey yapmaz.

---

## Çalışma zinciri

```
kamera / video
      │
      ▼
TESPİT        YOLO  (PyTorch · ONNX · TensorRT)
      │       çıktı: [x1, y1, x2, y2, skor]
      ▼
TAKİP         ByteTrack + Kalman filtresi
      │       iki aşamalı eşleştirme, kalıcı kimlik
      ▼
KİLİT         hedef üzerinde kesintisiz süre sayacı
      │
      ▼
KONTROL       dört ayrı PID döngüsü
      │       yaw · pitch · roll · hız
      ▼
ÇIKIŞ         UART / MAVLink  +  OpenCV görselleştirme
```

---

## Hızlı başlangıç

```bash
git clone https://github.com/MYusufTOSUN/ihatracker.git
cd ihatracker
pip install -r requirements.txt
python iha_tracking_system.py
```

**Ağırlık dosyası olmadan da çalışır.** `yolo/best.pt` bulunamazsa hazır bir
COCO modeline düşer ve onu ilk çalıştırmada kendisi indirir (~6 MB). Yani
depoyu klonlayıp doğrudan deneyebilirsin — varsayılan ayarla insan takip eder.

```bash
python iha_tracking_system.py --source ucus.mp4          # video dosyası
python iha_tracking_system.py --source 1                  # ikinci kamera
python iha_tracking_system.py --source ucus.mp4 --record cikti.mp4
python iha_tracking_system.py --no-display                # başlıksız / SSH
```

Çıkmak için görüntü penceresinde `q`.

---

## Kendi modelinle kullanmak

**1. Ağırlığı yerleştir.** Proje kök dizininde `yolo` adında bir klasör aç ve
eğitilmiş modelini içine koy:

```
ihatracker/
├── iha_tracking_system.py
└── yolo/
    └── best.pt          ← kendi modelin
```

Sistem açılışta `yolo/` klasörünü tarar ve şu sırayla ilk bulduğunu kullanır:

| sıra | dosya | not |
|---|---|---|
| 1 | `best_fp16_640.engine` | TensorRT — en hızlısı, yalnız CUDA varsa |
| 2 | `best.engine` | TensorRT |
| 3 | `best.pt` | PyTorch — her yerde çalışır |
| 4 | `best.onnx` | ONNX Runtime |

Başka bir yol vermek istersen: `--model yol/model.pt`

> **TensorRT motoru taşınmaz.** `.engine` dosyası üretildiği GPU mimarisine ve
> TensorRT sürümüne bağlıdır; başka makinede açılmaz. Jetson'da kullanacaksan
> motoru **Jetson üzerinde** üretmelisin.

**2. Sınıfını seç.** Modelin birden fazla sınıf tespit ediyorsa hangisini takip
edeceğini söyle:

```bash
python iha_tracking_system.py --classes 0        # yalnız 0. sınıf
python iha_tracking_system.py --classes 0 2 3    # birden fazla
```

**3. Ayarları kendi durumuna göre bul.** Aşağıdaki bölüm bunun için.

---

## Ayarlama

Varsayılanlar makul bir başlangıç ama **her senaryo farklı.** Hedefin boyutu,
hızı, kameranın çözünürlüğü ve donanımın gücü değiştikçe bu değerler değişir.
Doğru yöntem: bir kayıt al, `--record` ile çıktı üret, izle, ayarla, tekrarla.

### Hız / doğruluk dengesi

| seçenek | düşürürsen | yükseltirsen |
|---|---|---|
| `--imgsz` | daha hızlı, küçük hedefi kaçırır | küçük hedefi görür, yavaşlar |
| `--skip` | her kare tespit, en doğru, en yavaş | daha hızlı, aradakiler Kalman tahmini |
| `--conf` | daha çok tespit, daha çok yanlış alarm | daha az yanlış alarm, hedefi kaçırabilir |

Jetson gibi sınırlı donanımda ilk elinizi atacağınız yer `--imgsz` ve `--skip`.
`--imgsz 256 --skip 3` çoğu durumda gözle fark edilmeyen bir kayıpla ciddi hız
kazandırır.

### İki aşamalı eşik (ByteTrack)

Bu iki değer takibin kalitesini belirliyor:

| seçenek | anlamı |
|---|---|
| `--conf` | **yüksek eşik** — bunun üstündeki tespit yeni iz başlatabilir |
| `--conf-low` | **düşük eşik** — bu aralıktakiler yeni iz başlatmaz, yalnız mevcut izi ayakta tutar |

Hedef kısmen kapandığında ya da hızlı hareket ettiğinde tespit güveni düşer.
Tek eşikli bir sistem o tespiti çöpe atar ve iz kopar. İkinci eşik onu yakalar.

- Kimlik sık değişiyorsa → `--conf-low` düşür (0.05'e kadar)
- Hayalet izler oluşuyorsa → `--conf` yükselt

### Takip davranışı

`SystemConfig` içinden ayarlanır (`iha_tracking_system.py` başında):

| alan | ne yapar | ne zaman değiştirilir |
|---|---|---|
| `sort_max_age` | iz kaç kare kayıp kalınca silinir | uzun kapanmalar varsa yükselt |
| `sort_min_hits` | iz kaç tespitte onaylanır | yanlış alarm çoksa yükselt |
| `sort_iou_threshold` | eşleştirme sıkılığı | hedef hızlıysa düşür |
| `max_objects` | aynı anda kaç iz | sahnede çok nesne varsa yükselt |
| `lock_duration_required` | kesintisiz kilit süresi | yarışma kuralı — 4 sn |
| `lock_lost_timeout` | kilit kaç sn kayıpta bozulur | kare atlama fazlaysa yükselt |

### Tespit filtreleri

Yanlış tespitleri geometriyle eler:

| alan | ne eler |
|---|---|
| `min_box_area` | çok küçük parazitleri |
| `max_box_area_percent` | ekranı kaplayan hatalı kutuları |
| `min_aspect_ratio` / `max_aspect_ratio` | şeritimsi, olamayacak oranları |

Hedefin çok uzakta ve küçük görünüyorsa `min_box_area`'yı düşür, yoksa gerçek
tespitler elenir.

### PID

Uçuş kontrolcüsüne giden komutların sertliği. `SystemConfig` içinde her eksen
için ayrı `kp`, `ki`, `kd`. Klasik yöntem: önce `ki` ve `kd` sıfırken `kp`'yi
salınım başlayana kadar yükselt, sonra geri çek, ardından `kd` ile salınımı
söndür, en son `ki` ile kalıcı hatayı kapat.

`max_yaw_rate`, `max_pitch_rate`, `max_roll_rate`, `max_speed` çıkışı
sınırlar — bunlar güvenlik sınırıdır, aracının kapasitesine göre ayarla.

Tam liste: `python iha_tracking_system.py --help`

---

## Neden ByteTrack

İlk sürüm her nesne için ayrı model örneği çalıştırıyor ve eşleştirmeyi düz IoU
ile yapıyordu: yavaştı ve nesneler birbirini kestiğinde kimlikler karışıyordu.

ByteTrack tek modeli bütün nesneler için paylaşır ve eşleştirmeyi **iki turda**
yapar:

| tur | hangi tespitler | neye eşleştirilir |
|---|---|---|
| 1 | yüksek güven (`--conf` üstü) | tüm izler |
| 2 | düşük güven (`--conf-low` ile `--conf` arası) | eşleşmeden kalan izler |

İkinci tur kritik: hedef kısmen kapandığında güven skoru düşer, tek turlu bir
eşleştirici o tespiti atıp kimliği kaybeder. Ölçüm — 26 karelik bir kapanma
senaryosu:

| | izin bildirildiği kare |
|---|---|
| tek aşamalı | **19 / 26** |
| iki aşamalı | **26 / 26** |

Kesintisiz kilit şartı olan bir sistemde bu 7 karelik boşluk doğrudan kilidin
bozulması demek.

---

## Kare atlama

`--skip N` ile YOLO her karede değil N karede bir çalışır; aradaki karelerde
konum Kalman filtresiyle tahmin edilir. Sınırlı donanımda FPS'i belirgin
şekilde yükseltir.

Bunun bir inceliği var: atlanan kareler **kayıp tespit değil**, bilinçli bir
tasarruf. Takipçi bu iki durumu ayırt etmezse izler her atlanan karede
cezalandırılır ve `sort_min_hits` eşiği hiçbir zaman aşılamaz. Ölçüm:

| | izin bildirildiği kare |
|---|---|
| ayrım yapılmadan, `--skip 2` | 90 karede **1** |
| ayrım yapılarak, `--skip 2` | 90 karede **86** |
| ayrım yapılarak, `--skip 3` | 90 karede **82** |

Bu yüzden takipçi o karede tespit çalışıp çalışmadığını bilir ve izleri yalnız
**gerçek** kayıplarda yaşlandırır.

---

## Dosyalar

| dosya | ne yapar |
|---|---|
| `iha_tracking_system.py` | **ana sistem** — tespit, ByteTrack takip, kilit, PID, seri port |
| `bytetrack_multi_tracker.py` | yalnız takip katmanı, bağımsız çalışır |
| `bytetrack_jetson_optimized.py` | Jetson Nano için ayarlanmış sürüm |
| `hybrid_tracker.py` | daha hafif SORT tabanlı alternatif |
| `test_kurulum.py` | kurulum doğrulama — kütüphaneler, CUDA, TensorRT, kamera |
| `test_tracker.py` | takipçi birim testleri |

---

## Testler

```bash
pip install pytest
pytest test_tracker.py -v
```

15 test; kamera, model veya GPU gerektirmez, saniyeler içinde biter. Çoğu
düzeltilmiş bir hatayı kilitleyen regresyon testidir — kare atlamada iz
sürekliliği, kapanmada kimlik korunması, PID çıkış sınırları, dejenere kutuda
NaN oluşmaması.

Kurulumu doğrulamak için:

```bash
python test_kurulum.py
```

---

## Gereksinimler

Python 3.8+. **GPU şart değil** — CPU'da da çalışır, sadece daha yavaş.

```bash
pip install -r requirements.txt
```

PyTorch'u platformuna göre ayrıca kur — Jetson Nano için NVIDIA'nın kendi
tekerlekleri gerekiyor, `pip install torch` doğru sürümü getirmiyor.

TensorRT isteğe bağlıdır; yoksa sistem PyTorch modeline düşer ve uyarı basar.

---

## Depoda olmayanlar

Eğitilmiş ağırlıklar boyutları nedeniyle depoya alınmadı (`yolo/` klasörü
`.gitignore` içinde). Kendi modelinle çalıştırmak için yukarıdaki
**Kendi modelinle kullanmak** bölümüne bak.

## Lisans

MIT — bkz. [LICENSE](LICENSE).
