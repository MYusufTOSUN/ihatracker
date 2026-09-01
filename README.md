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
      │       iki aşamalı eşleştirme · kamera hareketi telafisi
      │       görünümle yeniden tanıma · kalıcı kimlik
      ▼
KİLİT         hedef üzerinde kesintisiz süre sayacı
      │
      ▼
KONTROL       dört ayrı PID döngüsü
      │       yaw · pitch · roll · hız
      ▼
ÇIKIŞ         UART (düz metin)  +  OpenCV görselleştirme
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

### Kamera hareketi ve yeniden tanıma

İkisi de **varsayılan açık.** Kapatmak için:

| seçenek | ne yapar | ne zaman kapatılır |
|---|---|---|
| `--no-cmc` | kamera hareketi telafisini kapatır | kamera sabitse (~3 ms/kare kazanç) |
| `--no-reid` | görünümle yeniden tanımayı kapatır | tek hedef varsa, ya da hedefler birbirine çok benziyorsa |
| `--reid-thresh` | görünüm eşleşme sıkılığı (varsayılan 0.35) | yanlış eşleşme varsa düşür |

`--reid-thresh` iki yönlü bir takas: **düşürmek** yanlış eşleşmeyi azaltır ama
kaybolan hedefi geri kazanmayı zorlaştırır; **yükseltmek** tersini yapar.

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
| `reid_hafiza_karesi` | kayıp iz kaç kare hatırlanır | kapanmalar uzunsa yükselt |

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

### Seri çıkış

`--serial` ile UART'a (varsayılan `/dev/ttyTHS1`, 115200) **düz metin** yazılır:

```
YAW,PITCH,ROLL,THR
        örn.  12.40,-3.15,3.72,0.28
```

**MAVLink değildir.** Uçuş kontrolcünün bu satırı okuyup çözmesi gerekir; ArduPilot
veya PX4'e doğrudan bağlamak istersen araya pymavlink ile bir köprü koymalısın.

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

## Kamera hareketi telafisi

Kalman filtresi **sabit hız** varsayar: nesnenin kendi hareketini öğrenir ve
sürüklenmesini tahmin eder. Kamera sabitken bu doğrudur.

İHA'da değil. Kamera dönüyor, yalpalıyor, irtifa değiştiriyor — o anda
görüntüdeki **her şey** birlikte kayıyor. Kalman bunu nesnenin kendi hareketi
sanıyor, tahmin gerçek konumdan uzaklaşıyor, IoU düşüyor, kimlik kopuyor.

Çözüm, [BoT-SORT](https://arxiv.org/abs/2206.14651)'un yaklaşımı: kareler arası
global hareketi ölç, **tüm izlerin durumundan çıkar.** Seyrek optik akış
(`goodFeaturesToTrack` + Lucas-Kanade) ile nokta eşleşmeleri bulunur, aradaki
afin dönüşüm RANSAC ile kestirilir. Hareketli nesnelerin üzerindeki noktalar
aykırı değer olarak elenir, geriye sahnenin geneli — yani kamera — kalır.

Ölçüm — kameranın sert yalpaladığı 120 karelik sentetik senaryo, tek hedef.
**Doğru sonuç 1 kimliktir**; fazlası kopmuş takip demek:

| | CMC kapalı | CMC açık |
|---|---|---|
| `--skip 1` | 67/120 kare, **23 kimlik** | 120/120 kare, **1 kimlik** |
| `--skip 2` | 44/120 kare, **11 kimlik** | 118/120 kare, **1 kimlik** |
| `--skip 3` | 29/120 kare, **9 kimlik** | 116/120 kare, **1 kimlik** |

Kare atlama arttıkça fark büyüyor: ara karelerde konum tamamen Kalman
tahminine kaldığı için, telafi edilmemiş kamera hareketi orada birikiyor.

Maliyet CPU'da kare başına ~3 ms (640×480). Uçtan uca aynı videoda 22,5 → 20,6
FPS. Kamera sabitse `--no-cmc` ile kapat.

---

## Görünümle yeniden tanıma (Re-ID)

IoU ve Kalman yalnız **harekete** bakar. Hedef bir engelin arkasına girip
birkaç saniye sonra **başka bir yerde** çıktığında hareket modeli çaresizdir:
tahmin edilen konum artık geçersiz, IoU sıfır. İz silinir, nesne geri
geldiğinde **yeni kimlik** alır.

Bu yüzden süresi dolan izler doğrudan atılmıyor; bir **kayıp iz havuzunda**
`reid_hafiza_karesi` kadar bekletiliyor. Eşleşmeyen yeni bir tespit geldiğinde
önce bu havuza bakılıyor ve görünüm tutarsa **eski kimlik geri veriliyor.**

Görünüm tanımı HSV renk histogramı; izin profili üstel hareketli ortalamayla
güncelleniyor, böylece anlık bulanıklık veya kısmi kapanma profili bozmuyor.

**Neden derin ağ değil:** OSNet gibi Re-ID ağları daha ayırt edicidir ama her
kare, her nesne için ayrı bir ileri geçiş demek — Jetson Nano'da bu, tespit
modelinin kendisinden pahalıya gelebiliyor ve depoya ikinci bir model ekliyor.
Renk histogramı kare başına ~0,1 ms, sıfır bağımlılık. **Ayırt etme gücü derin
ağların altında — bu bilinçli bir takas.** Daha güçlüsü gerekirse aynı arayüzü
(`cikar` / `mesafe`) veren bir sınıf yazmak yeterli, takipçi kodu değişmez.

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

25 test; kamera, model veya GPU gerektirmez, saniyeler içinde biter. Çoğu
düzeltilmiş bir hatayı kilitleyen regresyon testidir — kare atlamada iz
sürekliliği, kapanmada kimlik korunması, kamera hareketinin doğru kestirilmesi
ve ötelemenin hıza eklenmemesi, uzun kapanmadan sonra kimliğin geri gelmesi ve
farklı bir nesneye **yanlışlıkla** verilmemesi, PID çıkış sınırları, dejenere
kutuda NaN oluşmaması.

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
