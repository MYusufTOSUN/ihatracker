#!/usr/bin/env python3
"""
Takipci birim testleri.

    pytest test_tracker.py -v

Buradaki testlerin cogu GERCEK hatalari yakalamak icin yazildi; her biri
duzeltilen bir davranisi kilitliyor. Hicbiri kamera, model ya da GPU
gerektirmiyor — sentetik kutularla calisiyorlar, saniyeler icinde bitiyor.
"""

import numpy as np
import pytest

from iha_tracking_system import (
    SystemConfig,
    ByteTrackTracker,
    KalmanBoxTracker,
    PIDController,
)


def kutu(x, y, w=40, h=100, skor=0.9):
    """Tek satirlik tespit uret: [x1, y1, x2, y2, skor]"""
    return [x, y, x + w, y + h, skor]


def dizi(*kutular):
    return np.array(kutular, dtype=float) if kutular else np.empty((0, 5))


# ═══════════════════════════════════════════════════════════════════════════
# KARE ATLAMA
# ═══════════════════════════════════════════════════════════════════════════

def test_kare_atlamada_izler_bildirilmeye_devam_eder():
    """
    REGRESYON: predict() atlanan karelerde hit_streak'i sifirliyordu.

    Sonuc: hit_streak asla 1'i gecemiyor, min_hits=2 esigi hicbir zaman
    asilamiyor ve takipci HICBIR iz bildirmiyordu. Varsayilan ayarlarda
    (skip=2, min_hits=2) sistem pratikte calismiyordu.
    """
    cfg = SystemConfig(yolo_skip_frames=2, sort_min_hits=2)
    trk = ByteTrackTracker(cfg)

    izli_kare = 0
    for i in range(1, 21):
        tespit_karesi = (i % 2 == 0)
        d = dizi(kutu(100 + i * 4, 100)) if tespit_karesi else dizi()
        izler = trk.update(d, tespit_karesi=tespit_karesi)
        if izler:
            izli_kare += 1

    # Duzeltmeden once bu sayi 1'di. Simdi izler atlanan karelerde de
    # bildirildigi icin karelerin buyuk cogunlugunda iz olmali.
    assert izli_kare >= 14, f"yalnizca {izli_kare}/20 karede iz bildirildi"


def test_hit_streak_atlanan_karede_sifirlanmaz():
    """Atlanan kare bir KAYIP degil; iz cezalandirilmamali."""
    t = KalmanBoxTracker(np.array([100.0, 100.0, 140.0, 200.0]))
    t.update(np.array([100.0, 100.0, 140.0, 200.0]))
    onceki = t.hit_streak

    t.predict(yaslandir=False)          # atlanan kare
    assert t.hit_streak == onceki
    assert t.time_since_update == 0

    t.predict(yaslandir=True)           # gercek kayip
    assert t.time_since_update == 1


def test_gercek_kayip_hala_yaslandirir():
    """Kare atlama duzeltmesi, gercek kayiplari maskelememeli."""
    cfg = SystemConfig(sort_max_age=3)
    trk = ByteTrackTracker(cfg)
    trk.update(dizi(kutu(100, 100)), tespit_karesi=True)

    for _ in range(6):
        trk.update(dizi(), tespit_karesi=True)   # YOLO calisti, hedef YOK

    assert len(trk.trackers) == 0, "max_age'i asan iz silinmeliydi"


# ═══════════════════════════════════════════════════════════════════════════
# IKI ASAMALI ESLESTIRME (ByteTrack)
# ═══════════════════════════════════════════════════════════════════════════

def test_dusuk_guvenli_tespit_izi_ayakta_tutar():
    """
    ByteTrack'in ASIL degeri: hedef kismen kapandiginda guven duser.
    Tek asamali eslestirici o tespiti atar ve kimlik kopar; ikinci tur
    onu yakalayip AYNI kimligi surdurur.
    """
    cfg = SystemConfig(track_high_thresh=0.5, track_low_thresh=0.1,
                       sort_min_hits=1)
    trk = ByteTrackTracker(cfg)

    for i in range(4):                                    # saglam tespitler
        trk.update(dizi(kutu(100 + i * 5, 100, skor=0.9)))
    kimlik = trk.trackers[0].id

    for i in range(4, 8):                                 # kapanma: guven dustu
        izler = trk.update(dizi(kutu(100 + i * 5, 100, skor=0.25)))

    assert len(trk.trackers) == 1, "dusuk guven yeni iz DOGURMAMALI"
    assert trk.trackers[0].id == kimlik, "kimlik korunmaliydi"


def test_dusuk_guvenli_tespit_yeni_iz_baslatmaz():
    """Gurultu iz uretmemeli: yeni iz yalniz yuksek guvenden dogar."""
    cfg = SystemConfig(track_high_thresh=0.5, track_low_thresh=0.1)
    trk = ByteTrackTracker(cfg)

    for _ in range(5):
        trk.update(dizi(kutu(300, 300, skor=0.2)))

    assert len(trk.trackers) == 0


def test_esik_altindaki_tespit_tamamen_yoksayilir():
    cfg = SystemConfig(track_high_thresh=0.5, track_low_thresh=0.1)
    trk = ByteTrackTracker(cfg)
    trk.update(dizi(kutu(100, 100, skor=0.05)))
    assert len(trk.trackers) == 0


# ═══════════════════════════════════════════════════════════════════════════
# KIMLIK SUREKLILIGI
# ═══════════════════════════════════════════════════════════════════════════

def test_hareketli_hedef_kimligini_korur():
    cfg = SystemConfig(sort_min_hits=1)
    trk = ByteTrackTracker(cfg)

    kimlikler = set()
    for i in range(25):
        izler = trk.update(dizi(kutu(50 + i * 6, 100)))
        kimlikler.update(t["id"] for t in izler)

    assert len(kimlikler) == 1, f"kimlik degisti: {kimlikler}"


def test_iki_hedef_karismaz():
    cfg = SystemConfig(sort_min_hits=1, max_objects=10)
    trk = ByteTrackTracker(cfg)

    for i in range(20):
        izler = trk.update(dizi(kutu(50 + i * 4, 100),
                                kutu(400 - i * 4, 300)))
    assert len(izler) == 2
    assert len({t["id"] for t in izler}) == 2


def test_max_objects_asilmaz():
    cfg = SystemConfig(max_objects=3, sort_min_hits=1)
    trk = ByteTrackTracker(cfg)
    trk.update(dizi(*[kutu(50 + i * 80, 100) for i in range(8)]))
    assert len(trk.trackers) <= 3


# ═══════════════════════════════════════════════════════════════════════════
# PID
# ═══════════════════════════════════════════════════════════════════════════

def test_pid_sifir_hatada_sifir_cikti():
    p = PIDController(kp=1.0, ki=0.1, kd=0.05)
    assert abs(p.compute(0.0, dt=0.1)) < 1e-9


def test_pid_limitleri_asilmaz():
    """Integral birikse bile cikti [output_min, output_max] disina cikmamali."""
    p = PIDController(kp=100.0, ki=10.0, kd=1.0,
                      output_min=-5.0, output_max=5.0)
    for _ in range(50):
        c = p.compute(1000.0, dt=0.1)
        assert -5.0 <= c <= 5.0, f"cikti limiti asti: {c}"


def test_pid_hatayi_dogru_yonde_karsilar():
    p = PIDController(kp=1.0, ki=0.0, kd=0.0,
                      output_min=-100.0, output_max=100.0)
    assert p.compute(10.0, dt=0.1) > 0
    p.reset()
    assert p.compute(-10.0, dt=0.1) < 0


def test_pid_reset_durumu_temizler():
    p = PIDController(kp=1.0, ki=1.0, kd=0.0,
                      output_min=-100.0, output_max=100.0)
    for _ in range(10):
        p.compute(5.0, dt=0.1)
    assert p.integral != 0.0
    p.reset()
    assert p.integral == 0.0
    assert p.prev_error == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# DAYANIKLILIK
# ═══════════════════════════════════════════════════════════════════════════

def test_bos_tespit_cokmez():
    trk = ByteTrackTracker(SystemConfig())
    for _ in range(10):
        assert trk.update(dizi()) == []


def test_tek_piksel_kutu_cokmez():
    """Bozuk/dejenere kutu Kalman'i NaN'a dusurmemeli."""
    trk = ByteTrackTracker(SystemConfig(sort_min_hits=1))
    for _ in range(5):
        trk.update(np.array([[100.0, 100.0, 101.0, 101.0, 0.9]]))
    for t in trk.trackers:
        assert not np.any(np.isnan(t.get_state()))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ═══════════════════════════════════════════════════════════════════════════
# KAMERA HAREKETI TELAFISI (CMC)
# ═══════════════════════════════════════════════════════════════════════════

def _sahne(kaydir_x=0, kaydir_y=0, boyut=(480, 640)):
    """Dokulu sentetik sahne — optik akis icin kose noktasi gerekiyor."""
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, (boyut[0] + 200, boyut[1] + 200, 3),
                       dtype=np.uint8)
    y = 100 + kaydir_y
    x = 100 + kaydir_x
    return np.ascontiguousarray(img[y:y + boyut[0], x:x + boyut[1]])


def test_cmc_saf_otelemeyi_kestirir():
    """Kamera saga kayarsa CMC bunu otelemeden okumali."""
    from iha_tracking_system import CameraMotionCompensator
    cmc = CameraMotionCompensator(olcek=1.0)

    cmc.hesapla(_sahne(0, 0))
    M = cmc.hesapla(_sahne(12, 0))       # sahne 12 px kaydi

    tx = M[0, 2]
    # Isaret, goruntunun kaydigi yonun tersi; buyuklugu tutmali.
    assert abs(abs(tx) - 12) < 3.0, f"oteleme yanlis kestirildi: {tx}"


def test_cmc_hareketsiz_sahnede_birim_dondurur():
    from iha_tracking_system import CameraMotionCompensator
    cmc = CameraMotionCompensator(olcek=1.0)
    kare = _sahne()
    cmc.hesapla(kare)
    M = cmc.hesapla(kare.copy())
    assert abs(M[0, 2]) < 1.0 and abs(M[1, 2]) < 1.0


def test_cmc_izin_konumunu_duzeltir():
    """Kamera hareketi izin Kalman durumundan cikarilmali."""
    t = KalmanBoxTracker(np.array([100.0, 100.0, 140.0, 200.0]))
    onceki_x = t.kf.x[0].item()

    M = np.array([[1.0, 0.0, 25.0],
                  [0.0, 1.0, -10.0]])       # saga 25, yukari 10
    t.kamera_hareketi_uygula(M)

    assert abs(t.kf.x[0].item() - (onceki_x + 25)) < 1e-6
    assert abs(t.kf.x[1].item() - (150.0 - 10)) < 1e-6


def test_cmc_hizi_otelemeden_etkilenmez():
    """Oteleme bir KONUM farki; hiza eklenmemeli."""
    t = KalmanBoxTracker(np.array([100.0, 100.0, 140.0, 200.0]))
    t.kf.x[4] = 5.0
    t.kf.x[5] = -3.0

    M = np.array([[1.0, 0.0, 50.0],
                  [0.0, 1.0, 50.0]])        # saf oteleme
    t.kamera_hareketi_uygula(M)

    assert abs(t.kf.x[4].item() - 5.0) < 1e-6
    assert abs(t.kf.x[5].item() - (-3.0)) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# GORUNUM / YENIDEN TANIMA (Re-ID)
# ═══════════════════════════════════════════════════════════════════════════

def _renkli_kare(renk, kutu, boyut=(480, 640)):
    """Belirli bir kutuda duz renkli bir nesne olan kare."""
    img = np.full((boyut[0], boyut[1], 3), 30, dtype=np.uint8)
    x1, y1, x2, y2 = [int(v) for v in kutu[:4]]
    img[y1:y2, x1:x2] = renk
    return img


def test_gorunum_ayni_nesneyi_yakin_bulur():
    from iha_tracking_system import AppearanceModel
    g = AppearanceModel()
    kutu = [100, 100, 180, 260]

    a = g.cikar(_renkli_kare((30, 200, 40), kutu), np.array(kutu))
    b = g.cikar(_renkli_kare((32, 198, 44), kutu), np.array(kutu))

    assert AppearanceModel.mesafe(a, b) < 0.2


def test_gorunum_farkli_nesneyi_ayirir():
    from iha_tracking_system import AppearanceModel
    g = AppearanceModel()
    kutu = [100, 100, 180, 260]

    yesil = g.cikar(_renkli_kare((30, 200, 40), kutu), np.array(kutu))
    kirmizi = g.cikar(_renkli_kare((40, 30, 210), kutu), np.array(kutu))

    assert AppearanceModel.mesafe(yesil, kirmizi) > 0.5


def test_reid_uzun_kapanmadan_sonra_kimligi_geri_verir():
    """
    ASIL TEST: hedef uzun sure kayboluyor, iz siliniyor, sonra BASKA BIR
    YERDE geri geliyor. Hareket modeli bunu bulamaz — gorunum bulmali.
    """
    cfg = SystemConfig(sort_max_age=3, sort_min_hits=1,
                       reid_enabled=True, cmc_enabled=False,
                       reid_mesafe_esigi=0.35, reid_hafiza_karesi=90)
    trk = ByteTrackTracker(cfg)

    renk = (30, 200, 40)
    kutu = [100, 100, 180, 260]

    # 1) Hedef gorunuyor, iz kuruluyor
    for _ in range(6):
        izler = trk.update(dizi(kutu + [0.9]), kare=_renkli_kare(renk, kutu))
    kimlik = izler[0]["id"]

    # 2) Uzun kapanma — iz max_age'i asip kayip havuzuna dusuyor
    bos = np.full((480, 640, 3), 30, dtype=np.uint8)
    for _ in range(10):
        trk.update(dizi(), kare=bos)
    assert len(trk.trackers) == 0, "iz aktif listeden dusmeliydi"
    assert len(trk.kayip_izler) == 1, "iz kayip havuzunda bekliyor olmaliydi"

    # 3) Ayni nesne EKRANIN BASKA YERINDE geri geliyor
    yeni_kutu = [420, 180, 500, 340]
    izler = trk.update(dizi(yeni_kutu + [0.9]),
                       kare=_renkli_kare(renk, yeni_kutu))

    assert izler, "yeniden tanima sonrasi iz bildirilmeliydi"
    assert izler[0]["id"] == kimlik, (
        f"kimlik korunmaliydi: {izler[0]['id']} != {kimlik}")


def test_reid_farkli_nesneye_ayni_kimligi_vermez():
    """Yanlis pozitif olmamali: baska renkte bir nesne eski kimligi almamali."""
    cfg = SystemConfig(sort_max_age=3, sort_min_hits=1,
                       reid_enabled=True, cmc_enabled=False,
                       reid_mesafe_esigi=0.35)
    trk = ByteTrackTracker(cfg)

    kutu = [100, 100, 180, 260]
    for _ in range(6):
        izler = trk.update(dizi(kutu + [0.9]),
                           kare=_renkli_kare((30, 200, 40), kutu))
    kimlik = izler[0]["id"]

    bos = np.full((480, 640, 3), 30, dtype=np.uint8)
    for _ in range(10):
        trk.update(dizi(), kare=bos)

    # Farkli renkte nesne. Yeni iz dogdugu karede bildirilmiyor
    # (hit_streak >= min_hits kosulu), bu yuzden iki kare besleniyor.
    yeni_kutu = [420, 180, 500, 340]
    for _ in range(2):
        izler = trk.update(dizi(yeni_kutu + [0.9]),
                           kare=_renkli_kare((40, 30, 210), yeni_kutu))

    assert izler, "yeni iz bildirilmeliydi"
    assert izler[0]["id"] != kimlik, "farkli nesne eski kimligi aldi"
    assert len(trk.kayip_izler) == 1, "eski iz havuzda kalmaliydi"


def test_reid_kapaliyken_havuz_kullanilmaz():
    cfg = SystemConfig(sort_max_age=2, sort_min_hits=1, reid_enabled=False,
                       cmc_enabled=False)
    trk = ByteTrackTracker(cfg)
    kutu = [100, 100, 180, 260]
    for _ in range(4):
        trk.update(dizi(kutu + [0.9]), kare=_renkli_kare((30, 200, 40), kutu))
    bos = np.full((480, 640, 3), 30, dtype=np.uint8)
    for _ in range(6):
        trk.update(dizi(), kare=bos)
    assert len(trk.kayip_izler) == 0


def test_kare_verilmese_de_calisir():
    """Geriye donuk uyumluluk: kare gecilmezse CMC/Re-ID sessizce atlanir."""
    trk = ByteTrackTracker(SystemConfig(sort_min_hits=1))
    for i in range(10):
        izler = trk.update(dizi(kutu(100 + i * 5, 100)))
    assert izler and len(izler) == 1
