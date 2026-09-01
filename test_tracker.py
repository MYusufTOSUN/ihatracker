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
