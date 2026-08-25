#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEKNOFEST SAVASAN IHA - KURULUM TEST SCRIPTI
YOLOv11 + TensorRT + SORT + PID Sistemi
"""

import sys
import os
import io

# Windows konsolunda UTF-8 encoding sorunu cozumu
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass


def test_python_version():
    """Python versiyonunu kontrol et"""
    print("=" * 60)
    print("0. PYTHON VERSIYONU")
    print("=" * 60)
    
    version = sys.version_info
    print(f"  Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("  [OK] Python 3.8+ gereksinimi karsilaniyor")
        return True
    else:
        print("  [X] Python 3.8 veya uzeri gerekli!")
        return False


def test_core_packages():
    """Temel paketleri test et"""
    print("\n" + "=" * 60)
    print("1. TEMEL PAKET KONTROLU")
    print("=" * 60)
    
    packages = {
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'scipy': 'SciPy',
        'yaml': 'PyYAML',
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'kurulu')
            print(f"  [OK] {name:20s} v{version}")
        except ImportError:
            print(f"  [X]  {name:20s} KURULU DEGIL!")
            all_ok = False
    
    return all_ok


def test_pytorch():
    """PyTorch ve CUDA'yi test et"""
    print("\n" + "=" * 60)
    print("2. PYTORCH & CUDA KONTROLU")
    print("=" * 60)
    
    try:
        import torch
        print(f"  [OK] PyTorch              v{torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            print(f"  [OK] CUDA                 v{cuda_version}")
            print(f"  [OK] GPU                  {device_name}")
        else:
            print("  [!]  CUDA                 Kullanilamiyor (CPU mode)")
        
        import torchvision
        print(f"  [OK] TorchVision          v{torchvision.__version__}")
        
        return True
    except ImportError as e:
        print(f"  [X]  PyTorch KURULU DEGIL: {e}")
        return False


def test_yolo():
    """YOLO (Ultralytics) test et"""
    print("\n" + "=" * 60)
    print("3. YOLO (ULTRALYTICS) KONTROLU")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        import ultralytics
        version = getattr(ultralytics, '__version__', 'kurulu')
        print(f"  [OK] Ultralytics          v{version}")
        return True
    except ImportError as e:
        print(f"  [X]  Ultralytics KURULU DEGIL: {e}")
        return False


def test_tracking_packages():
    """Takip sistemi paketlerini test et"""
    print("\n" + "=" * 60)
    print("4. TAKIP SISTEMI PAKETLERI")
    print("=" * 60)
    
    all_ok = True
    
    # FilterPy (Kalman Filter) - Opsiyonel, kendi implementasyonumuz var
    try:
        import filterpy
        from filterpy.kalman import KalmanFilter
        version = getattr(filterpy, '__version__', 'kurulu')
        print(f"  [OK] FilterPy             v{version}")
        print(f"       -> KalmanFilter      [OK]")
    except ImportError:
        print(f"  [!]  FilterPy             Kurulu degil (opsiyonel - kendi impl. var)")
    
    # SciPy (Hungarian Algorithm)
    try:
        from scipy.optimize import linear_sum_assignment
        import scipy
        print(f"  [OK] SciPy                v{scipy.__version__}")
        print(f"       -> linear_sum_assignment [OK]")
    except ImportError as e:
        print(f"  [X]  SciPy KURULU DEGIL: {e}")
        all_ok = False
    
    return all_ok


def test_serial():
    """Seri haberlesme paketini test et"""
    print("\n" + "=" * 60)
    print("5. SERI HABERLESME (OPSIYONEL)")
    print("=" * 60)
    
    try:
        import serial
        version = getattr(serial, '__version__', 'kurulu')
        print(f"  [OK] PySerial             v{version}")
        return True
    except ImportError:
        print(f"  [!]  PySerial             Kurulu degil (opsiyonel)")
        return True  # Opsiyonel oldugu icin True


def test_model_files():
    """Model dosyalarini kontrol et"""
    print("\n" + "=" * 60)
    print("6. MODEL DOSYALARI KONTROLU")
    print("=" * 60)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    files = {
        'YOLO Engine (TensorRT)': os.path.join(current_dir, 'yolo', 'best.engine'),
        'YOLO Engine FP16': os.path.join(current_dir, 'yolo', 'best_fp16_640.engine'),
        'YOLO PyTorch': os.path.join(current_dir, 'yolo', 'best.pt'),
        'YOLO ONNX': os.path.join(current_dir, 'yolo', 'best.onnx'),
    }
    
    yolo_found = False
    
    for name, path in files.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            size_mb = size / (1024 * 1024)
            print(f"  [OK] {name:25s} ({size_mb:.1f} MB)")
            yolo_found = True
        else:
            print(f"  [ ]  {name:25s} (bulunamadi)")
    
    if not yolo_found:
        print("\n  [!] UYARI: Hicbir YOLO model dosyasi bulunamadi!")
        print("      yolo/ klasorune best.engine veya best.pt koyun.")
        return False
    
    return True


def test_scripts():
    """Ana script dosyalarini kontrol et"""
    print("\n" + "=" * 60)
    print("7. SCRIPT DOSYALARI KONTROLU")
    print("=" * 60)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    scripts = {
        'Ana Sistem (YOLO+SORT+PID)': 'iha_tracking_system.py',
        'ByteTrack Tracker': 'bytetrack_multi_tracker.py',
        'Jetson Optimized': 'bytetrack_jetson_optimized.py',
    }
    
    all_ok = True
    for name, filename in scripts.items():
        path = os.path.join(current_dir, filename)
        if os.path.exists(path):
            print(f"  [OK] {name:30s} ({filename})")
        else:
            print(f"  [X]  {name:30s} BULUNAMADI!")
            all_ok = False
    
    return all_ok


def test_camera():
    """Kamera erisimini test et"""
    print("\n" + "=" * 60)
    print("8. KAMERA KONTROLU")
    print("=" * 60)
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"  [OK] Kamera erisilebilir")
                print(f"       -> Cozunurluk: {w}x{h}")
                cap.release()
                return True
            else:
                print("  [X]  Kamera acildi ama frame okunamadi")
                cap.release()
                return False
        else:
            print("  [!]  Kamera acilamadi")
            print("       -> Kameranin bagli oldugundan emin olun")
            print("       -> Baska uygulama kamerayi kullaniyor olabilir")
            return True  # Kritik degil
    except Exception as e:
        print(f"  [!]  Kamera test hatasi: {e}")
        return True  # Kritik degil


def test_tensorrt():
    """TensorRT kontrolu (opsiyonel)"""
    print("\n" + "=" * 60)
    print("9. TENSORRT KONTROLU (OPSIYONEL)")
    print("=" * 60)
    
    try:
        import tensorrt as trt
        print(f"  [OK] TensorRT             v{trt.__version__}")
        return True
    except ImportError:
        print("  [!]  TensorRT             Kurulu degil (opsiyonel)")
        print("       -> .engine dosyalari hala ultralytics ile calisabilir")
        return True  # Opsiyonel


def main():
    print("\n" + "=" * 60)
    print("  TEKNOFEST SAVASAN IHA - KURULUM TESTI")
    print("  YOLOv11 + TensorRT + SORT + PID Sistemi")
    print("=" * 60)
    
    results = []
    
    # Testleri calistir
    results.append(("Python Versiyonu", test_python_version()))
    results.append(("Temel Paketler", test_core_packages()))
    results.append(("PyTorch & CUDA", test_pytorch()))
    results.append(("YOLO (Ultralytics)", test_yolo()))
    results.append(("Takip Paketleri", test_tracking_packages()))
    results.append(("Seri Haberlesme", test_serial()))
    results.append(("Model Dosyalari", test_model_files()))
    results.append(("Script Dosyalari", test_scripts()))
    results.append(("Kamera", test_camera()))
    results.append(("TensorRT", test_tensorrt()))
    
    # Sonuclar
    print("\n" + "=" * 60)
    print("  TEST SONUCLARI")
    print("=" * 60)
    
    all_passed = True
    critical_failed = False
    
    for test_name, passed in results:
        status = "[OK] BASARILI" if passed else "[X] BASARISIZ"
        print(f"  {test_name:25s}: {status}")
        if not passed:
            all_passed = False
            if test_name in ["Python Versiyonu", "Temel Paketler", "PyTorch & CUDA", "YOLO (Ultralytics)", "Takip Paketleri"]:
                critical_failed = True
    
    print("\n" + "=" * 60)
    if all_passed:
        print("  [OK] TUM TESTLER BASARILI!")
        print("")
        print("  Sistemi baslatmak icin:")
        print("    python iha_tracking_system.py")
    elif not critical_failed:
        print("  [!] BAZI OPSIYONEL TESTLER BASARISIZ")
        print("  Sistem yine de calisabilir.")
        print("")
        print("  Denemek icin:")
        print("    python iha_tracking_system.py")
    else:
        print("  [X] KRITIK TESTLER BASARISIZ!")
        print("  Lutfen eksik paketleri kurun:")
        print("    pip install -r requirements.txt")
    print("=" * 60 + "\n")
    
    return 0 if all_passed or not critical_failed else 1


if __name__ == "__main__":
    sys.exit(main())
