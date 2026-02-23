# 🎾 Amateur Tennis Hawk-Eye System

> **Monocular camera-based tennis ball tracking and bounce detection system for amateur courts**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: In Development](https://img.shields.io/badge/status-in%20development-orange.svg)]()

---

## 📋 Project Overview

Bu proje, amatör tenis kortları için düşük maliyetli bir **"Hawk-Eye benzeri"** sistem geliştirmeyi hedeflemektedir. Profesyonel Hawk-Eye sistemleri 10-18 yüksek hızlı kamera ve ~€100,000 maliyet gerektirirken, bu proje **tek bir kamera (smartphone/webcam)** ile çalışan, yapay zeka tabanlı bir çözüm sunmayı amaçlamaktadır.

### 🎯 Hedefler

- **Ball Tracking:** Tenis topunun video üzerinde gerçek zamanlı takibi
- **Court Detection:** Kort çizgilerinin ve keypoint'lerin tespiti
- **Bounce Detection:** Topun kort yüzeyine değdiği anların belirlenmesi
- **In/Out Decision:** Topun kort içinde mi dışında mı olduğunun karar verilmesi
- **Heat Map Visualization:** Topun kort üzerindeki landing pozisyonlarının görselleştirilmesi

### ⚠️ Kapsam Sınırlamaları

| Yapılabilir | Yapılamaz (Monoküler Sistemle) |
|-------------|-------------------------------|
| 2D ball tracking | Gerçek 3D pozisyon (depth yok) |
| Court homography ile 2D projeksiyon | Real-time <100ms Hawk-Eye kalitesi |
| In/Out kararı (~±5-10cm) | Spin analizi |
| Offline video analizi | Multi-camera triangulation |
| Heat map oluşturma | - |

---

## 🗓️ Haftalık İş Planı

### 📅 Hafta 1: Araştırma & Ortam Kurulumu

#### Günler 1-2: Literatür Taraması
- [ ] TrackNet paper'ını detaylı oku (arxiv:1907.03698)
- [ ] TrackNetV4 paper'ını incele (arxiv:2409.14543)
- [ ] Court keypoint detection yaklaşımlarını araştır
- [ ] Bounce detection methodlarını incele

#### Günler 3-4: Ortam Kurulumu
- [ ] Python environment oluştur (conda/venv)
- [ ] PyTorch + CUDA kurulumu
- [ ] Gerekli kütüphaneleri yükle
- [ ] Google Colab notebook hazırla (GPU erişimi için)

#### Günler 5-7: Mevcut Projeleri Test Et
- [ ] [ArtLabss/tennis-tracking](https://github.com/ArtLabss/tennis-tracking) clone & run
- [ ] [yastrebksv/TennisProject](https://github.com/yastrebksv/TennisProject) test
- [ ] Pretrained modelleri kendi videolarında dene
- [ ] Domain gap analizi yap (broadcast vs amateur video)

### 📅 Hafta 2: Dataset Hazırlığı & Model Seçimi

#### Günler 1-3: Dataset İndirme & Analiz
- [ ] TrackNet dataset indir (~20K frame)
- [ ] Court Keypoint dataset indir (8,841 image)
- [ ] Dataset yapısını incele ve anla
- [ ] Annotation formatını öğren

#### Günler 4-5: Kendi Veri Toplama Planı
- [ ] Kamera açısı ve yerleşim stratejisi belirle
- [ ] Frame rate ve çözünürlük kararı (60fps önerilir)
- [ ] Test çekimleri yap
- [ ] Annotation tool seç (CVAT, LabelImg, veya Roboflow)

#### Günler 6-7: Model Karşılaştırması
- [ ] TrackNet vs YOLOv8 benchmark
- [ ] Inference hızı ölçümü
- [ ] Accuracy karşılaştırması
- [ ] Final model seçimi

---

## 🏗️ Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE OVERVIEW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Video Input] ──▶ [Frame Extraction]                          │
│        │                   │                                    │
│        ▼                   ▼                                    │
│  ┌─────────────┐    ┌─────────────────┐    ┌────────────────┐  │
│  │ Court       │    │ Ball Tracking   │    │ Player         │  │
│  │ Keypoint    │    │ (TrackNetV4)    │    │ Detection      │  │
│  │ Detection   │    │                 │    │ (YOLOv8)       │  │
│  │ (ResNet50)  │    │                 │    │                │  │
│  └──────┬──────┘    └────────┬────────┘    └───────┬────────┘  │
│         │                    │                      │           │
│         ▼                    ▼                      │           │
│  ┌─────────────┐    ┌─────────────────┐            │           │
│  │ Homography  │◀───│ 2D Ball         │            │           │
│  │ Matrix H    │    │ Coordinates     │            │           │
│  └──────┬──────┘    └────────┬────────┘            │           │
│         │                    │                      │           │
│         ▼                    ▼                      │           │
│  ┌─────────────────────────────────────┐           │           │
│  │     Bounce Detection                │           │           │
│  │     (CatBoostRegressor /            │           │           │
│  │      Trajectory Analysis)           │           │           │
│  └──────────────┬──────────────────────┘           │           │
│                 │                                   │           │
│                 ▼                                   │           │
│  ┌─────────────────────────────────────────────────┴──────┐    │
│  │              Court Projection (Mini-Map)               │    │
│  │         Top-down view of ball landing positions        │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Kullanılacak Modeller

### 1. Ball Tracking: TrackNetV4 (Önerilen)

**Neden TrackNetV4?**
- Heatmap-based yaklaşım küçük ve bulanık topları tespit etmede YOLO'dan üstün
- Motion attention maps ile hareket bölgeleri vurgulanıyor
- Ardışık frame'lerden trajectory pattern öğreniyor
- 640×360 input, 99.7% precision, 97.3% recall

**Alternatif:** YOLOv8 (daha hızlı inference, daha az doğru)

### 2. Court Detection: ResNet50-based Keypoint CNN

**Özellikler:**
- 14 keypoint tespiti
- Homography matrix hesaplama için gerekli
- Tüm kort tipleri destekleniyor (hard, clay, grass)

### 3. Bounce Detection: CatBoostRegressor

**Input Features:**
- x, y koordinatları
- Velocity (V2-V1/t2-t1)
- Trajectory pattern

---

## 📁 Dataset'ler

| Amaç | Dataset | Boyut | Link |
|------|---------|-------|------|
| Ball Tracking | TrackNet Dataset | ~20K frame | [Google Drive](https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut) |
| Court Detection | Court Keypoint Dataset | 8,841 image | [Google Drive](https://drive.google.com/file/d/1lhAaeQCmk2y440PmagA0KmIVBIysVMwu/view) |
| Bounce Detection | Pretrained CatBoost | — | [Google Drive](https://drive.google.com/file/d/1Eo5HDnAQE8y_FbOftKZ8pjiojwuy2BmJ/view) |

---

## 📚 Önerilen Paper'lar

### Temel Okumalar (Öncelikli)

1. **TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects**
   - arXiv: [1907.03698](https://arxiv.org/abs/1907.03698)
   - Temel ball tracking yaklaşımı, heatmap-based detection

2. **TrackNetV4: Enhancing Fast Sports Object Tracking with Motion Attention Maps**
   - arXiv: [2409.14543](https://arxiv.org/abs/2409.14543)
   - Motion-aware fusion, en güncel TrackNet versiyonu

3. **TTNet: Real-time temporal and spatial video analysis of table tennis**
   - Multi-task learning: events spotting + ball detection + segmentation
   - 120fps video işleme, 97% accuracy

### İleri Okumalar

4. **Where Is The Ball: 3D Ball Trajectory Estimation From 2D Monocular Tracking**
   - CVPR 2025 Workshop
   - Monoküler kameradan 3D trajectory tahmini

5. **Automated Tennis Player and Ball Tracking with Court Keypoints Detection**
   - arXiv: [2511.04126](https://arxiv.org/abs/2511.04126)
   - End-to-end pipeline, ResNet50 court detection

6. **A comprehensive review of ball detection techniques in sports**
   - PeerJ Computer Science, 2025
   - Genel bakış ve karşılaştırma

---

## 🛠️ Kurulum (Yakında)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/amateur-tennis-hawkeye.git
cd amateur-tennis-hawkeye

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download pretrained models
python scripts/download_models.py
```

---

## 📂 Proje Yapısı (Planlanan)

```
amateur-tennis-hawkeye/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── data/
│   ├── raw/                    # Ham videolar
│   ├── processed/              # İşlenmiş frame'ler
│   └── annotations/            # Label dosyaları
├── models/
│   ├── ball_tracking/          # TrackNet/YOLO modelleri
│   ├── court_detection/        # Court keypoint modeli
│   └── bounce_detection/       # CatBoost modeli
├── src/
│   ├── preprocessing/          # Video işleme
│   ├── tracking/               # Ball tracking modülü
│   ├── court/                  # Court detection & homography
│   ├── bounce/                 # Bounce detection
│   └── visualization/          # Mini-map ve görselleştirme
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_pipeline_test.ipynb
├── scripts/
│   ├── download_models.py
│   ├── train.py
│   └── inference.py
└── tests/
    └── ...
```

---

## 📹 Donanım Önerileri

### Minimum Gereksinimler

| Bileşen | Önerilen | Notlar |
|---------|----------|--------|
| Kamera | iPhone 11+ / Pixel 4+ / GoPro | 60fps minimum |
| Frame Rate | 60 fps | 30fps'de bounce tespiti zorlaşır |
| Çözünürlük | 1080p | 720p kabul edilebilir |
| Tripod | Sabit (kritik!) | Kamera hareketi homography'i bozar |
| GPU | RTX 3060+ / Colab Pro | Training için gerekli |

### Frame Rate Analizi

| FPS | Ball Movement/Frame (140 km/h) | Kalite |
|-----|-------------------------------|--------|
| 30 | ~130cm | Zor |
| 60 | ~65cm | Kabul edilebilir |
| 120 | ~32cm | İdeal |

---

## 🤝 Katkıda Bulunma

Bu proje aktif geliştirme aşamasındadır. Katkıda bulunmak için:

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'feat: add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

---

## 📄 Lisans

MIT License - Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 📬 İletişim

- **GitHub Issues:** Bug report ve feature request için
- **Discussions:** Genel sorular ve tartışmalar için

---

## 🙏 Teşekkürler

Bu proje aşağıdaki açık kaynak projelerden ilham almıştır:

- [TrackNet](https://nol.cs.nctu.edu.tw/ndo3je6av9/) - Ball tracking
- [ArtLabss/tennis-tracking](https://github.com/ArtLabss/tennis-tracking) - Open-source Hawk-Eye
- [yastrebksv/TennisProject](https://github.com/yastrebksv/TennisProject) - Tennis analysis pipeline
