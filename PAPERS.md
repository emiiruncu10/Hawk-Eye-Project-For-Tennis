# 📚 Recommended Papers & Resources

Bu dosya, proje boyunca okunması gereken akademik makaleler ve kaynakları içermektedir.

---

## 🎯 Öncelik Sıralaması

### ⭐⭐⭐ Kritik (İlk Hafta Okunmalı)

#### 1. TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications
- **arXiv:** [1907.03698](https://arxiv.org/abs/1907.03698)
- **Yıl:** 2019
- **Yazar:** Huang et al.
- **Özet:** Tenis topu takibi için temel heatmap-based yaklaşım. VGG-16 encoder + DeconvNet decoder mimarisi.
- **Neden Önemli:** Projenin temel ball tracking yaklaşımı bu paper'a dayanıyor.
- **Anahtar Kavramlar:**
  - Gaussian heatmap generation
  - Consecutive frame input (temporal information)
  - Small object detection challenges

#### 2. TrackNetV4: Enhancing Fast Sports Object Tracking with Motion Attention Maps
- **arXiv:** [2409.14543](https://arxiv.org/abs/2409.14543)
- **Yıl:** 2024
- **Yazar:** Raj et al.
- **Özet:** TrackNet ailesine motion attention maps ekleyerek performansı artırıyor.
- **Neden Önemli:** En güncel TrackNet versiyonu, state-of-the-art performans.
- **Anahtar Kavramlar:**
  - Motion-aware fusion mechanism
  - Frame differencing maps
  - Motion prompt layer

---

### ⭐⭐ Önemli (İlk 2 Hafta İçinde)

#### 3. TrackNetV2: Efficient Shuttlecock Tracking Network
- **Konferans:** ICPAI 2020
- **Yazar:** Sun et al.
- **Özet:** TrackNet'in badminton için optimize edilmiş versiyonu, multi-frame processing.
- **Anahtar Kavramlar:**
  - Frame triplet input
  - Efficiency improvements

#### 4. TTNet: Real-time temporal and spatial video analysis of table tennis
- **Link:** OpenTTGames dataset ile birlikte
- **Özet:** Multi-task learning: events spotting + ball detection + segmentation. 120fps video işleme.
- **Neden Önemli:** Bounce detection ve event spotting için referans.
- **Anahtar Kavramlar:**
  - Multi-task architecture
  - Temporal and spatial analysis
  - Real-time processing (6ms/frame)

#### 5. Automated Tennis Player and Ball Tracking with Court Keypoints Detection
- **arXiv:** [2511.04126](https://arxiv.org/abs/2511.04126)
- **Yıl:** 2025
- **Özet:** End-to-end pipeline: YOLOv8 player detection + YOLOv5 ball + ResNet50 court keypoints.
- **Neden Önemli:** Tam pipeline implementasyonu, homography hesaplama.
- **Anahtar Kavramlar:**
  - Court keypoint detection
  - Homography transformation
  - Mini-court visualization

---

### ⭐ Faydalı (Gerektiğinde Okuma)

#### 6. Where Is The Ball: 3D Ball Trajectory Estimation From 2D Monocular Tracking
- **Konferans:** CVPR 2025 Workshop
- **Özet:** Monoküler kameradan 3D trajectory tahmini.
- **Neden Önemli:** Monoküler sistemin sınırlarını ve olasılıklarını anlamak için.

#### 7. A comprehensive review of ball detection techniques in sports
- **Dergi:** PeerJ Computer Science, 2025
- **Link:** [peerj.com/articles/cs-3079](https://peerj.com/articles/cs-3079.pdf)
- **Özet:** Spor videolarında top tespiti tekniklerinin kapsamlı incelemesi.
- **Neden Önemli:** Genel bakış ve method karşılaştırması.

#### 8. Accurate Tennis Court Line Detection on Amateur Recorded Matches
- **arXiv:** [2404.06977](https://arxiv.org/abs/2404.06977)
- **Özet:** Amatör kayıtlar için kort çizgi tespiti, shadow removal dahil.
- **Neden Önemli:** Amateur video'lardaki zorluklar için spesifik çözümler.

#### 9. Deep Learning-Based Algorithm for Recognizing Tennis Balls
- **Dergi:** Applied Sciences, 2022
- **Özet:** Mask R-CNN ile tenis topu tanıma.
- **Neden Önemli:** Alternatif detection yaklaşımı.

---

## 🔧 Teknik Referanslar

### Object Detection

| Model | Paper | Yıl | Use Case |
|-------|-------|-----|----------|
| YOLOv8 | Ultralytics (no paper) | 2023 | Fast detection |
| YOLOv5 | Ultralytics | 2020 | Ball detection |
| Faster R-CNN | [arxiv:1506.01497](https://arxiv.org/abs/1506.01497) | 2015 | Player detection |

### Semantic Segmentation / Keypoint Detection

| Model | Paper | Yıl | Use Case |
|-------|-------|-----|----------|
| ResNet50 | [arxiv:1512.03385](https://arxiv.org/abs/1512.03385) | 2015 | Backbone |
| U-Net | [arxiv:1505.04597](https://arxiv.org/abs/1505.04597) | 2015 | Heatmap generation |
| DeconvNet | ICCV 2015 | 2015 | TrackNet decoder |

### Homography & Geometric Vision

| Topic | Resource | Notes |
|-------|----------|-------|
| Homography estimation | Multiple View Geometry (Hartley & Zisserman) | Book, Chapter 4 |
| DLT Algorithm | [OpenCV Documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) | findHomography |
| RANSAC | Fischler & Bolles, 1981 | Robust estimation |

---

## 📂 GitHub Repositories

### Aktif Kullanılacak

1. **ArtLabss/tennis-tracking**
   - https://github.com/ArtLabss/tennis-tracking
   - Open-source Monocular Python HawkEye
   - ⭐ İlk PoC için kullan

2. **yastrebksv/TennisProject**
   - https://github.com/yastrebksv/TennisProject
   - Ball tracking + court detection + bounce detection
   - ⭐ Pipeline referansı

3. **yastrebksv/TrackNet**
   - https://github.com/yastrebksv/TrackNet
   - Unofficial PyTorch TrackNet implementation
   - ⭐ Ball tracking modeli

4. **yastrebksv/TennisCourtDetector**
   - https://github.com/yastrebksv/TennisCourtDetector
   - Court keypoint detection neural network
   - ⭐ Court detection modeli

### Referans

5. **AggieSportsAnalytics/CourtCheck**
   - https://github.com/AggieSportsAnalytics/CourtCheck
   - Detectron2 based court detection

6. **TrackNetV4 Official**
   - https://time.griffith.edu.au/paper-sites/tracknet-v4/
   - Code and pretrained models

---

## 📊 Datasets

| Dataset | Size | Purpose | Link |
|---------|------|---------|------|
| TrackNet Tennis | ~20K frames | Ball tracking | [Google Drive](https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut) |
| Court Keypoints | 8,841 images | Court detection | [Google Drive](https://drive.google.com/file/d/1lhAaeQCmk2y440PmagA0KmIVBIysVMwu/view) |
| OpenTTGames | Table tennis | Multi-task training | Check TTNet paper |
| COCO | General | Transfer learning | [cocodataset.org](https://cocodataset.org) |

---

## 📖 Okuma Planı

### Hafta 1
- [ ] TrackNet (1907.03698) - Tam okuma
- [ ] TrackNetV4 (2409.14543) - Tam okuma
- [ ] YOLOv8 documentation - Hızlı geçiş

### Hafta 2
- [ ] TTNet - Bounce detection kısmı
- [ ] Automated Tennis Player and Ball Tracking - Pipeline kısmı
- [ ] Court detection repositories - Kod inceleme

### Hafta 3+
- [ ] Monocular 3D reconstruction papers
- [ ] Domain adaptation techniques
- [ ] TensorRT optimization guides
