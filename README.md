# Tokopedia Review Sentiment - Workflow CI

Repositori ini berisi alur otomatisasi MLOps (Continuous Integration Pipeline) menggunakan GitHub Actions dan MLflow Project. Sistem ini dirancang untuk melakukan pelatihan ulang model (*model retraining*) secara otomatis ketika terdapat perubahan kode atau data pada cabang `main`, melacak metrik ke DagsHub, serta mengemas model menjadi Docker Image secara lokal untuk dipublikasikan ke Docker Hub.

## Struktur Repositori
```text
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci_pipeline.yml    # Konfigurasi GitHub Actions Workflow
├── MLProject/
│   ├── tokopedia_preprocessing/
│   │   ├── train_clean.csv    # Dataset Pelatihan
│   │   └── test_clean.csv     # Dataset Pengujian
│   ├── conda.yaml             # Definisikan Environment Dependensi
│   ├── MLProject              # Manifes Cetak Biru MLflow Project
│   └── modelling.py           # Skrip Utama Retraining Model
└── README.md                  # Dokumentasi Proyek