# SPK Penilaian Kinerja Destinasi Wisata Kerajinan

Sistem Pendukung Keputusan (SPK) untuk menilai kinerja destinasi wisata kerajinan
(Handicraft Center) menggunakan metode **Fuzzy Logic Mamdani**.

Aplikasi dikembangkan menggunakan **Python**, **Streamlit**, dan **scikit-fuzzy**
dengan studi kasus Rural Heritage Tourism Industry Chain Dataset.

---

## 📌 Tentang Project

Project ini merupakan implementasi Sistem Cerdas Pendukung Keputusan untuk
melakukan evaluasi kinerja destinasi wisata kerajinan secara komprehensif.

Dataset yang digunakan merupakan **Rural Heritage Tourism Industry Chain Dataset**
yang terdiri dari 6.000 record. Data kemudian difilter berdasarkan:

`Heritage_Type = Handicraft Center`

sehingga diperoleh **784 data destinasi** yang digunakan dalam proses evaluasi.

Sistem menggunakan lima variabel input:

| Kode | Variabel | Tipe |
|------|----------|------|
| C1 | Visitor Count | Benefit |
| C2 | Ticket Price | Cost |
| C3 | Tourist Satisfaction | Benefit |
| C4 | Revenue Generated | Benefit |
| C5 | Operational Cost | Cost |

Output sistem berupa **skor kinerja destinasi** yang dikategorikan menjadi:

- 🔴 Rendah
- 🟡 Sedang
- 🟢 Tinggi

---

## 🎯 Tujuan

Project ini bertujuan untuk:

1. Membangun sistem penilaian kinerja destinasi wisata kerajinan berbasis Fuzzy
   Logic Mamdani.
2. Mengimplementasikan proses fuzzifikasi, evaluasi rule, implication,
   agregasi, dan defuzzifikasi.
3. Melakukan evaluasi terhadap 784 destinasi Handicraft Center secara otomatis.
4. Menghasilkan skor dan peringkat kinerja destinasi.
5. Menyediakan analisis sensitivitas untuk mengetahui variabel yang paling
   berpengaruh terhadap perubahan skor kinerja.
6. Menyediakan sistem fuzzy yang dapat dikonfigurasi melalui penambahan variabel
   dan rule secara dinamis.

---

## 🧠 Metode

Metode utama yang digunakan adalah **Fuzzy Logic Mamdani**.

Alur proses inferensi:

```text
Input Data
    ↓
Fuzzifikasi
    ↓
Evaluasi Rule
    ↓
Implication / Clipping
    ↓
Agregasi
    ↓
Defuzzifikasi
    ↓
Skor Kinerja
    ↓
Kategori Kinerja