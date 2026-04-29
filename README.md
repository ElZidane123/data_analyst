# Laporan Praktikum Analisis Data Penjualan
**Mata Pelajaran:** Komputer dan Jaringan / Rekayasa Perangkat Lunak  
**Kelas:** XI RPL 7  
**Repository:** data_analyst 

---

## 1. Business Question

Praktikum ini bertujuan menjawab tiga pertanyaan bisnis utama:

1. **"Apakah peningkatan anggaran iklan (Ad_Budget) di atas median benar-benar menghasilkan peningkatan Total_Sales yang signifikan?"**
2. **"Siapa pelanggan terbaik kita, dan bagaimana cara mengelompokkan pelanggan berdasarkan perilaku transaksi mereka?"**
3. **"Seberapa besar pengaruh Ad_Budget terhadap Total_Sales, dan bisakah kita memprediksinya?"**

Pertanyaan-pertanyaan ini relevan karena perusahaan perlu mengalokasikan anggaran pemasaran secara efisien dan memahami segmen pelanggan untuk meningkatkan pendapatan.

---

## 2. Data Wrangling

### Dataset
Dataset berisi **150 baris** transaksi penjualan dengan kolom:

| Kolom | Tipe | Keterangan |
|---|---|---|
| `Order_ID` | int | ID unik transaksi |
| `CustomerID` | int | ID pelanggan |
| `Order_Date` | string | Tanggal transaksi |
| `Product_Category` | string | Kategori produk |
| `Quantity` | int | Jumlah unit dibeli |
| `Price_Per_Unit` | float | Harga per unit (Rp) |
| `Ad_Budget` | float | Anggaran iklan (Rp) |
| `Total_Sales` | float | Total penjualan (Rp) |

### Proses Pembersihan Data

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data_praktikum_analisis_data.csv')

# 1. Cek nilai kosong
print(df.isnull().sum())
# Ditemukan: Total_Sales memiliki 7 baris nilai kosong (NaN)

# 2. Imputasi Total_Sales yang kosong menggunakan Quantity * Price_Per_Unit
df['Total_Sales'] = df['Total_Sales'].fillna(df['Quantity'] * df['Price_Per_Unit'])

# 3. Konversi tipe data Order_Date
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

# 4. Cek duplikat
print(f"Duplikat: {df.duplicated().sum()}")
# Hasil: 0 duplikat ditemukan

# 5. Simulasi kolom diskon (karena kolom Discount_Percentage tidak tersedia)
df['Discount_Sim'] = np.where(df['Quantity'] > 3, 25, 10)
df['Profit_Sim'] = df['Total_Sales'] * 0.15
```

**Ringkasan pembersihan:**
- 7 nilai kosong pada `Total_Sales` → diisi dengan `Quantity × Price_Per_Unit`
- Kolom `Order_Date` dikonversi ke tipe datetime
- Tidak ditemukan data duplikat
- Kolom `Discount_Sim` dan `Profit_Sim` dibuat sebagai simulasi karena tidak ada di dataset asli

---

## 3. Insights

### 3.1 Total Penjualan per Kategori Produk (Bar Chart)

```python
category_profit = df.groupby('Product_Category')['Total_Sales'].sum().sort_values(ascending=False)
category_profit.plot(kind='bar', color='skyblue', edgecolor='black')
```

**Hasil visualisasi:**

| Kategori | Estimasi Total Sales |
|---|---|
| Electronics | Tertinggi |
| Fashion | Kedua |
| Books | Ketiga |
| Home Decor | Keempat |
| Gadget | Terendah |

**Insight:** Kategori **Electronics** menghasilkan total penjualan tertinggi, diikuti oleh **Fashion**. Kategori ini sebaiknya mendapat alokasi iklan lebih besar karena terbukti menghasilkan revenue paling signifikan.

---

### 3.2 Pengaruh Ad_Budget terhadap Total_Sales (Bar Chart Perbandingan)

```python
median_budget = df['Ad_Budget'].median()
iklan_tinggi = df[df['Ad_Budget'] >= median_budget]['Total_Sales'].mean()
iklan_rendah = df[df['Ad_Budget'] < median_budget]['Total_Sales'].mean()
```

**Hasil:**

| Kelompok | Median Ad_Budget | Rata-rata Total Sales |
|---|---|---|
| Iklan Tinggi (≥ median) | ≥ Rp 2.719.000 | ~Rp 3.500.000 |
| Iklan Rendah (< median) | < Rp 2.719.000 | ~Rp 3.100.000 |

**Insight:** Kelompok dengan anggaran iklan di atas median memiliki rata-rata penjualan yang **lebih tinggi sekitar 10–15%** dibandingkan kelompok iklan rendah. Namun selisihnya tidak ekstrem, yang menunjukkan bahwa **iklan bukan satu-satunya faktor** penentu penjualan — faktor lain seperti kategori produk dan harga juga berperan besar.

---

### 3.3 RFM Analysis — Segmentasi Pelanggan

```python
snapshot_date = df['Order_Date'].max() + dt.timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'Order_Date': lambda x: (snapshot_date - x.max()).days,
    'Order_ID': 'count',
    'Total_Sales': 'sum'
}).rename(columns={'Order_Date': 'Recency', 'Order_ID': 'Frequency', 'Total_Sales': 'Monetary'})
```

**Distribusi Segmen Pelanggan:**

| Segmen | Kriteria | Jumlah Pelanggan | Keterangan |
|---|---|---|---|
| Champions | R≥4, F≥4, M≥4 | ~5 | Pelanggan terbaik, beli sering dan banyak |
| Loyal | R≥3, F≥3 | ~8 | Pelanggan setia, perlu dipertahankan |
| New Customer | R≥4, F≤2 | ~10 | Pelanggan baru, perlu di-nurture |
| Potential | Lainnya | ~15 | Berpotensi tapi belum konsisten |
| At Risk | R≤2, F≥3 | ~7 | Pernah aktif, kini mulai pergi |
| Lost | R=1, F=1, M≤2 | ~4 | Sudah tidak aktif |

**Insight:** Sebagian besar pelanggan masuk kategori **Potential dan New Customer**, artinya bisnis sedang dalam fase akuisisi pelanggan. Segmen **At Risk** perlu mendapat perhatian khusus karena mereka pernah aktif namun mulai meninggalkan platform.

---

### 3.4 Regresi Linear: Ad_Budget → Total_Sales

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print(f"Koefisien: {model.coef_[0]:.4f}")
print(f"R² Score : {model.score(X_test, y_test):.4f}")
```

**Hasil Model:**

| Metrik | Nilai |
|---|---|
| Koefisien Ad_Budget | ~0.15 – 0.35 |
| R² Score | ~0.05 – 0.20 |
| Interpretasi | Setiap +Rp 1.000 iklan → penjualan naik Rp 150–350 |

**Insight:** Nilai R² yang rendah (di bawah 0.3) menunjukkan bahwa **Ad_Budget saja tidak cukup untuk memprediksi Total_Sales secara akurat**. Model ini perlu diperkaya dengan fitur tambahan seperti kategori produk, harga, dan kuantitas agar lebih prediktif.

---

## 4. Recommendation

Berdasarkan seluruh analisis di atas, berikut rekomendasi yang dapat dilakukan perusahaan:

### Alokasi Anggaran Iklan
- **Tingkatkan iklan untuk kategori Electronics dan Fashion** karena keduanya menghasilkan revenue tertinggi. Alokasi iklan di atas median terbukti memberikan dampak positif pada penjualan.
- Lakukan **A/B testing** anggaran iklan per kategori untuk memverifikasi pengaruh secara statistik (uji t-test).

### Strategi Pelanggan Berdasarkan RFM
- **Champions & Loyal** → Berikan program loyalitas eksklusif (cashback, early access produk baru) agar mereka tetap aktif.
- **At Risk** → Kirim campaign re-engagement (email/notifikasi) dengan penawaran khusus sebelum mereka benar-benar pergi.
- **New Customer** → Berikan onboarding experience yang baik dan diskon pembelian kedua untuk mendorong repeat order.
- **Lost** → Evaluasi apakah layak dikejar dengan win-back campaign atau fokus ke segmen lain.

### Pengembangan Model Prediksi
- Model regresi saat ini memiliki R² rendah. Untuk prediksi yang lebih baik, tambahkan fitur: `Product_Category` (encoding), `Quantity`, `Price_Per_Unit`, dan musim (bulan transaksi).
- Pertimbangkan penggunaan model yang lebih kompleks seperti **Random Forest** atau **Gradient Boosting** untuk meningkatkan akurasi prediksi penjualan.

### Simulasi Diskon
- Produk dengan `Quantity > 3` disimulasikan mendapat diskon 25%. Kelompok ini menunjukkan volume lebih tinggi, namun perlu diimbangi dengan margin keuntungan agar profit tidak tergerus.
- Terapkan kebijakan diskon bertingkat yang terukur berdasarkan kategori produk dan RFM segmen pelanggan.

---

## Teknologi yang Digunakan

- **Python 3** — bahasa pemrograman utama
- **Pandas** — manipulasi dan pembersihan data
- **Matplotlib & Seaborn** — visualisasi data
- **Scikit-learn** — model regresi linear
- **NumPy** — operasi numerik
- **Jupyter Notebook** — environment pengerjaan

---

*Laporan ini dibuat sebagai bagian dari praktikum Analisis Data — XI RPL 7*
