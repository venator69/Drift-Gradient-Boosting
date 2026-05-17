# Iter_2 — Analisis prediksi drift (concat) dengan XGBoost dan interpretasi SHAP

Repositori ini berisi **data gabungan** (`concat.csv`) dari beberapa run CSV perangkat/host, serta **notebook Jupyter** untuk regresi prediksi **besar drift translasi dalam jarak** (skalar), optimasi model, dan interpretasi fitur.

---

## 1. Tujuan proyek

- **Prediksi target:** besaran **drift translasi** sebagai **jarak skalar**  
  `drift_distance = sqrt(dx^2 + dy^2 + dz^2)`  
  (bukan prediksi per komponen sumbu secara terpisah untuk target utama analisis tabular).

- **Motivasi:** drift translasi mencerminkan **ketidakstabilan estimasi pose** relatif antar frame; memprediksinya dari sinyal telemetri/tracking membantu **monitoring kualitas** dan analisis penyebab error.

- **Deliverable utama:** notebook `analisis_prediksi_xgboost_concat.ipynb` (pipeline otomatis: preprocessing → baseline → tuning → optimasi tambahan → importance/SHAP → encoder LSTM/PatchTST + XGBoost → perbandingan).  
  Notebook lain: `analisis_tracking_concat.ipynb` (gabungan CSV sumber, ringkasan missing/tracking, imputasi ringkas).

---

## 2. Mengapa model-model ini dipilih?

| Pendekatan | Alasan pemilihan (rationale) | Sumber |
|------------|------------------------------|--------|
| **XGBoost Regressor** | Kuat pada data **tabular** dengan hubungan **nonlinear** dan interaksi antarfitur; stabil dan skalabel untuk regresi. | Chen & Guestrin (2016); dokumentasi [XGBoost](https://xgboost.readthedocs.io/) |
| **Hyperparameter tuning (RandomizedSearchCV, Optuna Random/TPE)** | Mengurangi under/overfitting dengan mencari `depth`, `learning_rate`, regularisasi, subsampling, dll. **TPE** memperlakukan optimasi sebagai optimasi bayesian sekuensial (surrogate). | Bergstra & Bengio (2012) untuk random search; [Optuna](https://optuna.org/) (Akiba et al., 2019); [scikit-learn](https://scikit-learn.org/stable/modules/grid_search.html) |
| **StandardScaler** | Menyamakan skala fitur numerik agar pohon tidak mendominasi fitur ber skala besar (meskipun XGBoost cukup robust, scaling tetap membantu perbandingan dan optimasi numerik). | [scikit-learn preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html) |
| **SHAP (TreeExplainer)** | Menjelaskan **kontribusi marginal** setiap fitur terhadap output model untuk satu prediksi atau agregat (global importance). | Lundberg & Lee (2017); [SHAP](https://shap.readthedocs.io/) |
| **LSTM + XGBoost (latent)** | **Urutan waktu** diekstrak menjadi representasi vektor per jendela; XGBoost memetakan latent → target. Memungkinkan model memakai **konteks temporal** tanpa hanya satu baris. | Hochreiter & Schmidhuber (1997) untuk LSTM; [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) |
| **PatchTST-style encoder + XGBoost** | **Patch + Transformer** pada deret waktu multivariat merangkap pola jangka panjang/patch; cocok untuk eksperimen paralel dengan LSTM. | Nie et al. (2023) *A Time Series is Worth 64 Words* ([arXiv:2211.14730](https://arxiv.org/abs/2211.14730)) |

**Catatan:** Di notebook, **SHAP untuk model tabular optimal** diarahkan ke **`tuned_xgb`** (hiperparameter terpilih), agar interpretasi fitur sejajar dengan fitur asli (bukan subset PCA/RFE kecuali Anda ubah secara eksplisit).

---

## 3. Langkah analisis (alur notebook prediksi)

1. **Muat `concat.csv`** — pemeriksaan duplikat, missing, ringkasan outlier IQR.
2. **Definisi target** — `__drift_distance__` = `sqrt(dx^2 + dy^2 + dz^2)`.
3. **Pemilihan fitur** — kolom berikut **dikecualikan** dari matriks fitur `X` (bukan target): timestamp/indeks waktu wall-clock, ordinal state, komponen translasi mentah dan `e_trans` (mitigasi **leakage** ke target jarak), serta komponen rotasi per sumbu (`drx`, `dry`, `drz`). Urutan temporal untuk model sekuensial memakai **`__row_order__`** (indeks baris asli CSV).
4. **Train/test split** acak (reproducible seed) + `StandardScaler` fit pada train.
5. **Baseline XGBoost** — metrik train/test.
6. **Tuning** — `RandomizedSearchCV` dengan distribusi `scipy.stats` (menghindari grid diskret raksasa), Optuna Random & TPE, pemilihan parameter terbaik + `tuned_xgb`.
7. **Optimasi tambahan** — SelectKBest, RFE, PCA, voting ensemble, early stopping (API kompatibel XGBoost lama/baru), mutual information selection, K-Fold ringkasan, perbandingan sebelum/sesudah.
8. **Importance & SHAP** — gain/weight/cover + `mean |SHAP|`; model SHAP = `tuned_xgb` pada `X_train_s`.
9. **Learning curve** — diagnosa bias/varians.
10. **LSTM latent + XGBoost** & **PatchTST latent + XGBoost** — jendela temporal, latent **tidak** mengurangi dimensi keluaran (dimensi latent = jumlah fitur input encoder), tuning XGBoost pada latent, SHAP pada ruang laten (diberi label sejajar nama fitur asli).
11. **Ringkasan** — tabel perbandingan model; opsional sel peringkat SHAP untuk model terbaik menurut tabel (jika Anda tambahkan).

---

## 4. Parameter performa: makna dan rationale

Rumus di bawah sengaja **di luar tabel** agar parser Markdown (GitHub, VS Code, dsb.) tidak memecah tabel karena simbol `|` di dalam notasi matematika.

**R² (koefisien determinasi)** — proporsi varians target yang dijelaskan relatif terhadap memprediksi konstanta rata-rata `y_bar`:

```text
R2 = 1 - sum((y - y_hat)^2) / sum((y - y_bar)^2)
```

**RMSE** — akar rata-rata kuadrat error; satuan sama dengan target (mis. jarak drift):

```text
RMSE = sqrt( (1/n) * sum((y - y_hat)^2) )
```

**MAE** — rata-rata nilai mutlak error (tanpa karakter `|` di dalam sel tabel):

```text
MAE = (1/n) * sum( abs(y - y_hat) )
```

| Metrik | Definisi (ringkas) | Rationale penggunaan | Sumber |
|--------|--------------------|------------------------|--------|
| **R²** | Seberapa baik model vs baseline rata-rata `y_bar` (lihat rumus di atas). | Skala intuitif; mudah dibanding antar-model pada **split yang sama**. | [scikit-learn `r2_score`](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score-the-coefficient-of-determination) |
| **RMSE** | Error dalam unit target; memberi bobot lebih ke error besar. | Sensitif ke **outlier**; cocok bila error besar sangat merugikan. | [`mean_squared_error`](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error) (RMSE = sqrt(MSE)) |
| **MAE** | Rata-rata besar error absolut. | Lebih **robust** ke outlier ekstrem vs RMSE; mudah dijelaskan ke non-statistik. | [`mean_absolute_error`](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error) |

**Mengapa sering melihat train vs test:** train tinggi + test rendah mengindikasikan **overfitting**; keduanya rendah mengindikasikan **underfitting** atau sinyal lemah. **Cross-validation** memberi estimasi performa lebih stabil daripada satu split (Lopez de Prado, 2018 mendiskusikan leakage temporal—relevan jika Anda beralih ke validasi kronologis murni).

---

## 5. Glosar kolom `concat.csv` dan hubungan plausibel dengan **error / drift translasi**

Interpretasi di bawah ini mengasumsikan **nama kolom** mengikuti konvensi umum **visual(-inertial) odometry / tracking**: pose berubah antar frame; `dx`, `dy`, `dz` adalah komponen translasi antar frame (dipakai hanya untuk **target jarak** di notebook prediksi, lalu **dihapus** dari `X` untuk mengurangi leakage). Jika pipeline logging internal Anda berbeda, sesuaikan makna dengan dokumentasi internal tim Anda.

### 5.1 Identitas & waktu

| Kolom | Deskripsi umum | Mengapa bisa berkaitan dengan drift/error translasi |
|-------|----------------|------------------------------------------------------|
| `frame` | Indeks frame / urutan sampel | **Dihapus dari X** di notebook prediksi (ordinal waktu, bukan penyebab fisik); urutan alternatif: `__row_order__`. |
| `timestamp` | Cap waktu (mis. epoch) | **Dihapus** — drift seharusnya dijelaskan oleh **dinamika tracking**, bukan label waktu absolut (kecuali ada drift sistematis jam; biasanya dihindari). |
| `realtime_sec` | Waktu “nyata” / wall clock | **Dihapus** — alasan serupa; risiko spurious correlation. |
| `track_inner_ms` | Latensi internal loop tracking (ms) | Latensi tinggi dapat berkorelasi dengan pose “tertinggal” dan error reprojeksi; dapat memengaruhi stabilitas translasi. (Analogi latency sistem real-time; lihat Liu & Layland, 1973.) |

### 5.2 Status & flag tracking

| Kolom | Deskripsi umum | Kaitan plausibel dengan drift |
|-------|----------------|------------------------------|
| `flag_ok` | Indikator frame “OK” | Status tracking buruk sering beriringan dengan lonjakan error pose. |
| `flag_recently_lost` | Baru saja kehilangan track | Transisi tracking lemah → lonjakan drift/inisialisasi ulang. |
| `flag_lost` | Kehilangan track | Kehilangan fitur/visual lock → pose tidak terikat geometri → error translasi besar (literatur VO/SLAM: kegagalan tracking; mis. Mur-Artal et al., ORB-SLAM, 2015). |
| `flag_vo` | Mode / flag VO | Perubahan mode mengubah sumber pose (visual vs inertial) → pola drift berbeda. |
| `tracking_state_name_OK`, `tracking_state_name_RECENTLY_LOST` | One-hot nama state | State machine tracking eksplisit. |
| `tracking_state_2`, `tracking_state_3` | Kode ordinal state | **Dihapus dari X** di notebook prediksi (ordinal diskret). |
| `host_hurwitz`, `build_id_unknown_build` | Metadata host/build | Variasi perangkat/firmware dapat mengubah perilaku estimator. |
| `source_file_*.csv` | One-hot sumber file | **Confounder** run/lingkungan berbeda; drift dapat bervariasi antar sumber. |

### 5.3 IMU & suhu

| Kolom | Deskripsi umum | Kaitan plausibel |
|-------|----------------|------------------|
| `cpu_temp_c` | Suhu CPU | Thermal throttling / clock dapat memengaruhi latensi dan kualitas estimasi. |
| `imu_scale_s` | Skala / faktor IMU | Skala salah → propagasi inertial salah → drift (jika IMU dipakai dalam rantai pose). |
| `bias_bax` … `bias_bwz` | Bias akselerometer & gyro (biasanya rad/s & m/s² tergantung konvensi) | Bias gyro memengaruhi integrasi orientasi; bias aksel memengaruhi skala gravitasi/linear acceleration — keduanya mempengaruhi **kompensasi** dan sisa error translasi dalam sistem VI-VO (Bar-Shalom et al., 2001). |

### 5.4 Pose delta & error geometri

| Kolom | Deskripsi umum | Kaitan plausibel |
|-------|----------------|------------------|
| `dx`, `dy`, `dz` | Translasi antar frame | **Menjadi target jarak** di notebook; **dihapus dari X** agar model tidak “mencontek” komponen yang sama. |
| `e_trans` | Besaran error translasi teragregasi (sering norm atau skor terkait) | Sangat berkorelasi dengan target jarak → **dihapus dari X** (leakage). |
| `drx`, `dry`, `drz` | Komponen rotasi / delta orientasi | Rotasi mempengaruhi proyeksi 3D→2D dan epipolar geometry → memengaruhi residual translasi; di notebook **dihapus** sesuai permintaan fokus “drift jarak” tanpa saluran rotasi di fitur. |
| `mean_desc_dist`, `desc_std` | Statistik deskriptor | Deskriptor buruk → matching lemah → pose tidak stabil. |
| `inlier_ratio` | Rasio inlier (RANSAC/geometry consistency) | Inlier rendah → model geometri lemah → error pose naik (Hartley & Zisserman, 2004 — residual & estimasi robust). |
| `reproj_error` | Error reprojeksi (piksel atau unit dinormalisasi) | Langsung mengukur konsistensi reproyeksi 3D–2D; naik saat tracking goyah → berkorelasi dengan error translasi. |
| `reproj_error_lag1`, `reproj_error_lag5`, `reproj_error_rolling_mean` | Lag & rolling reproj | Memberi **memori singkat** kualitas visual; autokorelasi error. |
| `feature_spread`, `feature_grid_coverage` | Sebaran & cakupan distribusi fitur | Scene degenerate (fitur kolinear/sedikit) memperburuk estimasi pose. |
| `feature_spread_trend`, `feature_grid_coverage_lag1` | Tren / lag cakupan | Deteksi degradasi visual sebelum lonjakan drift. |
| `map_inliers`, `map_inliers_trend`, `map_inliers_rolling_mean` | Inlier terhadap peta / trennya | Relokalisasi / constraint map mempengaruhi stabilitas translasi. |
| `keypoints`, `keypoints_rolling_mean` | Jumlah keypoint | Sedikit keypoint → observabilitas turun → drift. |
| `velocity_change`, `velocity_change_lag1`, `velocity_acceleration` | Perubahan kecepatan / akselerasi | Gerakan agresif memperbesar motion blur & nonlinearitas dinamika → residual naik. |
| `drift_prev`, `drift_rolling_mean` | Drift historis / smoothed | **Autokorelasi kuat** pada deret drift; sering dominan di SHAP untuk target drift jarak. |
| `inlier_ratio_lag1`, `inlier_ratio_trend` | Lag & tren inlier | Indikator memburuknya konsistensi geometri. |
| `tracking_health_score` | Skor komposit kesehatan tracking | Ringkasan multi-sinyal; berkorelasi dengan risiko error pose. |

---

## 6. File data di folder ini

| File | Peran |
|------|--------|
| `concat.csv` | Dataset gabungan utama untuk notebook prediksi. |
| `MH*.csv`, `VC*.csv` | CSV sumber (digabung untuk analisis tracking / konkatenasi). |

---

## 7. Cara menjalankan

1. Buka folder `Iter_2` di Jupyter / VS Code / Cursor.  
2. Instal dependensi (notebook memakai `%pip` di sel pertama, atau install manual: `pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`, `optuna`, `shap`, `torch`, `matplotlib`, `seaborn`).  
3. Jalankan `analisis_prediksi_xgboost_concat.ipynb` dari atas (**Restart Kernel & Run All** disarankan setelah perubahan dependensi).

---

## 8. Daftar pustaka (sumber metodologi & perangkat lunak)

1. Chen, T., & Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System.** KDD. ([ACM](https://dl.acm.org/doi/10.1145/2939672.2939785) / [arXiv:1603.02754](https://arxiv.org/abs/1603.02754))  
2. Lundberg, S. M., & Lee, S.-I. (2017). **A Unified Approach to Interpreting Model Predictions** (SHAP). NeurIPS. ([arXiv:1705.07874](https://arxiv.org/abs/1705.07874))  
3. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). **Optuna: A Next-generation Hyperparameter Optimization Framework.** KDD workshop / OSS. ([optuna.org](https://optuna.org/))  
4. Bergstra, J., & Bengio, Y. (2012). **Random Search for Hyper-Parameter Optimization.** JMLR. ([JMLR](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf))  
5. Pedregosa, F. et al. (2011). **Scikit-learn: Machine Learning in Python.** JMLR — metrik, `RandomizedSearchCV`, preprocessing. ([jmlr.org](https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf); [user guide](https://scikit-learn.org/stable/modules/model_evaluation.html))  
6. Hochreiter, S., & Schmidhuber, J. (1997). **Long Short-Term Memory.** Neural Computation. ([MIT Press](https://direct.mit.edu/neco/article-abstract/9/8/1735/6109))  
7. Paszke, A. et al. (2019). **PyTorch: An Imperative Style, High-Performance Deep Learning Library.** NeurIPS. ([papers.neurips.cc](https://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf))  
8. Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers** (PatchTST). ICLR. ([arXiv:2211.14730](https://arxiv.org/abs/2211.14730))  
9. Mur-Artal, R., Montiel, J. M. M., & Tardós, J. D. (2015). **ORB-SLAM: A Versatile and Accurate Monocular SLAM System.** IEEE T-RO. (referensi umum **tracking loss** / keyframe VO). ([IEEE](https://ieeexplore.ieee.org/document/7219438))  
10. Hartley, R., & Zisserman, A. (2004). **Multiple View Geometry in Computer Vision.** Cambridge University Press. (reprojeksi, geometri multi-tampilan).  
11. Bar-Shalom, Y., Li, X.-R., & Kirubarajan, T. (2001). **Estimation with Applications to Tracking and Navigation.** Wiley. (bias IMU, filtering).  
12. Lopez de Prado, M. (2018). **Advances in Financial Machine Learning.** Wiley — **penting** untuk CV temporal tanpa leakage (bab purging/embargo; analogi untuk deret robotik).  

---

## 9. Batasan & disclaimer

- Nama kolom diinterpretasikan menurut **konvensi umum** VO/tracking; **definisi pasti** harus diverifikasi dengan dokumentasi internal proyek Anda.  
- **R² tinggi** tidak menjamin model aman untuk deployment: periksa **leakage**, **split evaluasi** (acak vs temporal), dan **distribusi shift** antar `source_file_*`.  
- Model **LSTM/PatchTST** di notebook memakai **split temporal** untuk jendela; **tidak** selalu sebanding secara numerik dengan metrik tabular pada split acak — bandingkan hanya dengan konteks evaluasi yang sama.

---

*README ini mendokumentasikan alur dan rationale pada saat penulisan; jika notebook diubah, sesuaikan bagian “langkah” dan daftar kolom dengan versi terbaru.*
