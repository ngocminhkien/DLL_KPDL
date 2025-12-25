# Phân Tích Dữ Liệu Bệnh Tim: Luật Kết Hợp & Phân Cụm

Dự án này thực hiện khai phá dữ liệu (Data Mining) trên tập dữ liệu bệnh tim (`Heart Disease Dataset`) nhằm tìm kiếm các mẫu tiềm ẩn và phân nhóm bệnh nhân dựa trên các chỉ số y tế. 

Phương pháp tiếp cận bao gồm: **Tiền xử lý nâng cao → Luật kết hợp (Apriori) → Phân cụm (K-Means) → Đánh giá trực quan**.

---

## 📋 Mục Lục
1. [Giới thiệu](#-giới-thiệu)
2. [Thông tin bộ dữ liệu](#-thông-tin-bộ-dữ-liệu)
3. [Quy trình phân tích](#-quy-trình-phân-tích)
4. [Yêu cầu hệ thống & Cài đặt](#-yêu-cầu-hệ-thống--cài-đặt)
5. [Hướng dẫn chạy chương trình](#-hướng-dẫn-chạy-chương-trình)
6. [Kết quả chính](#-kết-quả-chính)

---

## 📖 Giới thiệu
Mục tiêu của dự án là áp dụng các kỹ thuật học máy không giám sát (Unsupervised Learning) để giải quyết hai bài toán:
1.  **Tìm luật kết hợp (Association Rules):** Xác định các triệu chứng hoặc chỉ số y tế thường xuất hiện cùng nhau (Ví dụ: Mối liên hệ giữa Tuổi tác, Cholesterol và Huyết áp).
2.  **Phân cụm bệnh nhân (Clustering):** Gom nhóm bệnh nhân thành các cụm có đặc điểm tương đồng để có cái nhìn tổng quan về quần thể dữ liệu.

---

## 💾 Thông tin bộ dữ liệu
* **Tên file:** `HeartDiseaseTrain-Test.csv`
* **Số lượng bản ghi:** 1025 mẫu.
* **Số lượng thuộc tính:** 14 cột.

**Các thuộc tính quan trọng:**
| Tên cột | Mô tả |
| :--- | :--- |
| `age` | Tuổi của bệnh nhân. |
| `sex` | Giới tính (Male/Female). |
| `chest_pain_type` | Loại đau ngực (Typical angina, Atypical angina, v.v.). |
| `resting_blood_pressure` | Huyết áp khi nghỉ ngơi (mm Hg). |
| `cholestoral` | Chỉ số Cholesterol huyết thanh (mg/dl). |
| `fasting_blood_sugar` | Đường huyết khi đói (> 120 mg/dl hoặc < 120 mg/dl). |
| `target` | Nhãn phân loại gốc (1: Có bệnh, 0: Không bệnh). |
| ... | Các chỉ số khác (ECG, Max Heart Rate, Slope, Thalassemia...). |

---

## ⚙️ Quy trình phân tích

Toàn bộ quy trình được thực hiện qua 4 bước chính trong Notebook:

### 1. Tiền xử lý dữ liệu (Preprocessing)
Do thuật toán **Apriori** yêu cầu dữ liệu đầu vào dạng "giỏ hàng" (transaction/categorical) và **K-Means** cần dữ liệu số hóa, chúng tôi thực hiện:
* **Rời rạc hóa (Binning/Discretization):** Chuyển đổi các biến liên tục thành các khoảng giá trị phân loại.
    * *Age:* `<45`, `45-54`, `55-64`, `>=65`.
    * *Blood Pressure:* Normal (`<120`), Prehypertension (`120-139`), High (`>=140`).
    * *Cholesterol:* Desirable (`<200`), Borderline (`200-239`), High (`>=240`).
    * *Max Heart Rate:* Chia theo tứ phân vị (Quartiles).
    * *Oldpeak:* Chia ngưỡng `0`, `0-1.5`, `>1.5`.
* **Mã hóa One-Hot (One-Hot Encoding):** Chuyển đổi toàn bộ dữ liệu (cả biến hạng mục gốc và biến vừa rời rạc hóa) thành ma trận nhị phân (0 và 1).

### 2. Khai phá luật kết hợp (Apriori)
* Sử dụng thư viện `mlxtend`.
* Tìm các **Tập phổ biến (Frequent Itemsets)**: Các nhóm thuộc tính xuất hiện cùng nhau với tần suất cao (Support ≥ 0.2).
* Sinh **Luật kết hợp**: Tìm các luật nhân quả dạng "Nếu A thì B" dựa trên độ đo `Lift` (Lift > 1).

### 3. Phân cụm (Clustering - K-Means)
* Sử dụng dữ liệu đã mã hóa One-Hot làm đầu vào.
* **Chọn số cụm K tối ưu:**
    * *Phương pháp Elbow:* Quan sát điểm gãy của độ lỗi (Inertia).
    * *Phương pháp Silhouette Score:* Chọn K có điểm hệ số dáng điệu cao nhất.

### 4. Đánh giá & Trực quan hóa
* Sử dụng **PCA (Principal Component Analysis)** để giảm chiều dữ liệu xuống 2D nhằm vẽ biểu đồ phân bố các cụm (Scatter plot).
* Thống kê số lượng mẫu trong từng cụm.

---

## 🛠 Yêu cầu hệ thống & Cài đặt

Dự án được viết bằng Python 3. Bạn cần cài đặt các thư viện sau:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend
