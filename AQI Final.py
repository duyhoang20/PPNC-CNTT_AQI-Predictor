import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt

# === 1. Đọc và làm sạch dữ liệu ===
df = pd.read_csv(r'C:/Users/Admin/Downloads/historical_air_quality_2021_vi.csv')


def clean_value(x):
    try:
        if x == '-' or pd.isna(x):
            return np.nan
        return float(str(x).replace(',', ''))
    except:
        return np.nan

columns_to_clean = ['CO', 'NO2', 'O3', 'SO2', 'Bụi PM10', 'Bụi PM2.5', 'Áp suất', 'Nhiệt độ', 'Độ ẩm', 'Tốc độ gió', 'Chỉ số AQI']
for col in columns_to_clean:
    df[col] = df[col].apply(clean_value)

df['Datetime'] = pd.to_datetime(df['Thời gian cập nhật'], errors='coerce')

# === 2. Nới lỏng điều kiện loại bỏ dòng thiếu dữ liệu ===
required_inputs = ['CO', 'NO2', 'O3', 'SO2', 'Bụi PM10', 'Bụi PM2.5',
                   'Áp suất', 'Nhiệt độ', 'Độ ẩm', 'Tốc độ gió']
df = df[df[required_inputs].isnull().sum(axis=1) <= 2]
df = df.dropna(subset=['Chỉ số AQI', 'Datetime', 'Tên trạm'])

# === 3. Tạo dữ liệu huấn luyện ===
features = ['Bụi PM2.5', 'Bụi PM10', 'CO', 'NO2', 'O3', 'SO2',
            'Áp suất', 'Nhiệt độ', 'Độ ẩm', 'Tốc độ gió']
X = df[features]
y = df['Chỉ số AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Chuẩn hóa dữ liệu ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 5. Huấn luyện mô hình Random Forest ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# === 6. Đánh giá mô hình ===
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== ĐÁNH GIÁ MÔ HÌNH ===")
print(f"MAE (Sai số tuyệt đối trung bình): {mae:.2f}")
print(f"RMSE (Căn sai số bình phương trung bình): {rmse:.2f}")
print(f"R² (Hệ số xác định): {r2:.4f}")

print("\n=== NHẬN XÉT ===")
print(f"- MAE ≈ {mae:.2f} → Trung bình dự đoán lệch khoảng {mae:.2f} đơn vị AQI.")
if mae <= 5:
    print("  → Rất chính xác, sai số rất thấp.")
elif mae <= 10:
    print("  → Khá tốt, sai số ở mức chấp nhận được.")
else:
    print("  → Sai số khá lớn, cần cải thiện thêm.")

print(f"- RMSE ≈ {rmse:.2f} → Thấp cho thấy mô hình ổn định.")
if rmse <= 10:
    print("  → Mô hình không có sai số lớn đột biến.")
else:
    print("  → Có khả năng tồn tại vài điểm dữ liệu gây sai số lớn.")

if r2 >= 0.9:
    print(f"- R² ≈ {r2:.4f} → Xuất sắc! Mô hình giải thích ~{r2*100:.1f}% phương sai AQI.")
elif r2 >= 0.75:
    print(f"- R² ≈ {r2:.4f} → Tốt! Mô hình giải thích phần lớn sự biến động AQI.")
elif r2 >= 0.5:
    print(f"- R² ≈ {r2:.4f} → Trung bình. Có thể dự đoán xu hướng nhưng chưa ổn định.")
else:
    print(f"- R² ≈ {r2:.4f} → Kém. Cần cải thiện mô hình.")

# === 7. Dự đoán toàn bộ dữ liệu ===
df['Predicted_AQI'] = model.predict(scaler.transform(X))

def classify_aqi(aqi_value):
    if aqi_value <= 50: return 'Tốt'
    elif aqi_value <= 100: return 'Trung bình'
    elif aqi_value <= 150: return 'Kém'
    elif aqi_value <= 200: return 'Xấu'
    elif aqi_value <= 300: return 'Rất xấu'
    else: return 'Nguy hại'

df['AQI_Level'] = df['Predicted_AQI'].apply(classify_aqi)

# === 8. Chọn 20 trạm khác nhau ===
stations = df['Tên trạm'].dropna().unique()
sample_size = min(20, len(stations))
chosen_stations = np.random.choice(stations, size=sample_size, replace=False)
sample_df = df[df['Tên trạm'].isin(chosen_stations)].groupby('Tên trạm', group_keys=False)[df.columns.tolist()].apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)
output_df = sample_df[['Datetime', 'Tên trạm'] + features + ['Chỉ số AQI', 'Predicted_AQI', 'AQI_Level']]


# === 9. Giao diện PyQt5 để hiển thị bảng kết quả ===
class AQITable(QWidget):
    def __init__(self, data):
        super().__init__()
        self.setWindowTitle("Dự đoán chất lượng không khí - VN AQI")
        self.resize(1200, 600)
        layout = QVBoxLayout()

        label = QLabel("\u2728 Kết quả dự đoán AQI tại 20 trạm khác nhau")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        table = QTableWidget()
        table.setRowCount(len(data))
        table.setColumnCount(len(data.columns))
        table.setHorizontalHeaderLabels(data.columns.tolist())

        for i in range(len(data)):
            for j in range(len(data.columns)):
                item = QTableWidgetItem(str(data.iloc[i, j]))
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(i, j, item)

        table.resizeColumnsToContents()
        layout.addWidget(table)
        self.setLayout(layout)

def ve_bieu_do_so_sanh(output_df):
    import matplotlib.pyplot as plt
    output_df['Trạm rút gọn'] = [f'Trạm {i+1}' for i in range(len(output_df))]

    stations = output_df['Trạm rút gọn']
    real_aqi = output_df['Chỉ số AQI']
    predicted_aqi = output_df['Predicted_AQI']

    x = np.arange(len(stations))
    width = 0.35

    plt.figure(figsize=(13, 6))
    bars1 = plt.bar(x - width/2, real_aqi, width, label='Thực tế', color='skyblue')
    bars2 = plt.bar(x + width/2, predicted_aqi, width, label='Dự đoán', color='orange')

    for i, bar in enumerate(bars1):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{real_aqi.iloc[i]:.0f}', ha='center', fontsize=8)

    for i, bar in enumerate(bars2):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{predicted_aqi.iloc[i]:.2f}', ha='center', fontsize=8)

    plt.xlabel('Trạm quan trắc')
    plt.ylabel('Chỉ số AQI')
    plt.title('Hình 1. So sánh chỉ số AQI thực tế và mô hình dự đoán tại 20 trạm')
    plt.xticks(x, stations, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('so_sanh_AQI_20_tram.png')
    plt.show()



if __name__ == '__main__':
    ve_bieu_do_so_sanh(output_df)  
    app = QApplication(sys.argv)
    win = AQITable(output_df)
    win.show()
    sys.exit(app.exec_())

