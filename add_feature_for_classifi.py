import pandas as pd
import numpy as np

# Load the features CSV
df = pd.read_csv('SnP_daily_update_AMZN_features.csv', parse_dates=['Date'])

# 1. สร้างคอลัมน์ราคาในอนาคต 7 วัน (Future Close)
# shift(-7) คือการดึงราคาจากอนาคตมาใส่วันปัจจุบัน (มองไปข้างหน้า 7 แถว)
df['Future_Close_7d'] = df['Close'].shift(-7)

# 2. คำนวณผลตอบแทน 7 วัน (Target แบบ Regression)
df['Target_Return_7d'] = (df['Future_Close_7d'] - df['Close']) / df['Close']

# 3. สร้าง Target แบบ Classification (Up/Down/Sideways)
# กำหนด Threshold ที่ต้องการ (เช่น 3% หรือ 0.03)
threshold = 0.02

conditions = [
    (df['Target_Return_7d'] > threshold),       # เงื่อนไขขาขึ้น
    (df['Target_Return_7d'] < -threshold)      # เงื่อนไขขาลง
]

# ให้ค่า 2 = Up, 0 = Down, 1 = Sideways (ค่า Default)
choices = [2, 0]
df['Target_Class'] = np.select(conditions, choices, default=1)

# 4. ลบข้อมูล 7 แถวสุดท้ายทิ้ง (เพราะไม่มีข้อมูลอนาคตให้ทำนาย จะเป็น NaN)
df_clean = df.dropna(subset=['Target_Class', 'Future_Close_7d'])

# Save to new CSV
df_clean.to_csv('CLASSIFY-SnP_daily_update_AMZN_features_with_target.csv', index=False)
print(f"Saved {len(df_clean)} rows to CLASSIFY-SnP_daily_update_AMZN_features_with_target.csv")
print(f"\nTarget_Class distribution:")
print(df_clean['Target_Class'].value_counts().sort_index().rename({0: '0 (Down)', 1: '1 (Sideways)', 2: '2 (Up)'}))