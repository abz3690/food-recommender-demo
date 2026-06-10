# Hanoi Food Recommender Demo

Ứng dụng Streamlit được tách từ notebook `GK3_fixed_with_cuisine_filter.ipynb`.

## 1. Chuẩn bị dữ liệu

Cần hai file Excel:

- Food master có các sheet: `foods`, `places`, `place_food_map`.
- File rating có các sheet: `users`, `user_ratings`, `rated_only`.

Có hai cách sử dụng:

1. Đặt cả hai file `.xlsx` vào thư mục `data/`; app sẽ tự nhận diện theo tên sheet.
2. Không đặt file trong `data/` và tải hai file trực tiếp trên thanh bên của app.

## 2. Chạy trên máy

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 3. Public bằng Streamlit Community Cloud

1. Tạo repository GitHub.
2. Upload toàn bộ nội dung thư mục này.
3. Để dữ liệu trong `data/` nếu dữ liệu có thể công khai. Nếu không, app vẫn cho tải file thủ công mỗi phiên.
4. Trên Streamlit Community Cloud, chọn repository và file chính `app.py`.
5. Bấm Deploy.

## Chức năng

- Gợi ý món cá nhân theo user và cuisine.
- Chỉ lấy món chính và loại món đã đánh giá.
- Gợi ý cho hai người theo ba chiến lược.
- Gợi ý quán ăn và liên kết Google Maps.
- Khám phá danh mục món.
