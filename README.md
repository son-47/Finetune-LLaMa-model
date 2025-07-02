# Dự án Fine-tune & Đánh giá Mô hình AI cho Câu hỏi An ninh mạng

## Mô tả

Dự án này sử dụng mô hình ngôn ngữ lớn (LLM) được fine-tune với adapter LoRA để trả lời các câu hỏi trắc nghiệm về an ninh mạng. Dữ liệu đầu vào là file CSV chứa các câu hỏi và đáp án gốc. Script `test_model.py` sẽ sinh đáp án từ mô hình và lưu kết quả dự đoán ra file CSV để đánh giá.

## Cấu trúc thư mục

- `test_model.py`: Script chính để chạy mô hình và sinh kết quả.
- `B1.csv`: File dữ liệu đầu vào chứa các câu hỏi và đáp án gốc.
- `result_finetune_1.csv`: File kết quả dự đoán sau khi chạy script.
- `requirements.txt`: Danh sách các thư viện Python cần thiết.
- `README.txt` hoặc `README.md`: Hướng dẫn sử dụng.
- Các file/thư mục khác: dữ liệu, tài liệu mô tả, kết quả trước/sau fine-tune,...

## Hướng dẫn cài đặt & sử dụng

### 1. Cài đặt môi trường

Chuyển tới thư mục chứa project:

```sh
cd NguyenHoangSon.sonthaihoathcs
```

Cài đặt các thư viện cần thiết:

```sh
pip install -r requirements.txt
```

### 2. Thiết lập Hugging Face Token

- Thay dòng sau trong `test_model.py` bằng Hugging Face token của bạn:
  ```python
  login("your_huggingface_token_here")
  ```

### 3. Chạy mô hình để sinh kết quả

```sh
python test_model.py
```

Sau khi chạy xong, kết quả sẽ được lưu vào file `result_finetune_1.csv`.

## Thông tin file dữ liệu

- `B1.csv` cần có các cột: `index`, `question`, `ground_truth`.
- Đảm bảo dữ liệu được mã hóa UTF-8.

