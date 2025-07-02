

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import pandas as pd
from transformers import TextStreamer
torch.cuda.empty_cache()

import os
from huggingface_hub import login
login("your_huggingface_token_here")  # Replace with your Hugging Face token

# Base model
base_model_id = "meta-llama/Llama-3.2-1B"

# Your fine-tuned LoRA adapter repository
lora_adapter_id = 'sypher47/Test_AI_security'

# Load the model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    trust_remote_code=True,
)

# Attach the LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    lora_adapter_id,
    device_map="auto",
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)



# Định nghĩa alpaca_prompt gốc
alpaca_prompt = """This is a question about cybersecurity. Please write an answer.

### Question:
{}

### Response:
{}"""

# Đọc file CSV
csv_file = "B1.csv"  # Thay bằng đường dẫn thực tế
df = pd.read_csv(csv_file)
# Khởi tạo TextStreamer
text_streamer = TextStreamer(tokenizer)

# Danh sách để lưu kết quả dự đoán
predictions = []

# Duyệt qua từng dòng trong file CSV
for index, row in df.iterrows():
    question = row['question']
    ground_truth = row['ground_truth']
    
    # Tạo prompt cho câu hỏi
    prompt = alpaca_prompt.format(question, "") + tokenizer.eos_token
    
    # Tokenize input giống như trong mã gốc
    inputs = tokenizer(
        [prompt],
        return_tensors="pt"
    ).to("cuda")
    
    # Sinh đáp án từ mô hình, sử dụng TextStreamer
    with torch.no_grad():  # Tắt gradient để tiết kiệm tài nguyên
        outputs = model.generate(
            **inputs,
            # streamer=text_streamer,
            max_new_tokens=200,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Giải mã output
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Trích xuất phần đáp án từ response
    if "### Response:" in response:
        predicted_response = response.split("### Response:")[-1].strip()
    else:
        predicted_response = response.strip()  # Nếu không có ### Response:, lấy toàn bộ output
    
    # Chuẩn hóa predicted_response để so sánh (giả định đáp án là A, B, C, hoặc D)
    # predicted_response = predicted_response[0] if predicted_response and predicted_response[0] in ['A', 'B', 'C', 'D'] else predicted_response
    
    # Lưu kết quả
    predictions.append({
        'index': row['index'],
        'question': question,
        'llm_answer': predicted_response,
        'ground_truth': ground_truth
    })
    
print(f"Question {row['index']}: Predicted = {llm_answer}, Ground Truth = {ground_truth}")

# Lưu kết quả dự đoán vào file CSV
results_df = pd.DataFrame(predictions)
results_df.to_csv("result_finetune_1.csv", index=False, encoding='utf-8')