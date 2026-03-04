import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

def main():
    print("--- АВТОНОМНАЯ НЕЙРОСЕТЬ GPT-2 ---")
    model_path = get_resource_path("model")
    
    if not os.path.exists(model_path):
        print(f"Ошибка: Папка с моделью не найдена: {model_path}")
        input("Нажми Enter для выхода...")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            local_files_only=True,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print("Система готова. Введи текст:")
        
        while True:
            user_input = input("\nВы: ")
            if user_input.lower() in ['выход', 'exit']: break
            res = generator(user_input, max_new_tokens=50, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            print(f"\nИИ: {res[0]['generated_text'][len(user_input):].strip()}")
    except Exception as e:
        print(f"Ошибка: {e}")
        input("Нажми Enter...")

if __name__ == "__main__":
    main()
