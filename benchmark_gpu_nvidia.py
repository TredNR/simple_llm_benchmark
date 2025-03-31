import time
import pandas as pd
from langchain_ollama import ChatOllama
from tqdm import tqdm
import subprocess
import psutil

questions_df = pd.read_csv("questions.csv")
questions = questions_df["questions"].tolist()

models = ["phi4", "deepseek-r1:32b", "deepseek-r1:70b", "gemma3:27b", "qwen2.5:72b"]



def get_gpu_memory_usage():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    gpu_usage = result.stdout.strip().split('\n')
    used, total = map(int, gpu_usage[0].split(', '))
    return used, total



def test_model(model, questions, model_name):
    total_time = 0
    memory_usage = []

    columns = [
        "question", "answer", "time_taken", 
        "gpu_memory_before", "gpu_memory_after", "gpu_memory_delta"
    ]
    model_df = pd.DataFrame(columns=columns)

    model_df.to_csv(f"{model_name}_results.csv", mode='w', header=True, index=False)

    for question in tqdm(questions):
        start_time = time.time()

        gpu_used_before, gpu_total = get_gpu_memory_usage()

        print(question)
        response = model.invoke(question)
        print(response)

        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time

        gpu_used_after, _ = get_gpu_memory_usage()

        result = {
            "question": question,
            "answer": response,
            "time_taken": elapsed_time,
            "gpu_memory_before": gpu_used_before,
            "gpu_memory_after": gpu_used_after,
            "gpu_memory_delta": gpu_used_after - gpu_used_before
        }

        model_df = model_df.append(result, ignore_index=True)

        model_df.to_csv(f"{model_name}_results.csv", mode='a', header=False, index=False)

        memory_usage.append({
            "time": elapsed_time,
            "gpu_memory_before": gpu_used_before,
            "gpu_memory_after": gpu_used_after,
            "gpu_memory_delta": gpu_used_after - gpu_used_before,
            "cpu_usage": psutil.cpu_percent(interval=1),  # Загрузка CPU
            "ram_usage": psutil.virtual_memory().percent  # Использование RAM
        })

    resource_df = pd.DataFrame(memory_usage)
    resource_df.to_csv(f"{model_name}_resources.csv", index=False)

    return total_time



total_time_all_models = {}
overall_times = []

for model in models:
    print(f"Тестирование модели: {model}")
    llm_model = ChatOllama(model=model)
    
    total_time = test_model(llm_model, questions, model)
    total_time_all_models[model] = total_time
    overall_times.append({
        "model": model,
        "total_time": total_time
    })
    print(f"Общее время для модели {model}: {total_time} секунд")

total_time_overall = sum(total_time_all_models.values())
print(f"Общее время на все вопросы для всех моделей: {total_time_overall} секунд")

overall_times_df = pd.DataFrame(overall_times)
overall_times_df.to_csv("overall_times.csv", index=False)