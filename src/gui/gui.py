import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# Добавляем папку src в путь поиска модулей
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

# Теперь импортируем из processing
from processing.processing import process_video, save_report


# Функция для выбора папки с видео
def select_input_folder():
    folder_path = filedialog.askdirectory(title="Выберите папку с видео")
    if folder_path:
        input_folder_entry.delete(0, tk.END)
        input_folder_entry.insert(0, folder_path)


# Функция для выбора папки для сохранения результатов
def select_output_folder():
    folder_path = filedialog.askdirectory(title="Выберите папку для сохранения результатов")
    if folder_path:
        output_folder_entry.delete(0, tk.END)
        output_folder_entry.insert(0, folder_path)


# Функция для обработки всех видео в выбранной папке
def process_videos():
    input_folder_path = input_folder_entry.get()
    output_folder_path = output_folder_entry.get()

    # Проверим, выбраны ли обе папки
    if not input_folder_path or not output_folder_path:
        messagebox.showerror("Ошибка", "Пожалуйста, выберите папки для ввода и вывода")
        return

    # Получаем список всех .mp4 файлов в папке
    video_files = [f for f in os.listdir(input_folder_path) if f.endswith((".mp4", '.MP4'))]
    print(video_files)
    if not video_files:
        messagebox.showerror("Ошибка", "В выбранной папке нет видео файлов (.mp4)")
        return
    report = dict()

    # Обрабатываем каждое видео
    for video_file in video_files:
        input_video_path = os.path.join(input_folder_path, video_file)
        output_video_path = os.path.join(output_folder_path, f"processed_{video_file}")

        try:
            process_video(input_video_path, output_video_path, report)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при обработке {video_file}: {str(e)}")
            continue
    
    save_report(report, f'{output_folder_path}/report.json')

    # Уведомление об окончании обработки
    messagebox.showinfo("Успех", "Обработка видео завершена!")


# Создаем основной GUI
root = tk.Tk()
root.title("Обработка видео")

# Поле для выбора папки с видео
input_folder_label = tk.Label(root, text="Выберите папку с видео:")
input_folder_label.pack(pady=5)
input_folder_entry = tk.Entry(root, width=50)
input_folder_entry.pack(pady=5)
input_folder_button = tk.Button(root, text="Выбрать папку", command=select_input_folder)
input_folder_button.pack(pady=5)

# Поле для выбора папки для сохранения обработанных видео
output_folder_label = tk.Label(root, text="Выберите папку для сохранения результатов:")
output_folder_label.pack(pady=5)
output_folder_entry = tk.Entry(root, width=50)
output_folder_entry.pack(pady=5)
output_folder_button = tk.Button(root, text="Выбрать папку", command=select_output_folder)
output_folder_button.pack(pady=5)

# Кнопка для запуска обработки всех видео
process_button = tk.Button(root, text="Запустить обработку", command=process_videos)
process_button.pack(pady=20)

# Запуск GUI
root.mainloop()

