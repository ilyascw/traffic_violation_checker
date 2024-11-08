import cv2
import torch
import json

# Функция для сохранения отчета в JSON файл
def save_report(report, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)  # Сохраняем с отступами для читаемости
        print(f"Отчет успешно сохранен в {filename}")
    except Exception as e:
        print(f"Ошибка при сохранении отчета: {str(e)}")

def process_video(input_video, output_path, report):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Загрузка модели YOLOv5 Small (yolov5s)

    # Открываем видеофайл с помощью OpenCV
    cap = cv2.VideoCapture(input_video)

    # Получение ширины и высоты кадров для сохранения видео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Создание объекта VideoWriter для сохранения обработанного видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

    frame_number = 0  # Инициализируем номер кадра

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Отправка кадра в модель для детекции
        results = model(frame)

        if input_video not in report:
            report[input_video] = {}  # Если видео еще нет в отчете, создаем запись

        # Добавляем новый кадр в отчет
        report[input_video][frame_number] = []

        # Обработка результатов
        for det in results.xyxy[0]:
            x1, y1, x2, y2, confidence, cls = det
            label = model.names[int(cls)]  # Получаем метку класса

            # Добавляем аннотацию в отчет
            report[input_video][frame_number].append({
                "coordinates": (x1.item(), y1.item(), x2.item(), y2.item()),  # Преобразуем в обычные числа
                "confidence": confidence.item(),  # Преобразуем в обычное число
                "class": label
            })

            # Рисуем рамку и метку
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Записываем обработанный кадр в выходное видео
        out.write(frame)

        # Отображение обработанного кадра
        cv2.imshow('Processed Frame', frame)

        frame_number += 1  # Инкрементируем номер кадра

        # Выход из цикла по нажатию клавиши "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    out.release()
    cv2.destroyAllWindows()
