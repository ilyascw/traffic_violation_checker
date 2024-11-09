import cv2
import yaml
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm


class VideoProcessor:
    def __init__(self, sign_model_path, traffic_light_model_url, sign_classes_yaml, lane_detection_algo_path):
        """
        Инициализация класса для обработки видео с использованием моделей для распознавания знаков и светофоров.
        
        :param sign_model_path: Путь к файлу модели для распознавания знаков (YOLO).
        :param traffic_light_model_url: Ссылка на модель для распознавания светофоров.
        :param sign_classes_yaml: Путь к файлу в формате YAML, который содержит классы знаков.
        :param lane_detection_algo_path: Путь к алгоритму для распознавания разметки (пока не реализовано).
        """
        self.sign_model = YOLO(sign_model_path)
        self.traffic_light_model = YOLO(traffic_light_model_url)
        
        # Загружаем классы знаков из YAML
        with open(sign_classes_yaml, 'r') as f:
            yaml_dict = yaml.safe_load(f)
        self.sign_classes = yaml_dict['names']
        
        self.lane_detection_algo_path = lane_detection_algo_path  # Алгоритм для распознавания разметки (будет реализовано позже)

    def process_traffic_signs(self, frame, confidence=0.2):
            """
            Функция для обработки кадра и распознавания дорожных знаков.
            
            :param frame: Входной кадр для обработки.
            :return: Список с типами знаков и их координатами.
            """
            results = self.sign_model(frame)
            boxes = results[0].boxes.xywh.tolist()
            cls = results[0].boxes.cls.tolist()
            conf = results[0].boxes.conf
            sign_bboxes = []

            # Проходим по всем найденным объектам и определяем их тип
            for (box, cls_, conf_) in zip(boxes, cls, conf):  # Перебор объектов

                x, y, w, h = box
                class_name = self.sign_classes[int(cls_)]
                if conf_ > confidence:
                
                    sign_bboxes.append((class_name, (x, y, w, h)))
            
            return sign_bboxes
    
    def process_traffic_lights_and_stop_signs(self, frame, confidence=0.2):
        """
        Функция для обработки кадра и распознавания светофоров и стоп-знаков.
        
        :param frame: Входной кадр для обработки.
        :return: Список с координатами для светофоров и стоп-знаков.
        """
        results = self.traffic_light_model(frame)
        boxes = results[0].boxes.xywh.tolist()
        cls = results[0].boxes.cls.tolist()
        conf = results[0].boxes.conf
        bboxes_traffic_lights = []
        bboxes_stop_signs = []

        # Проходим по всем найденным объектам и определяем их тип
        for (box, cls_, conf_) in zip(boxes, cls, conf):  # Перебор объектов

            x, y, w, h = box
            class_name = results[0].names[int(cls_)]
            if conf_ > confidence:
                if class_name.lower() == 'traffic light':  # Если это светофор
                    bboxes_traffic_lights.append((x, y, w, h))
                elif class_name.lower() == 'stop sign':  # Если это стоп-знак
                    bboxes_stop_signs.append((x, y, w, h))
        
        return bboxes_traffic_lights, bboxes_stop_signs

    
    def process_frame(self, frame):
        """
        Обработка одного кадра: распознавание светофоров, стоп-знаков и дорожных знаков.
        
        :param frame: Входной кадр для обработки.
        :return: Словарь с результатами по каждому типу объектов.
        """
        # Получаем координаты светофоров и стоп-знаков
        bboxes_traffic_lights, bboxes_stop_signs = self.process_traffic_lights_and_stop_signs(frame)
        
        # Получаем координаты дорожных знаков
        sign_bboxes = self.process_traffic_signs(frame)
        
        return {
            'traffic_lights': bboxes_traffic_lights,
            'stop_signs': bboxes_stop_signs,
            'traffic_signs': sign_bboxes
        }



    def process_video(self, video_path, frame_step=1, show=False):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Получаем FPS видео
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Общее количество кадров в видео

        # Проверка корректности параметров
        if frame_step <= 0:
            raise ValueError("Параметр frame_step должен быть больше нуля.")
        if total_frames == 0:
            raise ValueError("Видео не содержит кадров.")

        # Определяем шаг для обработки кадров
        if frame_step >= fps:
            frame_step = total_frames  # Берем только один кадр из видео, если frame_step >= FPS

        print(f"Общее количество кадров: {total_frames}, FPS: {fps}")
        print(f"Обрабатывается каждый {frame_step}-й кадр.")

        frame_count = 0
        processed_count = 0

        # Прогресс-бар для отслеживания обработки
        with tqdm(total=(total_frames // frame_step), desc="Processing Video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Обработка кадра только при условии, что номер кадра кратен frame_step
                if frame_count % frame_step != 0:
                    continue

                # Обрабатываем кадр
                results = self.process_frame(frame)
                processed_count += 1

                # Обновляем прогресс-бар
                pbar.update(1)

                if show:
                    # Отображаем кадр с результатами
                    for label, bboxes in results.items():
                        for x, y, w, h in bboxes:
                            cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)),
                                        (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
                            # Отображаем метку класса рядом с bbox
                            cv2.putText(frame, label, (int(x - w / 2), int(y - h / 2) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    cv2.imshow("Processed Frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()

        print(f"Обработано {processed_count} кадров.")


