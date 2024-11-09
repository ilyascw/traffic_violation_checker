import cv2
import yaml
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class VideoProcessor:
    '''
    Класс предназначен для обработки 1 видео на предмет нарушений ПДД.
    Класс работает с изображением в цветовом пространстве RGB.
    '''
    def __init__(self, sign_model_path, traffic_light_model_url, sign_classes_yaml):
        """
        Инициализация класса для обработки видео с использованием моделей для распознавания знаков и светофоров.
        
        :param sign_model_path: Путь к файлу модели для распознавания знаков (YOLO).
        :param traffic_light_model_url: Ссылка на модель для распознавания светофоров.
        :param sign_classes_yaml: Путь к файлу в формате YAML, который содержит классы знаков.
        :param lane_detection_algo_path: Путь к алгоритму для распознавания разметки (пока не реализовано).
        """
        self.sign_model = YOLO(sign_model_path)
        self.traffic_light_model = YOLO(traffic_light_model_url)
        self.traffic_signs = {"2.5": "Движение без остановки запрещено", "1.12": "Разметка стоп-линия", "5.15.1": "Полоса для маршрутных транспортных средств", "5.15.2": "Полоса для маршрутных транспортных средств", "3.1": "Въезд запрещен (кирпич)", "1.4.1": "Обозначение полос движения", "4.1.1": "Движение прямо", "2.1": "Главная дорога", "3.27": "Остановка запрещена", "3.28": "Стоянка запрещена", "3.18.1": "Поворот налево запрещен", "3.18.2": "Разворот запрещен", "4.1.3": "Движение налево", "3.20": "Обгон запрещен", "1.1": "Сплошная линия разметки"}

        # Загружаем классы знаков из YAML
        with open(sign_classes_yaml, 'r') as f:
            yaml_dict = yaml.safe_load(f)
        self.sign_classes = yaml_dict['names']
        
    def violation_1(self):
        pass

    def draw_annotations(self, frame, results:dict):
        """
        Рисует аннотации на изображении на основе результатов распознавания объектов.

        :param frame: Кадр изображения, на котором будут отображаться аннотации.
        :param results: Словарь с результатами распознавания, где ключи — метки объектов, 
                        а значения — списки координат их bounding boxes.
                        Формат словаря: {'label1': [(x1, y1, w1, h1), ...], 'label2': [(x2, y2, w2, h2), ...], ...}
        
        Описание:
        - Функция проходит по всем меткам и их bounding boxes в словаре результатов и рисует их на изображении.
        - Рисует прямоугольники вокруг обнаруженных объектов и отображает метки классов рядом с ними.
        """
        for label, bboxes in results.items():
            for x, y, w, h in bboxes:
                cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)),
                            (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
                # Отображаем метку класса рядом с bbox
                cv2.putText(frame, label, (int(x - w / 2), int(y - h / 2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                plt.imshow(frame)
                plt.show()

    def process_traffic_signs(self, frame, confidence=0.2):
            """
            Функция для обработки кадра и распознавания дорожных знаков.
            
            :param confidence: уверенность модели в правильной идентификации объекта.
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
    
    def crop_and_identify_color(self, frame, bbox):
        """
        Обрезает изображение по координатам bounding box и определяет цвет светофора.

        :param frame: Исходное изображение (кадр) для обработки.
        :param bbox: Координаты bounding box в формате (x, y, w, h), где x и y - координаты центра, 
                    w и h - ширина и высота.
        :return: Целочисленное значение, обозначающее преобладающий цвет:
                1 - красный или желтый (сигналы, означающие остановку),
                0 - зеленый (сигнал продолжения движения),
                None - цвет не определен.
        """
        # bbox — координаты в формате (x, y, w, h)
        x, y, w, h = bbox

        # Определение верхнего левого и нижнего правого углов
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # Обрезка изображения по заданным координатам
        cropped_image = frame[y1:y2, x1:x2]

        # Преобразование в цветовое пространство HSV
        hsv_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)

        # Определение диапазонов цвета для красного, желтого и зеленого
        red_lower1 = np.array([0, 70, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 70, 50])
        red_upper2 = np.array([180, 255, 255])
        green_lower = np.array([35, 70, 50])
        green_upper = np.array([85, 255, 255])
        yellow_lower = np.array([20, 70, 50])
        yellow_upper = np.array([30, 255, 255])

        # Маски для определения цветов
        mask_red = cv2.inRange(hsv_cropped, red_lower1, red_upper1) + cv2.inRange(hsv_cropped, red_lower2, red_upper2)
        mask_green = cv2.inRange(hsv_cropped, green_lower, green_upper)
        mask_yellow = cv2.inRange(hsv_cropped, yellow_lower, yellow_upper)

        # Подсчет количества пикселей каждого цвета
        red_count = np.sum(mask_red > 0)
        green_count = np.sum(mask_green > 0)
        yellow_count = np.sum(mask_yellow > 0)

        # Определение преобладающего цвета
        if red_count > green_count and red_count > yellow_count:
            return 1
        elif green_count > red_count and green_count > yellow_count:
            return 0
        elif yellow_count > red_count and yellow_count > green_count:
            return 1
        else:
            return 0
        
    def process_traffic_lights_and_stop_signs(self, frame, confidence=0.25):
        """
        Функция для обработки кадра и распознавания светофоров и стоп-знаков.
        
        :param frame: Входной кадр для обработки.
        :param confidence: уверенность модели в правильной идентификации объекта.
        :return: Список с координатами для светофоров и стоп-знаков.
        """
        results = self.traffic_light_model(frame)
        boxes = results[0].boxes.xywh.tolist()
        cls = results[0].boxes.cls.tolist()
        conf = results[0].boxes.conf
        bboxes_traffic_lights = []
        bboxes_stop_signs = []
        traffic_lights_colors = []

        # Проходим по всем найденным объектам и определяем их тип
        for (box, cls_, conf_) in zip(boxes, cls, conf):  # Перебор объектов

            x, y, w, h = box
            class_name = results[0].names[int(cls_)]
            if conf_ > confidence:
                if class_name.lower() == 'traffic light':  # Если это светофор
                    bboxes_traffic_lights.append((x, y, w, h))
                    traffic_lights_colors.append(self.crop_and_identify_color(frame, box))
        
        return bboxes_traffic_lights, traffic_lights_colors

    
    def process_frame(self, frame):
        """
        Обработка одного кадра: распознавание светофоров, стоп-знаков и дорожных знаков.
        
        :param frame: Входной кадр для обработки.
        :return: Словарь с результатами по каждому типу объектов.
        """
        # Получаем координаты светофоров и стоп-знаков
        bboxes_traffic_lights, color = self.process_traffic_lights_and_stop_signs(frame)
        
        # Получаем координаты дорожных знаков
        sign_bboxes = self.process_traffic_signs(frame)
        
        return {
            'traffic_lights': bboxes_traffic_lights,
            'traffic_signs': sign_bboxes,
            'color': color
        }

    def draw_annotations(self, frame, results:dict):
        """
        Рисует аннотации на изображении на основе результатов распознавания объектов.

        :param frame: Кадр изображения, на котором будут отображаться аннотации.
        :param results: Словарь с результатами распознавания, где ключи — метки объектов, 
                        а значения — списки координат их bounding boxes.
                        Формат словаря: {'label1': [(x1, y1, w1, h1), ...], 'label2': [(x2, y2, w2, h2), ...], ...}
        
        Описание:
        - Функция проходит по всем меткам и их bounding boxes в словаре результатов и рисует их на изображении.
        - Рисует прямоугольники вокруг обнаруженных объектов и отображает метки классов рядом с ними.
        """
        for label, bboxes in results.items():
            for x, y, w, h in bboxes:
                cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)),
                            (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
                # Отображаем метку класса рядом с bbox
                cv2.putText(frame, label, (int(x - w / 2), int(y - h / 2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # преобразовываем сразу в правильное цветое пространство, это важно
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


