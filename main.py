from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
import numpy as np
from tqdm import tqdm

from argparse import ArgumentParser

# Настройка обработчика для аргументов командной строки
parser = ArgumentParser(description='Детектор, находящий людей на видео')
parser.add_argument('-i', '--input', help='Путь до входного видеофайла', required=True)
parser.add_argument('-o', '--output', help='Желаемый путь до выходного видеофайла', required=True)
parser.add_argument('-m', '--model', help='Обозначение желаемой версии модели YOLO11',
                    choices=('n', 's', 'm', 'l', 'x'), default='x')
parser.add_argument('-c', '--confidence', help='Минимальный порог уверенности при детекции',
                    type=float, default=0.25)
parser.add_argument('-s', '--show', help='Отображать видео в процессе обработки', action='store_true')

# Парсинг аргументов командной строки
args = parser.parse_args()

# Валидация аргумента `confidence`
if args.confidence < 0 or args.confidence > 1:
    raise ValueError('Значение аргумента `confidence` должно лежать между 0.0 и 1.0')
# Остальные аргументы валидируются библиотекой `argparse` автоматически


def draw_boxes(frame: np.ndarray, result: Results, color=(0, 255, 0)) -> np.ndarray:
    """
    Отрисовка прямоугольников и подписей к ним на изображении

    :param frame: изображение
    :param result: результат работы YOLO над изображением
    :param color: цвет прямоугольников и текста
    """

    # Копирование кадра
    new_frame = frame.copy()

    # Цикл по предсказанным боксам
    for box in result.boxes:
        # Получение описания бокса
        x1, y1, x2, y2 = map(int, np.round(box.xyxy[0]))
        class_label = int(box.cls[0])
        confidence = box.conf[0]

        # Если бокс соответствует классу человека (0), то рисуем и подписываем его
        if class_label == 0:
            cv2.rectangle(new_frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.putText(new_frame, f'Person ({confidence:.2f})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return new_frame


# Инициализация выбранной модели YOLO
yolo_model = YOLO(f'yolo11{args.model}.pt')

# Открытие входного видео
input_video = cv2.VideoCapture(args.input)

# Получение параметров входного видео
width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(input_video.get(cv2.CAP_PROP_FPS))
frame_cnt = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

# Создание результирующего видео
# noinspection PyUnresolvedReferences
output_video = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Цикл по кадрам в видео
for _ in tqdm(range(frame_cnt)):
    # Получение следующего кадра
    frame_captured, frame = input_video.read()
    if not frame_captured:  # Если кадры закончились, завершаем цикл
        break

    # Инференс при помощи YOLO
    result = yolo_model(frame, conf=args.confidence, verbose=False)[0]

    # Отрисовка результатов
    new_frame = draw_boxes(frame, result)

    # Запись нового кадра в видео
    output_video.write(new_frame)

    # Если включен мгновенный показ результатов, то отображаем новый кадр
    if args.show:
        cv2.imshow(args.output, new_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Закрытие всех видео и окон
input_video.release()
output_video.release()
cv2.destroyAllWindows()
