# Пути к исходным данным на сервере и каталогу вывода (ARD100 → YOLOMG dual).
ARD100_ROOT = "/data/ARD100"
TRAIN_VIDEOS = f"{ARD100_ROOT}/train_videos"
TEST_VIDEOS = f"{ARD100_ROOT}/test_videos"
ANNOTATIONS = f"{ARD100_ROOT}/annotations"

OUT_ROOT = "/home/tanyadiplom/data/ard100"
IMAGES_OUT = f"{OUT_ROOT}/images"
IMAGES2_OUT = f"{OUT_ROOT}/images2"  # второй поток (маски); в коде datasets ожидается сегмент "images2"
LABELS_OUT = f"{OUT_ROOT}/labels"
