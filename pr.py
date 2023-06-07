import cv2
import numpy as np

def calculate_iou(rect1, rect2):
    intersection_x1 = max(rect1[0], rect2[0])
    intersection_y1 = max(rect1[1], rect2[1])
    intersection_x2 = min(rect1[2], rect2[2])
    intersection_y2 = min(rect1[3], rect2[3])

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    rect1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    rect2_area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    union_area = rect1_area + rect2_area - intersection_area

    iou = intersection_area / union_area
    return iou

def calculate_tp_tn_fp_fn(gt_rect, pred_rect, image_size):
    tp = tn = fp = fn = 0
    gt_mask = np.zeros(image_size, dtype=np.uint8)
    pred_mask = np.zeros(image_size, dtype=np.uint8)
    print("GT",gt_rect)
    print("PRED",pred_rect)
    print("IMAGE SIZE",image_size)
    cv2.drawContours(gt_mask, [np.array(gt_rect)], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(pred_mask, [np.array(pred_rect)], -1, 255, thickness=cv2.FILLED)

    for y in range(image_size[0]):
        for x in range(image_size[1]):
            gt_pixel = gt_mask[y, x]
            pred_pixel = pred_mask[y, x]

            if gt_pixel == 255 and pred_pixel == 255:
                tp += 1
            elif gt_pixel == 0 and pred_pixel == 0:
                tn += 1
            elif gt_pixel == 0 and pred_pixel == 255:
                fp += 1
            elif gt_pixel == 255 and pred_pixel == 0:
                fn += 1

    return tp, tn, fp, fn

def resize_image_percent(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def precrec(image, gt_rect_coords, pred_rect_coords):
    # # Mendefinisikan path file gambar asli
    # image_path = '003.jpg'
    print("GT",gt_rect_coords)
    print("PRED",pred_rect_coords)
    # print("IMAGE",image)
    # # Membaca gambar menggunakan cv2.imread()
    # image = cv2.imread(image_path)

    # Contoh penggunaan
    image_size = image.shape[:2]  # Mengambil ukuran gambar dari shape
    # gt_rect_coords = [(0, 320), (image_size[1], 413), (image_size[1], 3293), (0, 3233)]  # Koordinat segiempat ground truth
    # pred_rect_coords = [(0, 320), (image_size[1]-100, 413), (image_size[1]-100, image_size[0]), (0, image_size[0])]  # Koordinat segiempat hasil prediksi

    tp, tn, fp, fn = calculate_tp_tn_fp_fn(gt_rect_coords, pred_rect_coords, image_size)

    # Menghitung persentase tp, tn, fp, fn
    total_pixels = image_size[0] * image_size[1]
    tp_percent = (tp / total_pixels) * 100
    tn_percent = (tn / total_pixels) * 100
    fp_percent = (fp / total_pixels) * 100
    fn_percent = (fn / total_pixels) * 100

    # Menampilkan hasil
    print("TP:", tp, "(", tp_percent, "% )")
    print("TN:", tn, "(", tn_percent, "% )")
    print("FP:", fp, "(", fp_percent, "% )")
    print("FN:", fn, "(", fn_percent, "% )")

    # Menghitung precision dan recall
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    precision_percent = precision * 100
    recall_percent = recall * 100
    print("Precision:", precision, "(", precision_percent, "% )")
    print("Recall:", recall, "(", recall_percent, "% )")

    # Menampilkan hasil pada gambar
    gt_mask = image.copy()
    pred_mask = image.copy()
    cv2.drawContours(gt_mask, [np.array(gt_rect_coords)], -1, (0,255,0), thickness=10)
    cv2.drawContours(pred_mask, [np.array(pred_rect_coords)], -1, (0,255,0), thickness=10)

    # cv2.imshow('Ground Truth', resize_image_percent(gt_mask, 30))
    # cv2.imshow('Prediction', resize_image_percent(pred_mask, 30))
    # cv2.imshow('Original Image', resize_image_percent(image, 30))  # Menampilkan gambar asli
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return gt_mask, pred_mask, tp, tn, fp, fn, precision, recall