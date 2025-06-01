import cv2
import numpy as np
import os
import glob

# Определение размеров шахматной доски
CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Создание вектора для хранения векторов трехмерных точек для каждого изображения шахматной доски
objpoints = []
# Создание вектора для хранения векторов 2D точек для каждого изображения шахматной доски
imgpoints = [] 


# Определение мировых координат для 3D точек
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Извлечение пути отдельного изображения, хранящегося в данном каталоге
images = glob.glob('./images/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Найти углы шахматной доски
    # Если на изображении найдено нужное количество углов, тогда ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    """
    Если желаемый номер угла обнаружен,
    уточняем координаты пикселей и отображаем
    их на изображениях шахматной доски
    """
    if ret == True:
        objpoints.append(objp)
        # уточнение координат пикселей для заданных 2d точек.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        
        imgpoints.append(corners2)

        # Нарисовать и отобразить углы
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    
    cv2.imshow('img',img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

h,w = img.shape[:2]

"""
Выполнение калибровки камеры с помощью
Передача значения известных трехмерных точек (объектов)
и соответствующие пиксельные координаты
обнаруженные углы (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n=== Результаты калибровки камеры ===")
print("\nМатрица камеры:")
print(np.array2string(mtx, precision=4, suppress_small=True))
print("\nКоэффициенты искажения:")
print(np.array2string(dist, precision=4, suppress_small=True))
print("\nВекторы вращения:")
for i, rvec in enumerate(rvecs):
    print(f"Изображение {i+1}:")
    print(np.array2string(rvec, precision=4, suppress_small=True))
print("\nВекторы перемещения:")
for i, tvec in enumerate(tvecs):
    print(f"Изображение {i+1}:")
    print(np.array2string(tvec, precision=4, suppress_small=True))

# Оценка ошибки репроекции
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print(f"\nСредняя ошибка репроекции: {mean_error/len(objpoints):.4f}")