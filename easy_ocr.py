import easyocr
import cv2
import numpy as np

def easyocr_prediction(image):

    reader = easyocr.Reader(['en'], gpu=True)
    #mean = [0.485, 0.456, 0.406]
    #std_dev = [0.229, 0.224, 0.225] 
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    #image = cv2.GaussianBlur(image , (5, 5), 0)
    #image = cv2.Canny(image, 10, 200)
    #normalized_image = (image - mean) / std_dev
    #normalized_image = np.float32(normalized_image) 

    #image = cv2.resize(image, None, fx = 0.3, fy = 0.3, interpolation = cv2.INTER_AREA)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    result = reader.readtext(image, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .-')
    #print(result)
    #print("easyocr_prediction finished")
    if len(result)>0:
        return result[0][1]
    else:
        return ''


if __name__ == '__main__':
    
    img = cv2.imread("datasets/ePillID_data/classification_data/fcn_mix_weight/dc_224/1004.jpg")
    res = easyocr_prediction(img)


    print(res)
