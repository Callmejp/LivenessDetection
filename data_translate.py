import cv2
import os

from Config import IMG_SIZE


"""
根据传入的目录参数，得到该目录所有子文件夹下的所有的mp4文件
"""
def get_mp4path(main_dir):
    list_mp4 = []
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            # 只要mp4文件
            if file.endswith(".mp4"):  
                # print(os.path.join(root, file))
                list_mp4.append(os.path.join(root, file))
    
    # 得到所有的mp4文件的列表(包含路径)
    return list_mp4 


"""
根据传入的MP4文件的路径，从视频文件中抽帧得到图片数据并保存
"""
def extract_frame_from_video(path):
    global current_save_path
 
    # 通过opencv将视频读取到内存中 
    cap = cv2.VideoCapture(path)
    cur_frame_cnt, pic_cnt, mod = 0, 0, 6
    while(True):
        cur_frame_cnt += 1
        ret, frame = cap.read()
        # 每隔 mod 帧保存一次
        if(cur_frame_cnt % mod == 0):       
            save_name = str(pic_cnt) + ".jpg"
            save_path = os.path.join(current_save_path, save_name)
            if ret:
                cv2.imwrite(save_path, frame) 
                pic_cnt += 1
            else:
                break
    
    # 总共抽出的图片数
    return pic_cnt


def normalize_face_pic():  
    global current_save_path
    
    image_list = os.listdir(current_save_path)
    image_cnt = 0
    for image_path in image_list:
        image_path = current_save_path + "\\" + image_path
        image = cv2.imread(image_path)
        # 直接删除原来的视频帧
        os.remove(image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("img", gray)
        # cv2.waitKey()
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5, minSize=(5,5),)
        # 如果没有检测出人脸(因为有些图片是倒转的)
        if len(faces) == 0:
            (h, w) = image.shape[:2]
            center = (w/2, h/2)
            # 将图像旋转180度
            M = cv2.getRotationMatrix2D(center, 180, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
            gray = cv2.warpAffine(gray, M, (w, h))
            faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5, minSize=(5,5),)

        
        for (x, y, width, height) in faces:
            # print(x, y, width, height)
            face_part = gray[y: y + height, x: x + width]
            face_part = cv2.resize(face_part, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(current_save_path + "\\" + str(image_cnt) + ".bmp", face_part)
            # 一张图片里应该只有一个人脸
            # assert(len(faces) == 1)
            image_cnt += 1
            break
        
       


if __name__ == '__main__':
    # used to recognize face in the picture
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # file path prefix
    file_path_prefix = ["train", "test"]
    # sample classification
    sample_classification = ["neg", "pos"]
    # target_path = "test\\neg"
    # temp_save_path = "temp_save"
    # save_path = "test\\neg_picture\\"

    """
    Expected Directory:
    data_translate.py
    train/
        pos_picture/
        pos/
            0.mp4
            ...
        neg_picture/
        neg/
            0.mp4
            ...
    test/
        pos_picture/
        pos/
            0.mp4
            ...
        neg_picture/
        neg/
            0.mp4
            ...
    """
    for prefix in file_path_prefix:
        for classification in sample_classification:
            # e.g. train\\pos
            target_path = prefix + "\\" + classification
            # e.g. train\\pos_picture
            save_path_prefix = target_path + "_picture"

            all_mp4_full_path = get_mp4path(target_path)
            print(all_mp4_full_path)
            
            
            mp4_index = 1
            for element in all_mp4_full_path:
                # e.g. train\\pos_picture\\1
                current_save_path = save_path_prefix + "\\" + str(mp4_index)
                os.makedirs(current_save_path)

                print("正在处理mp4文件: ", element)
                pic_cnt = extract_frame_from_video(element)
                print("共得到 ", pic_cnt, "个图片")
                # 去掉其余部分，只要人脸
                normalize_face_pic()

                mp4_index += 1

            # exit()
                