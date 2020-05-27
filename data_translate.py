import cv2
import os

"""根据传入的目录参数，得到该目录所有子文件夹下的所有的mp4文件"""
def get_mp4path(main_dir):
    list_mp4 = []
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            #只要mp4文件
            if file.endswith(".mp4"):  
                # print(os.path.join(root, file))
                list_mp4.append(os.path.join(root, file))
    
    # 得到所有的mp4文件的列表(包含路径)
    return list_mp4 


"""根据传入的目录地址和MP4文件的名字，从视频文件中抽帧得到图片数据并保存"""
def extract_frame_from_video(path):
    #分别得到目录和文件名
    mp4_path, mp4_name = os.path.split(path)  
    cap = cv2.VideoCapture(path)
    cur_frame_cnt, pic_cnt, mod = 0, 0, 50
    while(True):
        cur_frame_cnt += 1
        ret, frame = cap.read()
        # 每隔 mod 帧保存一次
        if(cur_frame_cnt % mod == 0):       
            save_name = str(pic_cnt) + ".jpg"
            save_path = os.path.join(temp_save_path, save_name)
            if (ret):
                # 保存路径中包含中文，不能用imwrite保存，要用下一行的imencode的方法
                cv2.imwrite(save_path, frame) 
                pic_cnt += 1
                # [1]表示imencode的第二个返回值，也就是这张图片对应的内存数据
                # ret = cv2.imencode('.jpg', frame)[1].tofile(save_path)
            else:
                break

    return pic_cnt, mp4_path, mp4_name


def normalize_face_pic():  
    global current_save_path
    
    # print(current_save_path)
    image_list = os.listdir(temp_save_path)
    
    count = 1
    face_part = ""

    for image_path in image_list:
        # print(temp_save_path, image_path)
        image = cv2.imread(temp_save_path + "\\" + image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("img", gray)
        # cv2.waitKey()
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5, minSize=(5,5),)
        # 如果没有检测出人脸
        if len(faces) == 0:
            (h, w) = image.shape[:2]
            center = (w/2, h/2)
            # 将图像旋转180度
            M = cv2.getRotationMatrix2D(center, 180, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
            gray = cv2.warpAffine(gray, M, (w, h))
            faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5, minSize=(5,5),)

        for (x, y, width, height) in faces:
            print(x, y, width, height)
            face_part = gray[y: y + height, x: x + width]
            face_part = cv2.resize(face_part, (64, 64), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(current_save_path + str(count) + ".bmp", face_part)
        
        count += 1
        if count >= 6: 
            break
    
    # print(face_part, count)
    while count < 6:
        cv2.imwrite(current_save_path + str(count) + ".bmp", face_part)
        count += 1


    for old_pic in os.listdir(temp_save_path):
        path_file = os.path.join(temp_save_path, old_pic)  
        if os.path.isfile(path_file):
                os.remove(path_file)


if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    target_path = "test\\neg"
    temp_save_path = "temp_save"
    save_path = "test\\neg_picture\\"

    list_all = get_mp4path(target_path)
    # print(list_all)
    
    mp4_cnt = 1
    for element in list_all:
        current_save_path = save_path + str(mp4_cnt) + "\\"
        print("正在处理mp4文件: ", element)
        pic_cnt, mp4_path, mp4_name = extract_frame_from_video(element)
        normalize_face_pic()

        mp4_cnt += 1
        # break