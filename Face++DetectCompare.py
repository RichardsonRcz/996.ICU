import requests
import json
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import os
from PIL import Image
import base64


def file_base64(file_name):
    with open(file_name, 'rb') as fin:
        file_data = fin.read()
        base64_data = base64.b64encode(file_data)
    return base64_data


def faceDetect(api_key, api_secret, image_url, return_landmark, return_attributes):
    url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    param = {
        "api_key": api_key,
        "api_secret": api_secret,
        "image_url": image_url,
        "return_landmark": return_landmark,
        "return_attributes": return_attributes
    }
    data = requests.post(url=url, params=param)
    # r =  json.loads(data.content)
    return data


# 传入两张人脸进行比对
def compare_file(api_key, api_secret, imagepath_1, imagepath_2):
    url = 'https://api-cn.faceplusplus.com/facepp/v3/compare'
    image_1 = file_base64(imagepath_1)
    image_2 = file_base64(imagepath_2)
    param = {
        "api_key": api_key,
        "api_secret": api_secret,
        "image_base64_1": image_1,
        "image_base64_2": image_2,
    }
    response = requests.post(url=url, data=param)
    return response


# 传入两张人脸进行比对
def compare_url(api_key, api_secret, image_url1, image_url2):
    url = 'https://api-cn.faceplusplus.com/facepp/v3/compare'
    image_1 = file_base64('./images/1.jpg')
    image_2 = file_base64('./images/3.jpg')
    param = {
        "api_key": api_key,
        "api_secret": api_secret,
        "image_url1": image_url1,
        "image_url2": image_url2,
        # "image_base64_1": image_1,
        # "image_base64_2": image_2,
    }
    response = requests.post(url=url, data=param)
    # response = requests.post(url=url,param=param)
    return response


def main():
    api_key = "P638X3XA40Aae3kKCyzEtVoGfVwZG2fo"
    api_secret = "3HhPQEeqt1VirTIeyQFBdW0jew3sb0o_"
    # data = faceDetect("P638X3XA40Aae3kKCyzEtVoGfVwZG2fo", "3HhPQEeqt1VirTIeyQFBdW0jew3sb0o_","http://n.sinaimg.cn/sinacn18/404/w1200h804/20180509/e8fd-haichqz1022973.jpg", 1, "gender,age,smiling,glass")
    # print(data.content)
    liyifeng1 = 'http://img3.duitang.com/uploads/item/201501/14/20150114183725_TBMiS.jpeg'
    liyifeng2 = 'http://5b0988e595225.cdn.sohucs.com/images/20190917/9e30c561bb664916bdf657530e9266c4.jpeg'
    liudehua1 = 'http://n.sinaimg.cn/sinacn18/404/w1200h804/20180509/e8fd-haichqz1022973.jpg'
    liudehua2 = 'http://qimg.hxnews.com/2018/0329/1522291321463.jpeg'
    shuzu = [liyifeng1, liyifeng2, liudehua1, liudehua2]
    j = 1
    if not os.path.exists('images'):
        os.mkdir('images')
    for item in shuzu:
        # print(item)
        with open('./images/%d.jpg' % j, 'wb') as file:
            resp = requests.get(item)
            # print(resp)
            file.write(resp.content)
        lena = mpimg.imread('./images/%d.jpg' % j)
        # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
        lena.shape  # (512, 512, 3)
        plt.imshow(lena)  # 显示图片
        plt.axis('off')  # 不显示坐标轴
        # plt.show()
        j += 1
    data = compare_url(api_key, api_secret, liudehua1, liudehua2)
    #data = compare_file(api_key, api_secret, './images/1.jpg','./images/2.jpg')
    content = json.loads(data.content)
    # print((content))
    result = content['confidence']
    # print(result)
    if result > 70:
        print("数据比对成功，是同一个人!,相似度为%d" % result)
    else:
        print("相似度为%d,不确定为同一个人！" % result)

if __name__ == '__main__':
    main()
