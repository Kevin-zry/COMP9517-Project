import cv2
import numpy as np


a = [1,2,3,4] # 只是举个例子，比如第一帧的时候（f=1）, 点击后可以print（a[1]）

def get_output(x,y,f):
    print("here to output")
    print(x,y,f)
    #print(a[f])

    '''
        output = {
        "Speed": 0,
        "Total distance": 0,
        "Net distance": 0,
        "Confinement ratio": 0
        }'''



def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "frame %d: (%d, %d)" % (param[0], x, y)
        print(xy)
        get_output(x,y,param[0])

        #cv2.circle(img, (x, y), 1, (255, 0, 0), thickness = -1)
        #cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    #1.0, (0,0,0), thickness = 1)
        #cv2.imshow("image", img)

        print()



def show_video(data, fsize=(512, 512), save=False, fname=None, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(fname, fourcc, fps, fsize, isColor=False)
    out.open(fname, fourcc, fps, fsize, isColor=False)
    for i, frame in enumerate(data):
        if i>0:
            cv2.destroyWindow(f'frame {i-1}') 
        if save:
            out.write(frame)

        cv2.namedWindow(f'frame {i}')
        cv2.moveWindow(f'frame {i}', 100, 100) # 确定windows的位置
        # cv.SetMouseCallback(windowName, onMouse, param=None) 

        cv2.setMouseCallback(f'frame {i}', on_EVENT_LBUTTONDOWN, [i])

        cv2.imshow(f'frame {i}', frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()


def show_video_compare(data_list, fsize=(512, 512)):
    for i, imgs in enumerate(zip(*data_list)):
        resized = tuple(cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
                        for img in imgs)
        horiztonal = np.hstack(resized)
        cv2.imshow(f'frame {i}', horiztonal)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


# show_video_compare([seq1, seq2, ...])

# if __name__ == '__main__':

#     from data import get_seg_data

#     fsize = (512, 512)
#     file_name = 'DIC_seq_02.mp4'
#     seq, label = get_seg_data('ST')
#     seq_1, seq_2 = seq[:84], seq[84:]
#     show_video(seq_1)

save_path = "D:/jupyter/9517_porject/res/"
pre_data = np.load(save_path + "s1_pre.npy")
# 读取数据，是多张图片的一个合集，自行替换
# 比如84张512*512的图片，shape就是84*512*512
#print(pre_data[0])

show_video(pre_data)
