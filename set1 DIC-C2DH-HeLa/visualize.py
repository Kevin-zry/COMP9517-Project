import cv2
import numpy as np


def show_video(data, fsize=(512, 512), save=False, fname=None, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(fname, fourcc, fps, fsize, isColor=False)
    out.open(fname, fourcc, fps, fsize, isColor=False)
    for i, frame in enumerate(data):
        if save:
            out.write(frame)
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
