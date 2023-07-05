import os
import cv2
import time


from config import cfg


if __name__ == '__main__':
    subj_idx = cfg.EasyCali.subj_idx
    image_save_dir = os.path.join(cfg.EasyCali.image_save_dir, 'subject_{}'.format(subj_idx))
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    pure_black = cv2.imread(cfg.EasyCali.pure_black_path)
    pure_black.resize(720, 1280)

    start = False
    co_frames = 0

    images_to_save = []

    while cap.isOpened():

        pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey == 27:
            break
        elif pressedKey == ord('s'):
            start = True
        elif co_frames >= 50:
            start = False
            co_frames = 0

        ret, frame = cap.read()

        if start:
            cv2.imshow('Frame', cv2.flip(frame, 1))
            images_to_save.append(frame)
            co_frames += 1
            time.sleep(0.05)
        else:
            cv2.imshow('Frame', pure_black)

    cap.release()
    cv2.destroyAllWindows()

    # Save
    save_num = len(images_to_save)

    already_captured_num = len(os.listdir(image_save_dir))
    print('Before:', already_captured_num)

    strt_idx = already_captured_num
    if cfg.EasyCali.to_save:
        for i in range(save_num):
            image_idx = strt_idx + i
            image_idx = str(image_idx).zfill(3)
            cv2.imwrite(os.path.join(image_save_dir, 'subject_{}_{}.jpg'.format(subj_idx, image_idx)), images_to_save[i])

    print('Now:', len(os.listdir(image_save_dir)))

