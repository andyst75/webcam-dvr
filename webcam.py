import os
import time
import argparse
from datetime import datetime

import numpy as np
import cv2

BUF_SIZE = 60

parser = argparse.ArgumentParser(description='Webcam motion recorder',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

vid = []
edges = []


def get_edges(img):
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    m = np.median(img_blur)
    if m < 5:
        th1, th2 = 10, 30
    elif m < 50:
        th1, th2 = 30, 80
    elif m < 100:
        th1, th2 = 50, 150
    else:
        th1, th2 = 100, 200
    
    edges = cv2.Canny(image=img_blur, threshold1=th1, threshold2=th2)
    return edges


def get_diff(edges):
    edges = np.vstack(edges).astype(float) / 255.
    buf = np.mean(edges, axis=0)
    cur = np.mean(edges[-3:], axis=0)
    
    diff = (((buf - cur) ** 2) ** 0.5).sum() / (cur.sum() + 1.0)
    return diff


if __name__ == "__main__":
    parser.add_argument('-webcam', type=int, default=0, help='Webcam id')
    parser.add_argument('-outdir', type=str, default='./', help='Output dir for videos')
    parser.add_argument('-height', type=int, default=720, help='Frame height')
    parser.add_argument('-width', type=int, default=1280, help='Frame width')
    parser.add_argument('-fps', type=float, default=15.0, help='FPS')
    parser.add_argument('-th', type=float, default=0.25, help='Motion threshold')
    parser.add_argument('-time', type=float, default=10.0, help='Wait time, seconds')
    parser.add_argument('-rec', type=float, default=20.0, help='Minimal record time, seconds')
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.webcam)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    time.sleep(0.1)

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'width={width}, height={height}, fps={fps}')

    time.sleep(args.time)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    ticks = 0
    while (True):
        ret, frame = capture.read()
        if ret:
            ticks += 1

            vid.append(frame)
            edge = get_edges(frame)
            edges.append(edge.flatten())

            if len(vid) < BUF_SIZE:
                continue

            if ticks % (3600 * fps) == 0:
                ticks = 0
                img_gray = cv2.cvtColor(vid[-1].copy(), cv2.COLOR_BGR2GRAY)
                img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
                m = np.median(img_blur)
                diff = get_diff(edges)
                print('check', datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), f'diff = {diff:.3f}, mean = {m:.3f}')

            if ticks % 10 == 0:
                continue

            vid   = vid[-BUF_SIZE:]
            edges = edges[-BUF_SIZE:]

            diff = get_diff(edges)

            if diff < args.th:
                continue

            ticks = 0

            print('start record at', datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
            videoWriter = cv2.VideoWriter(os.path.join(args.outdir, f"{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.mp4"), fourcc, fps, (width, height))

            for frame in vid:
                videoWriter.write(frame)

            cnt = 0
            while (True):
                ret, frame = capture.read()
                if ret:

                    videoWriter.write(frame)

                    vid.append(frame)
                    edge = get_edges(frame)
                    edges.append(edge.flatten())

                    vid   = vid[-BUF_SIZE:]
                    edges = edges[-BUF_SIZE:]

                    diff = get_diff(edges)

                    if diff >= args.th:
                        cnt = 0
                    else:
                        cnt += 1

                    if (cnt > 0) and (cnt % (10 * fps) == 0):
                        img_gray = cv2.cvtColor(vid[-1].copy(), cv2.COLOR_BGR2GRAY)
                        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
                        m = np.median(img_blur)
                        print('rec check', datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), f'diff = {diff:.3f}, mean = {m:.3f}, cnt={cnt}')

                    if (cv2.waitKey(1) == 27):
                        capture.release()
                        videoWriter.release()
                        break

                    if cnt > args.rec * fps:
                        break

            videoWriter.release()
            print('end record at', datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))

    capture.release()
