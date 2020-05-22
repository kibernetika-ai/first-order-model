import cv2
import ml_serving.streaming.inputs.ffmpeg as sffmpeg
import ml_serving.streaming.outputs.ffmpeg as offmpeg
import ml_serving.streaming.inputs.input as input

def opencv():
    video = cv2.VideoCapture('/Users/agunin/Downloads/IMG_4596.mov')
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    vout = cv2.VideoWriter('/Users/agunin/Downloads/result.mp4', fourcc, fps, (256, 256))
    while True:
        frame_img = video.read()
        if isinstance(frame_img, tuple):
            frame_img = frame_img[1]
        if frame_img is None:
            print("Oops frame is None. Possibly camera or display does not work")
            break
        frame_img = cv2.resize(frame_img,(256,256))
        vout.write(frame_img)
    vout.release()

def rffmpeg():
    stream = sffmpeg.FFMpegInput('/Users/agunin/Downloads/IMG_4596.mov','data',max_fps=True)
    meta = stream.start()
    print(meta)
    fps = meta['fps']
    out = offmpeg.FFMpegOutput('file:///Users/agunin/Downloads/result.mp4',bgr_mode=True,fps=fps,width=256,height=256)
    #fourcc = cv2.VideoWriter_fourcc(*"avc1")
    #vout = cv2.VideoWriter('/Users/agunin/Downloads/result.mp4', fourcc, fps, (256, 256))
    try:
        while True:
            image, _ = stream.read()
            image = cv2.resize(image, (256, 256))
            out.send_output(image)
    except input.BrokenStream as e:
        print('Stream finished: {}'.format(e))
    out.close()



rffmpeg()