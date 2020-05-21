import cv2


import imageio
import yaml
from argparse import ArgumentParser

import numpy as np
from skimage.transform import resize
import requests
import logging
import time
import os


LOG = logging.getLogger(__name__)

import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(
        **config["model_params"]["generator_params"],
        **config["model_params"]["common_params"],
    )
    if cpu:
        generator.cpu()
    else:
        generator.cuda()

    kp_detector = KPDetector(
        **config["model_params"]["kp_detector_params"],
        **config["model_params"]["common_params"],
    )
    if cpu:
        kp_detector.cpu()
    else:
        kp_detector.cuda()

    checkpoint = torch.load(checkpoint_path, map_location="cpu" if cpu else None)
    generator.load_state_dict(checkpoint["generator"])
    kp_detector.load_state_dict(checkpoint["kp_detector"])

    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def headers(token):
    ret = None
    if token:
        ret = {"Authorization": "Bearer {}".format(token)}
    return ret


def fetch_task(opt):
    while True:
        resp = requests.get(
            opt.master + f"/servings/{opt.id}/task", headers=headers(opt.token)
        )
        if resp.status_code in [204, 404]:
            time.sleep(5)
            continue
        if resp.status_code != 200:
            LOG.error("Response {}".format(resp.text))
            raise Exception("Failed fetch task")
        return resp.json()


def process(opt, generator, kp_detector):
    while True:
        # {'task_id': 'f548a28e-70ec-4756-aaad-ba8cf6ba6baa', 'percent': 0, 'state': 'executing', 'params': {
        #    'dst': {'filename': 'BB398819-0799-487A-829B-DD5D0C7449F7/f548a28e-70ec-4756-aaad-ba8cf6ba6baa/dst.jpg',
        #            'name': 'dst.jpg'},
        #    'src': {'filename': 'BB398819-0799-487A-829B-DD5D0C7449F7/f548a28e-70ec-4756-aaad-ba8cf6ba6baa/src.mov',
        #            'name': 'source.mov'}}}
        task = fetch_task(opt)
        task_id = task.get("task_id", "")
        if task_id == "":
            LOG.error("Got empty task id")
            continue

        try:
            task_dir, img, video = check_task(task)
            out_file = os.path.join(opt.dst_dir, task_dir, "result", "result.mp4")
            video = os.path.join(opt.src_dir, video)
            img = os.path.join(opt.src_dir, img)
            img = imageio.imread(img)
            process_task(task_id, opt, img, video, out_file, generator, kp_detector)

        except Exception as e:
            LOG.error(f"Task {task_id} process error: {str(e)}")
            send_status(opt, task_id, state="failed")
            continue



def check_task(task):
    params = task.get("params", {})
    LOG.info("params", params)
    task_dir = task.get("task_dir", "")
    if task_dir == "":
        raise ValueError("Got empty task dir")
    img = params.get("dst", {}).get("filename", "")
    LOG.info("img", img)
    if img == "":
        raise ValueError("Got empty dest image")
    video = params.get("src", {}).get("filename", "")
    if video == "":
        raise ValueError("Got empty src video")
    return task_dir, img, video


def process_task(task_id, opt, img_orig, video_file, out_file, generator, kp_detector):
    img = resize(img_orig, (256, 256))[..., :3]
    kp_driving_initial = None
    with torch.no_grad():
        spm = torch.tensor(img[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = spm.cpu() if opt.cpu else spm.cuda()
        kp_source = kp_detector(source)

        video = cv2.VideoCapture(video_file)
        fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        vout = cv2.VideoWriter(out_file, fourcc, fps, (256, 256))
        frames_count = int(cv2.VideoCapture.get(video, cv2.CAP_PROP_FRAME_COUNT))
        frame_counter = 0
        last_percent = 0

        while True:
            frame_img = video.read()
            if isinstance(frame_img, tuple):
                frame_img = frame_img[1]
            if frame_img is None:
                print("Oops frame is None. Possibly camera or display does not work")
                break

            # frame_img = cv2.rotate(frame_img,cv2.ROTATE_180)
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

            y, x, _ = frame_img.shape
            min_dim = min(y, x)
            startx = x // 2 - (min_dim // 2)
            starty = y // 2 - (min_dim // 2)
            frame_img = frame_img[
                starty : starty + min_dim, startx : startx + min_dim, :
            ]

            frame_img = resize(frame_img, (256, 256))[..., :3]
            frame = torch.tensor(frame_img[np.newaxis].astype(np.float32)).permute(
                0, 3, 1, 2
            )

            kp_driving = kp_detector(frame)
            if kp_driving_initial is None:
                kp_driving_initial = kp_driving
            kp_norm = normalize_kp(
                kp_source=kp_source,
                kp_driving=kp_driving,
                kp_driving_initial=kp_driving_initial,
                use_relative_movement=opt.relative,
                use_relative_jacobian=opt.relative,
                adapt_movement_scale=opt.adapt_scale,
            )
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            p = np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0]
            p = p * 255
            p = p.astype(np.uint8)
            p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
            vout.write(p)

            frame_counter += 1

            percent = int(frame_counter / frames_count * 100)
            if percent != last_percent:
                last_percent = percent
                send_status(opt, task_id, percent=min(100, percent))

        video.release()
        vout.release()
        send_status(opt, task_id, state="completed")


def send_status(opt, task_id, state="executing", percent=0):
    resp = requests.post(
        opt.master + f"/tasks/{task_id}",
        json={"state": state, "percent": percent,},
        headers=headers(opt.token),
    )
    if resp.status_code != 200:
        LOG.error("Failed update task state: {}".format(resp.text))


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument(
        "--checkpoint", default="vox-cpk.pth.tar", help="path to checkpoint to restore"
    )

    parser.add_argument("--src_dir", default="/inputs")
    parser.add_argument("--dst_dir", default="/outputs")
    parser.add_argument("--token", default="")
    parser.add_argument("--master", default="")
    parser.add_argument("--id", default="test")

    parser.add_argument(
        "--relative",
        dest="relative",
        action="store_true",
        help="use relative or absolute keypoint coordinates",
    )
    parser.add_argument(
        "--adapt_scale",
        dest="adapt_scale",
        action="store_true",
        help="adapt movement scale based on convex hull of keypoints",
    )

    parser.add_argument(
        "--cpu", dest="cpu", action="store_true", help="CPU only (no CUDA used)"
    )

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()
    LOG.error("Loading checkpoints...")
    generator, kp_detector = load_checkpoints(
        config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu
    )
    LOG.error("Start processing...")
    process(opt, generator, kp_detector)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
