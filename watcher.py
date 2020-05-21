import datetime
import glob
import json
import logging
import os
import shutil
import sys
import time

LOG = logging.getLogger(__name__)
try:
    from mlboardclient.api import client

    ml_board_client = client.Client()
except:
    ml_board_client = None

meta_suff = '.__meta.json'

video_dir = os.environ['DATA_DIR']


def start_processing(config_file, job_id):
    if ml_board_client is None:
        logging.error('Unable to prepare file: ml_board client is not defined')
        return
    LOG.info('Run task %s: %s', job_id, config_file)
    app = ml_board_client.apps.get()
    task = app.task('process')
    with open(config_file, 'r') as f:
        data = json.load(f)
        data = data.get('input', {})
    video = data.get('file', None)
    if video is None:
        logging.error("No input file: {}".format(data))
    video = os.path.join(video_dir, video)
    result = os.path.join(video_dir,'serving','results',job_id+".mp4")
    task.resource('worker')[
        'command'] = f'ffmpeg -i {video} -r 30 $CODE_DIR/tmp_video/{job_id}.mp4 && python simulate.py --config config/vox-adv-256.yaml --sources $MODEL_DIR/vova.jpg --video $CODE_DIR/tmp_video/{job_id}.mp4  --relative --checkpoint $MODEL_DIR/vox-adv-cpk.pth.tar --adapt_scale --out /tmp/result.mp4 && ffmpeg -i /tmp/result.mp4 -i {video} -map 0:v -map 1:a -c:v copy -c:a aac {result}'
    task.start()
    task.update_task_info({'#serving_job': job_id,'#outputs': 'results/'+job_id+".mp4"})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    done_path = os.path.join(path, "done")
    if not os.path.exists(done_path):
        os.makedirs(done_path)
    LOG.info('Listen: %s', path)
    while True:
        time.sleep(2)
        nfiles = glob.glob(os.path.join(path, '*.json'))
        for f in nfiles:
            LOG.info("New file: %s", f)
            try:
                file_name = os.path.basename(f)
                serv_path = os.path.join(done_path, file_name)
                shutil.move(f, serv_path)
                job_id = file_name.replace(meta_suff, '')
                start_processing(serv_path, job_id)
            except:
                logging.error("Unexpected error: {}".format(sys.exc_info()))
        files = nfiles
