from typing import List

from celery import shared_task

import torch
from torchaudio.backend.common import AudioMetaData
from df.enhance import enhance, load_audio, save_audio
from df.io import resample
import os
from libdf import DF
from df.model import ModelParams
from df import config
import moviepy.editor as mp
import boto3
import tempfile
import requests


config.load('config.ini')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

p = ModelParams()
df = DF(
    sr=p.sr,
    fft_size=p.fft_size,
    hop_size=p.hop_size,
    nb_bands=p.nb_erb,
    min_nb_erb_freqs=p.min_nb_freqs,
)

print("Device - ", DEVICE)
model = torch.load(("celery_tasks/model.pth"), map_location=torch.device('cpu'))
model.to(DEVICE)
model.eval()


@shared_task(bind=True,autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
            name='video:processing_video_task')
def process_vid(self, video_url: List[str]):
    with tempfile.TemporaryDirectory(dir = ".") as tmpdir:
        try:
            os.chdir(tmpdir)
            print(tmpdir)
            s3 = boto3.resource(
                service_name='s3',
                region_name='eu-north-1',
                aws_access_key_id='AKIA2R3K4L6H6KOG35F6',
                aws_secret_access_key='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'
            )
            input_vid = "input.webm"
            output_file = "tmp.wav"
            response = requests.get(video_url)
            with open(input_vid, 'wb') as f:
                f.write(response.content)
            
            # Convert the video to .mp4
            video = mp.VideoFileClip(input_vid)
            video.audio.write_audiofile(output_file.replace('.wav', '.mp3'))
            audio = video.audio
            audio.write_audiofile(output_file)
            video.close()
            audio.close()
            os.remove(input_vid)
            os.remove(output_file.replace('.wav', '.mp3'))

            print("File downloaded!")
            wav_file = "tmp.wav"
            print("Wav stored.")
            meta = AudioMetaData(-1, -1, -1, -1, "")
            sr = config("sr", 48000, int, section="df")
            sample, meta = load_audio(wav_file, sr)
            len_audio = (meta.num_frames/meta.sample_rate)/60
            max_min = 1
            if len_audio  % max_min < 0.1:
                num_chunks = len_audio // max_min
            else:
                num_chunks = len_audio // max_min + 1
            print(f"Total length of audio = {len_audio} chunks = {num_chunks}")
            estimate = []
            split_tensors = torch.tensor_split(sample, int(num_chunks), dim = 1)
            for i in range(len(split_tensors)):
                enhanced = enhance(model, df, split_tensors[i])
                enhanced = enhance(model, df, enhanced)
                lim = torch.linspace(0.0, 1.0, int(sr * 0.15)).unsqueeze(0)
                lim = torch.cat((lim, torch.ones(1, enhanced.shape[1] - lim.shape[1])), dim=1)
                enhanced = enhanced * lim
                enhanced = resample(enhanced, sr, meta.sample_rate)
                estimate.append(enhanced)
            estimate = tuple(estimate)
            enhanced = torch.cat(estimate, dim = -1)
            sr = meta.sample_rate
            save_audio("enhanced_aud.wav", enhanced, sr)
            print("Uploading to s3")
            response = s3.Bucket("denoise-async").upload_file(Key="outputs/enhanced.wav", Filename="enhanced_aud.wav")
            print("Uploaded to s3!")
        except Exception as e:
            print("Exception - ", e)
            return {"Status": "Failure", "Error": e}
    return {"Status": "Success"}
