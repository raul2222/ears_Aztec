
#nvidia-smi -L

print('/content/drive')

import IPython

#Token de autenticación de Google Drive
from google.colab import drive
drive.mount('drive', force_remount=True)


!pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
!pip install pydub
!pip install phonemizer
!pip install ffmpeg-normalize
!pip install git+https://github.com/wkentaro/gdown.git
!git clone -q https://github.com/rmcpantoja/tacotron2.git
!git clone -q --recursive https://github.com/justinjohn0306/hifi-gan
!pip install git+https://github.com/savoirfairelinux/num2words
!git submodule init
!git submodule update
!pip install matplotlib numpy inflect librosa scipy unidecode pillow tensorboardX resampy
!apt-get install pv
!apt-get -qq install sox
!apt-get install jq
!wget https://raw.githubusercontent.com/tonikelope/megadown/master/megadown -O megadown.sh
!chmod 755 megadown.sh










from pydub import AudioSegment
import os
from os.path import exists
import sys

sys.path.append('tacotron2')
# %cd content/tacotron2

sys.path.append('hifi-gan')


# %matplotlib inline

import IPython.display as ipd
import json
from layers import TacotronSTFT
from audio_processing import griffin_lim
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import Generator
from denoiser import Denoiser
import resampy
import scipy.signal

import os
if os.getcwd() != '/content/tacotron2':
    os.chdir('tacotron2')
import time
import argparse
import math
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams
 
import random
import numpy as np

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence
from math import e
#from tqdm import tqdm # Terminal
#from tqdm import tqdm_notebook as tqdm # Legacy Notebook TQDM
from tqdm.notebook import tqdm # Modern Notebook TQDM
from distutils.dir_util import copy_tree
import matplotlib.pylab as plt

# %cd /content/
def get_hifigan(MODEL_ID, conf_name):
    # Download HiFi-GAN
    hifigan_pretrained_model = 'hifimodel_' + conf_name
    #gdown.download(d+MODEL_ID, hifigan_pretrained_model, quiet=False)

    if MODEL_ID == "universal":
      !wget "https://github.com/johnpaulbin/tacotron2/releases/download/Main/g_02500000" -O $hifigan_pretrained_model
    else:
      !gdown --id "$MODEL_ID" -O $hifigan_pretrained_model

    # Load HiFi-GAN
    conf = os.path.join("hifi-gan", conf_name + ".json")
    with open(conf) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    hifigan = Generator(h).to(torch.device("cuda"))
    state_dict_g = torch.load(hifigan_pretrained_model, map_location=torch.device("cuda"))
    hifigan.load_state_dict(state_dict_g["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    denoiser = Denoiser(hifigan, mode="normal")
    return hifigan, h, denoiser
 
# Download character HiFi-GAN
hifigan, h, denoiser = get_hifigan("universal", "config_v1")
# Download super-resolution HiFi-GAN
hifigan_sr, h2, denoiser_sr = get_hifigan("14fOprFAIlCQkVRxsfInhEPG0n-xN4QOa", "config_32k")

# %cd /content/tacotron2

def download_from_google_drive(file_id, file_name):
  # download a file from the Google Drive link
  !rm -f ./cookie
  !curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id={file_id}" > /dev/null
  confirm_text = !awk '/download/ {print $NF}' ./cookie
  confirm_text = confirm_text[0]
  !curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm={confirm_text}&id={file_id}" -o {file_name}

def create_mels():
    print("Generating Mels")
    stft = layers.TacotronSTFT(
                hparams.filter_length, hparams.hop_length, hparams.win_length,
                hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                hparams.mel_fmax)
    def save_mel(filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != stft.sampling_rate:
            raise ValueError("{} {} SR does not match the objective {} SR".format(filename, 
                sampling_rate, stft.sampling_rate))
        audio_norm = audio / hparams.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0).cpu().numpy()
        np.save(filename.replace('.wav', ''), melspec)

    import glob
    wavs = glob.glob('wavs/*.wav')
    for i in tqdm(wavs):
        save_mel(i)


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    try:
        torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)
    except KeyboardInterrupt:
        print("interrupt received while saving, waiting for save to complete.")
        torch.save({'iteration': iteration,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict(),'learning_rate': learning_rate}, filepath)
    print("Model Saved")

def plot_alignment(alignment, info=None):
#     %matplotlib inline
    fig, ax = plt.subplots(figsize=(int(alignment_graph_width/100), int(alignment_graph_height/100)))
    im = ax.imshow(alignment, cmap='inferno', aspect='auto', origin='lower',
                   interpolation='none')
    ax.autoscale(enable=True, axis="y", tight=True)
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    fig.canvas.draw()
    plt.show()

def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank, epoch, start_eposh, learning_rate, sample_interbal, save_audio = False, audio_path = None):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Epoch: {} Validation loss {}: {:9f}  Time: {:.1f}m LR: {:.6f}".format(epoch, iteration, val_loss,(time.perf_counter()-start_eposh)/60, learning_rate))
        logger.log_validation(val_loss, model, y, y_pred, iteration)
        if hparams.show_alignments:
#             %matplotlib inline
            _, mel_outputs, gate_outputs, alignments = y_pred
            idx = random.randint(0, alignments.size(0) - 1)
            plot_alignment(alignments[idx].data.cpu().numpy().T)

    dv = epoch/sample_interbal
    if dv.is_integer():
      print(f"Generación de muestras... \n{sampletext}")
      for i in [x for x in sampletext.split("\n") if len(x)]:
          if i[-1] != ";": i=i+";" 
          with torch.no_grad():
              sequence = np.array(text_to_sequence(i, ['basic_cleaners']))[None, :]
              sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
              mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
              y_g_hat = hifigan(mel_outputs_postnet.float())
              audio = y_g_hat.squeeze()
              audio = audio * MAX_WAV_VALUE
              audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
 
              # Resample to 32k
              audio_denoised = audio_denoised.cpu().numpy().reshape(-1)
 
              normalize = (MAX_WAV_VALUE / np.max(np.abs(audio_denoised))) ** 0.9
              audio_denoised = audio_denoised * normalize
              wave = resampy.resample(
                  audio_denoised,
                  h.sampling_rate,
                  h2.sampling_rate,
                  filter="sinc_window",
                  window=scipy.signal.windows.hann,
                  num_zeros=8,
              )
              wave_out = wave.astype(np.int16)
 
              # HiFi-GAN super-resolution
              wave = wave / MAX_WAV_VALUE
              wave = torch.FloatTensor(wave).to(torch.device("cuda"))
              new_mel = mel_spectrogram(
                  wave.unsqueeze(0),
                  h2.n_fft,
                  h2.num_mels,
                  h2.sampling_rate,
                  h2.hop_size,
                  h2.win_size,
                  h2.fmin,
                  h2.fmax,
              )
              y_g_hat2 = hifigan_sr(new_mel)
              audio2 = y_g_hat2.squeeze()
              audio2 = audio2 * MAX_WAV_VALUE
              audio2_denoised = denoiser(audio2.view(1, -1), strength=35)[:, 0]
                  
              # High-pass filter, mixing and denormalizing
              audio2_denoised = audio2_denoised.cpu().numpy().reshape(-1)
              b = scipy.signal.firwin(
                  101, cutoff=10500, fs=h2.sampling_rate, pass_zero=False
              )
              y = scipy.signal.lfilter(b, [1.0], audio2_denoised)
              y *= 0
              y_out = y.astype(np.int16)
              y_padded = np.zeros(wave_out.shape)
              y_padded[: y_out.shape[0]] = y_out
              sr_mix = wave_out + y_padded
              sr_mix = sr_mix / normalize

              print("")
              ipd.display(ipd.Audio(sr_mix.astype(np.int16), rate=h2.sampling_rate))
              if save_audio:
                if not os.path.isdir(audio_path+"/audio samples"):
                  !os.makedirs(audio_path+"/audio samples")
                scipy.io.wavfile.write(audio_path+"/audio samples/_"+epoch+"test.wav", h2.sampling_rate, sr_mix.astype(np.int16))
                wav2mp3(audio_path+"/audio samples/_"+epoch+"test.wav")

def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams, log_directory2):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    svcount = 0
    epoch_offset = 0
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))
    elif model_base != "Ninguno":
        raise FileNotFoundError(f"Modelo '{checkpoint_path}' no se encuentra.")
    
    start_eposh = time.perf_counter()
    learning_rate = 0.0
    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in tqdm(range(epoch_offset, hparams.epochs)):
        print("\nStarting Epoch: {} Iteration: {}".format(epoch, iteration))
        start_eposh = time.perf_counter() # eposh is russian, not a typo
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            start = time.perf_counter()
            if iteration < hparams.decay_start: learning_rate = hparams.A_
            else: iteration_adjusted = iteration - hparams.decay_start; learning_rate = (hparams.A_*(e**(-iteration_adjusted/hparams.B_))) + hparams.C_
            learning_rate = max(hparams.min_learning_rate, learning_rate) # output the largest number
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)
                #print("Batch {} loss {:.6f} Grad Norm {:.6f} Time {:.6f}".format(iteration, reduced_loss, grad_norm, duration), end='\r', flush=True)

            iteration += 1
        validate(model, criterion, valset, iteration,
                 hparams.batch_size, n_gpus, collate_fn, logger,
                 hparams.distributed_run, rank, epoch, start_eposh, learning_rate, sample_interbal, save_audio, output_directory)
        svcount += 1
        if svcount == saving_interval:
          svcount = 0
          save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path)
        if log_directory2 != None:
            copy_tree(log_directory, log_directory2)


def wav2mp3(path_to_file):
    final_audio = AudioSegment.from_wav(file=path_to_file)
    path_to_file = path_to_file.replace(".wav",".mp3")
    final_audio.export(path_to_file, format="mp3")
    return path_to_file

def check_dataset(hparams):
    from utils import load_wav_to_torch, load_filepaths_and_text
    import os
    import numpy as np
    def check_arr(filelist_arr):
        for i, file in enumerate(filelist_arr):
            if len(file) > 2:
                print("|".join(file), " tiene múltiplos '|', esto puede no ser un error.\n")
            if hparams.load_mel_from_disk and '.wav' in file[0]:
                print("\033[31m\033[1m[AVISO]", file[0], " en la lista de archivos mientras que lo esperado era .npy\n")
            else:
                if not hparams.load_mel_from_disk and '.npy' in file[0]:
                    print("\033[31m\033[1m[AVISO]", file[0], " en la lista de archivos mientras que lo esperado era .wav\n")
            if (not os.path.exists(file[0])):
                raise FileNotFoundError(f"\'{'|'.join(file)}\' no existe. Compruebe su transcripción y sus audios.")
            if len(file[1]) < 3:
                print(f"\033[34m\033[1m[info]{'|'.join(file)} sin texto o demasiado corto.\n")
            if not ((file[1].strip())[-1] in r"!?,.;:"):
                print(f"\033[34m\033[1m[info]{'|'.join(file)} no tiene puntuación final.\n")
            mel_length = 1
            if hparams.load_mel_from_disk and '.npy' in file[0]:
                melspec = torch.from_numpy(np.load(file[0], allow_pickle=True))
                mel_length = melspec.shape[1]
            if mel_length == 0:
                print(f"\033[33m\033[1m[AVISO]{'|'.join(file)} has 0s duration.\n")

    print("Comprobación de los archivos de formación")
    audiopaths_and_text = load_filepaths_and_text(hparams.training_files) # get split lines from training_files text file.
    check_arr(audiopaths_and_text)
    print("Comprobación de los archivos de formación")
    audiopaths_and_text = load_filepaths_and_text(hparams.validation_files) # get split lines from validation_files text file.
    check_arr(audiopaths_and_text)
    print("\033[32m\033[1mComprobación realizada")
n_gpus=1
rank=0
group_name=None

# ---- DEFAULT PARAMETERS DEFINED HERE ----
hparams = create_hparams()
model_filename = 'current_model'
hparams.training_files = "filelists/clipper_train_filelist.txt"
hparams.validation_files = "filelists/clipper_val_filelist.txt"
#hparams.use_mmi=True,          # not used in this notebook
#hparams.use_gaf=True,          # not used in this notebook
#hparams.max_gaf=0.5,           # not used in this notebook
#hparams.drop_frame_rate = 0.2  # not used in this notebook
hparams.p_attention_dropout=0.1
hparams.p_decoder_dropout=0.1
hparams.decay_start = 15000
hparams.A_ = 5e-4
hparams.B_ = 8000
hparams.C_ = 0
hparams.min_learning_rate = 1e-5
generate_mels = True
hparams.show_alignments = True
alignment_graph_height = 600
alignment_graph_width = 1000
hparams.batch_size = 32
hparams.load_mel_from_disk = True
hparams.epochs = 10000
hparams.sampling_rate = 22050
hparams.max_decoder_steps = 3000 # Max Duration
hparams.gate_threshold = 0.5 # Model must be 50% sure the clip is over before ending generation
torch.backends.cudnn.enabled = hparams.cudnn_enabled
torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
output_directory = '/content/drive/My Drive/colab/outdir' # Location to save Checkpoints
log_directory = '/content/tacotron2/logs' # Location to save Log files locally
log_directory2 = '/content/drive/My Drive/colab/logs' # Location to copy log files (done at the end of each epoch to cut down on I/O)
checkpoint_path = output_directory+(r'/')+model_filename
sample_interbal = 5
save_audio = False
# ---- Replace .wav with .npy in filelists ----
!sed -i -- 's,.wav|,.npy|,g' filelists/*.txt
!sed -i -- 's,.wav|,.npy|,g' {hparams.training_files}
!sed -i -- 's,.wav|,.npy|,g' {hparams.validation_files}
# ---- Replace .wav with .npy in filelists ----

# %cd /content/tacotron2

data_path = 'wavs'
!mkdir {data_path}

# Commented out IPython magic to ensure Python compatibility.
#@markdown ## <font color="pink"> **3. Cargar el conjunto de datos.** 📥
#@markdown ---
#@markdown #### Los audios pueden estar comprimidos en un archivo **ZIP** (recomendado) o sueltos.
#@markdown #### También puedes subir manualmente el ZIP/carpeta e introducir la ruta en el campo de abajo, o importar tus audios desde Drive introduciendo la ruta en la plantilla "/content/drive/My Drive/tu_archivo". Si quieres **subir en la propia celda, deja el campo vacío**.
#@markdown ---
#@markdown #### ¿Habilitar procesamiento de audios?
#@markdown Recuerda que estos deberán tener un formato compatible, es decir, frecuencia de muestreo <font color="orange">**22050, 16 bit, mono**</font>. Si no tienes los audios con este formato, activa esta casilla para hacer la conversión, a parte de normalización y eliminación de silencios.
audio_processing = True #@param {type:"boolean"}
drive_path = "/content/drive/MyDrive/papa/audio" #@param {type: "string"}

from google.colab import files
import zipfile
import os
import wave
import shutil
import datetime

if os.listdir('/content/tacotron2/wavs/'):
  !rm /content/tacotron2/wavs/*
with open('/content/insert_silences.py', 'w') as silscript:
    silscript.write('''\
import argparse
import os
from tqdm import tqdm
from pydub import AudioSegment
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script para agregar silencios'
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        required=True,
        help="EL directorio de entrada con el que trabajar los silencios",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=False,
        help="La carpeta de salida donde se guardarán los audios modificados",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Modo silencioso"
    )
    args = parser.parse_args()
# definir la ruta de la carpeta de entrada y salida
carpeta_entrada = args.input_dir
if not carpeta_entrada:
    carpeta_entrada = "/content/audios/"
carpeta_salida = args.output_dir
if not carpeta_salida:
    carpeta_salida = "/content/audios_nuevos/"
def convert(input_dir, output_dir, quiet):
    # crear la carpeta de salida si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    # verificar que hay archivos en la carpeta de entrada:
    valid_extensions = ('.wav', '.mp3', '.ogg')
    files = [f for f in os.listdir(input_dir) if f.endswith(valid_extensions)]
    assert len(files) > 0, "La carpeta de entrada no tiene archivos con los que trabajar"
    # recorrer todos los archivos de la carpeta de entrada
    for archivo in tqdm(files):
        # cargar el archivo de audio
        ruta_archivo = os.path.join(input_dir, archivo)
        audio = AudioSegment.from_file(ruta_archivo)
        # agregar 0.5 segundos al final del archivo
        nuevo_audio = audio.append(AudioSegment.silent(duration=500))
        # guardar el nuevo archivo de audio en la carpeta de salida
        ruta_nuevo_archivo = os.path.join(output_dir, archivo)
        #ruta_nuevo_archivo = output_dir + archivo
        nuevo_audio.export(ruta_nuevo_archivo, format="wav")
    if not quiet:
        print("Se agregaron 0.5 segundos a todos los archivos de audio en la carpeta de entrada y se guardaron los nuevos archivos en la carpeta de salida.")
convert(carpeta_entrada, carpeta_salida, args.quiet)
''')
with open('/content/audios.sh', 'w') as rsh:
    rsh.write('''\
for file in /content/tacotron2/wavs/*.wav
do
    ffmpeg -y -i "$file" -ar 22050 /content/tempwav/srtmp.wav -loglevel error
    sox /content/tempwav/srtmp.wav  -c 1 /content/tempwav/ntmp.wav norm -0.1
    sox /content/tempwav/ntmp.wav /content/tempwav/ctmp.wav silence 1 0.05 0.1% reverse silence 1 0.05 0.1% reverse
    ffmpeg -y -i /content/tempwav/ctmp.wav -c copy -fflags +bitexact -flags:v +bitexact -flags:a +bitexact -ar 22050 /content/tempwav/poop.wav -loglevel error
    rm "$file"
    mv /content/tempwav/poop.wav "$file"
    rm /content/tempwav/*
done
python "/content/insert_silences.py" --input_dir "/content/tacotron2/wavs" --output_dir "/content/tacotron2/newwavs" --quiet
mv /content/tacotron2/newwavs/*.wav /content/tacotron2/wavs
''')

# %cd /content/tacotron2/wavs

drive_path = drive_path.strip()

if drive_path:
  if os.path.exists(drive_path):
    print(f"\n\033[34m\033[1mAudio importado de Drive.\n\033[90m")
    if zipfile.is_zipfile(drive_path):
      !unzip -q -j "$drive_path" -d /content/tacotron2/wavs

    else:
      fp = drive_path + "/."
      !cp -a "$fp" "/content/tacotron2/wavs"
  else:
    print(f"\n\033[33m\033[1m[AVISO] La ruta {drive_path} no se encuentra, compruebe los errores e inténtelo de nuevo.")
    print(f"\n\033[34m\033[1mCargue su conjunto de datos(audios)...")
    uploaded = files.upload()
else:
  print(f"\n\033[34m\033[1mCargue su conjunto de datos(audios)...")
  uploaded = files.upload()

  for fn in uploaded.keys():
    if zipfile.is_zipfile(fn):
      !unzip -q -j "$fn" -d /content/tacotron2/wavs
      !rm "$fn"

if os.path.exists("/content/tacotron2/wavs/wavs"):
    for file in os.listdir("/content/tacotron2/wavs/wavs"):
      !mv /content/tacotron2/wavs/wavs/"$file"  /content/tacotron2/wavs/"$file"
!rm /content/tacotron2/wavs/list.txt
if audio_processing:
  print(f"\n\033[37mNormalización, eliminación de metadatos y comprobación de audios...")
  !mkdir /content/tempwav
  !bash /content/audios.sh

totalduration = 0
try:
  for file_name in [x for x in os.listdir() if os.path.isfile(x)]:
      with wave.open(file_name, "rb") as wave_file:
          frames = wave_file.getnframes()
          rate = wave_file.getframerate()
          duration = frames / float(rate)
          totalduration += duration

          if duration >= 15.58:
            print(f"\n\033[33m\033[1m[AVISO] {file_name} es superior a 15 segundos. La falta de RAM puede" 
                  " ¡se producen en un lote de gran tamaño!")
except:
  print(file_name)
wav_count = len(os.listdir("/content/tacotron2/wavs"))
print(f"\n{wav_count} audios procesados. duración total: {str(datetime.timedelta(seconds=round(totalduration, 0)))}\n")

#shutil.make_archive("/content/processedwavs", 'zip', '/content/tacotron2/wavs')
#files.download('/content/processedwavs.zip')

print("\n\033[32m\033[1mTodo listo, proceda.")

# Commented out IPython magic to ensure Python compatibility.
#@markdown ## <font color="pink"> **4. Sube la transcripción.** 📝
#@markdown ---
#@markdown La transcripción debe ser un archivo ``.TXT`` formateado en <font color="red" size="+3"> ``UTF-8 sin BOM.``
# %cd /content/tacotron2/filelists/
!rm /content/tacotron2/filelists/list.txt

print("\n\033[34m\033[1mSube tu expediente académico(list)...")
listfn, length = files.upload().popitem()

if listfn != "list.txt":
  !mv "$listfn" list.txt
# %cd /content/tacotron2/
print("\n\033[32m\033[1mTodo listo, proceda.")

import os
import os.path
from pathlib import Path
from itertools import chain
from glob import glob
#@markdown ## <font color="pink"> **5. Configurar los parámetros del modelo.** 🎛️
#@markdown ---
#@markdown ####  Nombre deseado para el modelo:
model_filename = "raulv1" #@param {type: "string"}
Training_file = "/content/tacotron2/filelists/list.txt"
hparams.training_files = Training_file
hparams.validation_files = Training_file
# hparams to Tune
#hparams.use_mmi=True,          # not used in this notebook
#hparams.use_gaf=True,          # not used in this notebook
#hparams.max_gaf=0.5,           # not used in this notebook
#hparams.drop_frame_rate = 0.2  # not used in this notebook
hparams.p_attention_dropout=0.1
hparams.p_decoder_dropout=0.1

hparams.B_ = 8000                   # Decay Rate
hparams.C_ = 0                      # Shift learning rate equation by this value
hparams.min_learning_rate = 1e-5    # Min Learning Rate

# Quality of Life
generate_mels = True
hparams.show_alignments = True
alignment_graph_height = 600
alignment_graph_width = 1000

#@markdown ---
#@markdown #### Batch size. Marque la casilla si desea que el código calcule el valor óptimo automáticamente:
auto_batch_size = True #@param{type:"boolean"}
hparams.batch_size =   6#@param {type: "integer"}

if auto_batch_size:
  import subprocess
  gpu_check = subprocess.check_output("nvidia-smi -L", shell=True)
  if "K80" in str(gpu_check):
      hparams.batch_size = int(wav_count/48)
  else:
      hparams.batch_size = int(wav_count/32)
  print(f"batch size ajustado a {hparams.batch_size}")

#@markdown ---
#@markdown #### Calcular automáticamente la tasa de aprendizaje ideal: (Recomendado)
# Learning Rate             # https://www.desmos.com/calculator/ptgcz4vzsw / http://boards.4channel.org/mlp/thread/34778298#p34789030
hparams.decay_start = 15000         # wait till decay_start to start decaying learning rate
autocalculate_learning_rate = True #@param {type:"boolean"}

if autocalculate_learning_rate:
  hparams.A_ = 0.001*(hparams.batch_size/256)**0.5 # Start/Max Learning Rate
  print(f"Learning Rate definido como {hparams.A_}")
else:
  hparams.A_ = 3e-4 #@param ["3e-6", "1e-5", "1e-4", "5e-4", "1e-3"] {type:"raw", allow-input: true}              

hparams.load_mel_from_disk = True
hparams.text_cleaners=["basic_cleaners"]
hparams.ignore_layers=["embedding.weight"]
# Layers to reset (None by default, other than foreign languages this param can be ignored)

#@markdown ---
#@markdown #### Épocas de entrenamiento: (No se recomienda cambiar.)
hparams.epochs =  250#@param {type: "integer"}

#@markdown ---
#@markdown #### Rango de épocas para guardar: (No se recomienda cambiar.)
saving_interval =  1#@param {type: "integer"}

torch.backends.cudnn.enabled = hparams.cudnn_enabled
torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

#@markdown ---
#@markdown #### Dónde guardar tu modelo en tu Drive:
output_directory = '/content/drive/MyDrive/papa/audio' #@param {type: "string"}
log_directory = '/content/tacotron2/logs' # Location to save Log files locally
log_directory2 = None # Location to copy log files (done at the end of each epoch to cut down on I/O)
checkpoint_path = output_directory+(r'/')+model_filename

#@markdown ---
#@markdown ####¿Utilizar el Warmstart? Si vas a seguir entrenando desde un punto de guardado, desactívalo.
warm_start = True #@param {type:"boolean"}

#@markdown ---
#@markdown #### Intervalo de épocas para generar muestras de audio del modelo:
sample_interbal =  1#@param {type: "integer"}
#@markdown ---
#@markdown #### Guardar muestras de audio del modelo: (Bueno para enviar los progresos de entrenamiento.)
save_audios =  True #@param {type:"boolean"}
audio_path = output_directory
#@markdown ---
#@markdown ####Elija un modelo base para una alineación más rápida: (Obligatorio.)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
model_base = "ESPA\xD1A-Male" #@param ["LATAM-Male", "ESPAÑA-Male", "None"]
ids = {"LATAM-Male": "1rNwMflvcURMQ5kh8hVJgn458sXj5oehx", "ESPAÑA-Male": "1-1xz_0Bw5Xye1c4GE4kd7WmkNDb6Cm7M"}
file_status = os.path.isfile(checkpoint_path)
if model_base != "None":
    if file_status == True:
      print("\n\033[33m\033[1m[Aviso] Ya existe un archivo con ese nombre en tu disco. Si quieres utilizar el modelo base tendrás que borrar/renombrar este archivo.\nPero si has desactivado el warm start, puedes sobrescribir el archivo existente con el mismo nombre sin preocuparte para seguir entrenando\nel modelo sobre este sin perder datos.") 
    else:
      print(f"Descarga de la plantilla {model_base} en {checkpoint_path}")
      !gdown {ids[model_base]} -O "$checkpoint_path"
#@markdown ---

#@markdown ## <font color="pink"> **6. Convierte los .WAV en espectrogramas de Mel y comprueba los archivos.** 🔉>📈
#@markdown ---
#@markdown ####  Esta celda convierte automáticamente los .wav en .npy, los cuales serán necesarios para el entrenamiento de la voz.
import re
import num2words
import sys
from contextlib import redirect_stdout
import fileinput

with open('/content/tacotron2/filelists/list.txt', 'r') as f:  
    new_text = '\n'.join([line.strip() for line in f.read().split('\n') if line.strip()])
    with open('/content/tacotron2/filelists/list.txt', 'w') as n:  
      n.write(new_text)

f_input = open('/content/tacotron2/filelists/list.txt','r')
text = f_input.readlines()
#print(text)
text2 = len(text)
dummytext = ""
#print(text2)
for number in range(0,text2):
 dummytext = number
 if(str(dummytext) + ".wav") in (text[number]):
   pass
 else:
  dummytext = str(text[number])
  dummytext2 = dummytext.split("|")
  try:
    dummytext2[1] = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0)), lang='es'), dummytext2[1],)
  except IndexError:
    replace = dummytext.replace('\n', '')
    raise Exception(f"esta línea '{replace}' tiene un problema, por favor, compruebe.")
  #print(dummytext2[1])
  text[number]  = (str(dummytext2[0] + "|" + dummytext2[1].lower()))
  #print(text[number])

  #print(text[number])
textfinal = " "
for number in range(text2):
 textfinal += text[number]
textfinal = textfinal.replace('[','')
textfinal = textfinal.replace("',",'')
textfinal = textfinal.replace("'",'')
textfinal = textfinal.replace(']','')
textfinal = textfinal.replace('\\n','\n')
textfinal = textfinal.replace(' wavs','wavs')
t = textfinal
"".join([s for s in t.strip().splitlines(True) if s.strip()])
print(textfinal, file=open("/content/tacotron2/filelists/list1.txt", "a"))


!rm /content/tacotron2/filelists/list.txt
!mv /content/tacotron2/filelists/list1.txt /content/tacotron2/filelists/list.txt
if generate_mels:
    create_mels()

print("Checking for missing files")
# ---- Replace .wav with .npy in filelists ----
!sed -i -- 's,.wav|,.npy|,g' {hparams.training_files}; sed -i -- 's,.wav|,.npy|,g' {hparams.validation_files}

check_dataset(hparams)

#@markdown # <font color="pink"> **7. Empezar a entrenar (Por fin).** 🏋️
#@markdown ---
#@markdown #### Aquí puedes escribir el texto en el que se generarán las muestras de audio del modelo. Un texto muy grande puede llevar mucho tiempo y ser difícil de generar, y un texto muy pequeño puede no demostrar el modelo muy bien, así que elige sabiamente. <br> Si dejas puesto `random`, se pondrá una frase aleatoria que está en una lista de frases.
sampletext = "Hola campeon !" #@param {type: "string"}
if sampletext == "random":
    !gdown -q -O /content/tacotron2/filelists/randomtexts.txt 1zmYoPHdurHRxQZ7g4fkAZ4wDbIXZBljU
    with open("/content/tacotron2/filelists/randomtexts.txt", "r") as file:
        lines = file.readlines()
    sampletext = random.choice(lines)
print('FP16 Run:', hparams.fp16_run)
print('Dynamic Loss Scaling:', hparams.dynamic_loss_scaling)
print('Distributed Run:', hparams.distributed_run)
print('cuDNN Enabled:', hparams.cudnn_enabled)
print('cuDNN Benchmark:', hparams.cudnn_benchmark)
train(output_directory, log_directory, checkpoint_path,
      warm_start, n_gpus, rank, group_name, hparams, log_directory2)

"""#<font color="pink">🧍***¿Has terminado de entrenar por hoy?***🏠🚶


---


##🔊 Prueba el modelo en el cuaderno de síntesis pulsando [aquí.](https://colab.research.google.com/drive/1lRGlbiK2wUCm07BKIhjV3dKej7jV0s1y)
##📈 También puedes entrenar un Fine Tuning pulsando [aquí.](https://colab.research.google.com/github/justinjohn0306/FakeYou-Tacotron2-Notebook/blob/main/FakeYou_HiFi_GAN_Fine_Tuning.ipynb?) Esto es para hacer tu modelo menos robótico. **(Solo para cuando termines de entrenar definitivamente el modelo Tacotron.)**

#<font color="pink">✍️ ***Notas finales*** ✍️


---


## Un buen entrenamiento se ve así:
![img.png](https://media.discordapp.net/attachments/835971020569051216/851469553355587614/download_2.png)
"""