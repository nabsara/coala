import os
import copy
import click
import librosa
import lmdb
import torch
import numpy as np
from tqdm import tqdm
import pickle
import pretty_midi
from music2latent import EncoderDecoder

from coala.datasets.parsers import get_parser
from coala.datasets.data_example import DataExample


torch.set_grad_enabled(False)


def normalize_signal(x: np.ndarray, max_gain_db: int = 30, gain_margin: float = 0.9):
    peak = np.max(abs(x))
    if peak == 0:
        return x
    log_peak = 20 * np.log10(peak)
    log_gain = min(max_gain_db, -log_peak)
    gain = 10**(log_gain / 20)
    return gain_margin * x * gain


def get_midi(midi_data: pretty_midi.PrettyMIDI, chunk_number, num_signal, sample_rate, only_monophonic=False):
    do_silence_check = False

    length = num_signal / sample_rate
    tstart = chunk_number * num_signal / sample_rate
    tend = (chunk_number + 1) * num_signal / sample_rate

    if len(midi_data.instruments) == 0:
        do_silence_check = True
        midi_data = None
        return do_silence_check, midi_data

    out_notes = []
    for note in midi_data.instruments[0].notes:
        if note.end > tstart and note.start < tend:
            note.start = max(0, note.start - tstart)
            note.end = min(note.end - tstart, length)
            out_notes.append(note)

    if len(out_notes) == 0:
        do_silence_check = True
        midi_data = None
        return do_silence_check, midi_data
    
    if only_monophonic:
        all_notes = sorted(out_notes, key=lambda n: n.start)
        monophonic = True
        for i in range(len(all_notes) - 1):
            if all_notes[i+1].start < all_notes[i].end:
                monophonic = False
                break
        if not monophonic:
            return True, None

    midi_data.instruments[0].notes = out_notes
    midi_data.adjust_times([0, length], [0, length])

    return do_silence_check, midi_data

def get_midi_olap(midi_data: pretty_midi.PrettyMIDI,
                  chunk_start_sample,
                  num_signal,
                  sample_rate):

    # time window
    tstart = chunk_start_sample / sample_rate
    tend = (chunk_start_sample + num_signal) / sample_rate

    if len(midi_data.instruments) == 0:
        return True, None

    inst = midi_data.instruments[0]
    notes = inst.notes

    out_notes = []

    for note in notes:
        # since sorted, we can early stop
        if note.start >= tend:
            break

        # skip notes before chunk
        if note.end <= tstart:
            continue

        # clip note to chunk
        start = max(note.start, tstart) - tstart
        end = min(note.end, tend) - tstart

        # optional: skip very short clipped notes
        if end - start < 1e-3:
            continue

        out_notes.append(pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=start,
            end=end
        ))

    if not out_notes:
        return True, None

    # build new MIDI
    new_midi = pretty_midi.PrettyMIDI()
    new_inst = pretty_midi.Instrument(
        program=inst.program,
        name=inst.name
    )
    new_inst.notes = out_notes
    new_midi.instruments.append(new_inst)

    return False, new_midi


@click.command()
@click.option('-i', '--input_path', default="", help='folder with the audio files')
@click.option('-o', '--output_path', default="", help='lmdb save path')
@click.option('-s', '--db_size', default=40, help='Max Size of lmdb database')
@click.option('-p', '--parser_name', default="simple_parser", help='parser function to obtain the list audio files and metadatas')
@click.option('-m', '--emb_model_path', default="", help='code to use. Either "music2latent" or a torchscript path')
@click.option('--gpu', default=-1, help='device for basic-pitch and codec (-1 for cpu)')
@click.option('-n', '--num_signal', default=131_072, help='chunk sizes')
@click.option('--sr', default=44_100, help='sample rate')
@click.option('-b', '--batch_size', default=32, help='Batch size (for embedding model inference)')
@click.option('--normalize', is_flag=True, help='Normalize audio waveform (done once per file and not chunks ! )')
@click.option('--cut_silences', is_flag=True, help='Remove silence chunks')
@click.option('--save_waveform', is_flag=True, help='Wether to save the waveform in the lmdb')
@click.option('--mono', is_flag=True, help='filter midi to keep only monophonic audio chunks')
@click.option('--olap', is_flag=True, help='truncate into overlaped audio chunks')
def main(
        input_path, 
        output_path, 
        db_size, 
        parser_name,
        emb_model_path, 
        gpu, 
        num_signal, 
        sr, 
        batch_size, 
        normalize,     
        cut_silences, 
        save_waveform,
        mono,
        olap,
    ):

    # load pretrained codec
    device = "cuda:" + str(gpu) if torch.cuda.is_available() and gpu >= 0 else "cpu"

    # logging
    print("-"*60)
    print(" "*20 + "Config:")
    print("-"*60)
    print("Audios path : ", input_path)
    print("Output path : ", output_path)
    print("parser : ", parser_name)
    print("signal length : ", num_signal)
    print("sample rate : ", sr)
    print("pretrained codec : ", emb_model_path)
    print("device : ", device)
    print("-"*60)
    print(" "*20 + "Attributes:")
    print("-"*60)


    if emb_model_path == "music2latent":
        emb_model = EncoderDecoder(device=device)
    else:
        emb_model = torch.jit.load(emb_model_path).to(device)
    z_length = None

    # initialize lmdb database
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(
        output_path,
        map_size=int(db_size) * 1024**3,
        map_async=True,
        writemap=True,
        readahead=False,
    )

    # parse audio files
    audio_files, metadatas, instruments_values = get_parser(parser_name)(input_path)
    midi_files = False
    if "midi_filepath" in metadatas[0].keys():
        midi_files = True
    midi_loaded = False
    if "midi_data" in metadatas[0].keys():
        midi_loaded = True

 
    # init
    chunks_buffer, metadatas_buffer, midis = [], [], []
    cur_index = 0
    skip_examples_inf_loud = 0

    hop_size = 1
    if olap:
        hop_size = num_signal // 2

    # process loop
    for i, (file, metadata) in enumerate(zip(tqdm(audio_files), metadatas)):

        # load audio
        try:
            audio = librosa.load(file, sr=sr)[0]  # only mono not stereo
        except:
            print("error loading file : ", file)
            continue
        audio = audio.squeeze()

        if audio.shape[-1] == 0:
            print("Empty file")
            continue

        if normalize:
            audio = normalize_signal(audio)

        # check audio length to ensure power of 2 (ie. multiple of num signal)
        length = audio.shape[-1]
        if length < num_signal:
            print(f'Warning - skip audio {file} because audio length is too short : {length} < num_signal={num_signal}')
            continue
        
        if not olap:
            if length % num_signal < num_signal // 2:
                length_crop = (length // num_signal) * num_signal
                audio = audio[:length_crop]
            else:
                # pad audio signal to a power of 2 (num_signal)
                audio = np.pad(audio, (0, num_signal - audio.shape[-1] % num_signal))
            # reshape audio signal into chunks
            chunks = audio.reshape(-1, num_signal)
        else:
            pad_amount = (hop_size - (length - num_signal) % hop_size) % hop_size
            if pad_amount > 0:
                audio = np.pad(audio, (0, pad_amount))

            length = audio.shape[-1]

            # sliding window
            chunk_index = 0
            chunks = []
            chunk_start_sample = []
            for start in range(0, length - num_signal + 1, hop_size):
                chunk = audio[start:start + num_signal]

                chunk_start_sample.append(start)

                # OPTIONAL: skip silent / too loud
                if np.max(np.abs(chunk)) > 1.0:
                    skip_examples_inf_loud += 1
                    continue

                # --- store chunk ---
                chunks.append(chunk)
            chunks = np.stack(chunks)
        
        # process MIDI data
        if midi_files:
            midi_data = pretty_midi.PrettyMIDI(metadata["midi_filepath"])
        elif midi_loaded:
            midi_data = metadata["midi_data"]
        else:
            midi_data = None
        
        
        chunk_index = 0

        # get the number of latent frames computed by the codec
        if i == 0 and emb_model is not None and z_length is None:
            ex_chunk_torch = torch.from_numpy(chunks[0]).to(device)
            if emb_model_path == "music2latent":
                ex_chunk_torch = ex_chunk_torch.reshape(-1, num_signal)
            else:
                ex_chunk_torch = ex_chunk_torch.reshape(-1, 1, num_signal)

            z_ex = emb_model.encode(ex_chunk_torch)
            z_length = z_ex.shape[-1]

        empty_midis_indices = []
        for j, chunk in enumerate(chunks):
            
            # Chunk the midi
            if midi_data is not None:
                if not olap:
                    silence_test, midi = get_midi(
                        copy.deepcopy(midi_data), 
                        chunk_number=chunk_index,
                        num_signal=num_signal,
                        sample_rate=sr,
                        only_monophonic=mono,
                    )
                else:
                    silence_test, midi = get_midi_olap(
                        copy.deepcopy(midi_data), 
                        chunk_start_sample=chunk_start_sample[j],
                        num_signal=num_signal,
                        sample_rate=sr,
                    )
                if not silence_test: # safety check in case of wrong pitch estimation
                    silence_test = np.max(abs(chunk)) < 0.05 if cut_silences else False
                if midi is None:
                    empty_midis_indices.append(j)
            else:
                midi = None
                silence_test = np.max(abs(chunk)) < 0.05 if cut_silences else False

            # don't process buffer if empty slice
            if silence_test:
                chunk_index += 1
                continue

            midis.append(midi)
            chunks_buffer.append(chunk)
            metadatas_buffer.append(metadata)

            if len(chunks_buffer) == batch_size or (j == len(chunks) - 1 and i == len(audio_files) - 1):

                # get latent representation from pretrained codec
                if emb_model is not None:
                    chunks_buffer_torch = torch.from_numpy(np.stack(chunks_buffer)).to(device)
                    if emb_model_path == "music2latent":
                        chunks_buffer_torch = chunks_buffer_torch.reshape(-1, num_signal)
                    else:
                        chunks_buffer_torch = chunks_buffer_torch.reshape(-1, 1, num_signal)
                    try:
                        z = emb_model.encode(chunks_buffer_torch)
                    except:
                        z_list = []
                        for b in range(chunks_buffer_torch.shape[0]):
                            z_b = emb_model.encode(chunks_buffer_torch[b].unsqueeze(0))
                            z_list.append(z_b)
                        z = torch.cat(z_list, dim=0)
                else:
                    z = [None] * len(chunks_buffer)
                

                for i, (audio_array, z_array, midi, cur_metadata) in enumerate(zip(chunks_buffer, z, midis, metadatas_buffer)):
                    
                    assert audio_array.shape[-1] == num_signal

                    if i in empty_midis_indices:
                        # do not store chunk if you require midi data but midi is None
                        continue
                    
                    # create instance of our lmdb database
                    key = f"{cur_index:08d}"
                    data_ex = DataExample()

                    # save chunk audio waveform to lmdb database
                    if save_waveform:
                        if type(audio_array) == torch.Tensor:
                            audio_array = audio_array.cpu().numpy()
                        audio_array = (audio_array * (2**15 - 1)).astype(np.int16)
                        data_ex.put_array("waveform", audio_array, dtype=np.int16)

                    # save latent representation
                    if z_array is not None:
                        data_ex.put_array("z", z_array.cpu().numpy(), dtype=np.float32)

                    # save metadata
                    cur_metadata["chunk_index"] = chunk_index
                    cur_metadata["key"] = key
                    if "midi_data" in cur_metadata.keys():
                        cur_metadata.pop("midi_data")
                    data_ex.put_metadata(cur_metadata)

                    # save MIDI data
                    if midi is not None:
                        data_ex.put_buffer(key="midi", b=pickle.dumps(midi), shape=None)
                        
                    # save AudioExample instance to lmdb database
                    with env.begin(write=True) as txn:
                        txn.put(key.encode(), bytes(data_ex))
                    cur_index += 1

                chunks_buffer, midis, metadatas_buffer = [], [], []
            chunk_index += 1

    print("nb of audio chunks skipped because found -inf loudness: ", skip_examples_inf_loud)
    env.close()


if __name__ == '__main__':
    main()
