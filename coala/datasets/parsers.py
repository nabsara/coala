import os
import pathlib
import csv
import random
from tqdm import tqdm
import numpy as np

from typing import Iterable, Sequence


def flatten(iterator: Iterable):
    for elm in iterator:
        for sub_elm in elm:
            yield sub_elm


def search_for_audios(
        path_list: Sequence[str],
        extensions: Sequence[str] = [
            "wav", "opus", "mp3", "aac", "flac", "aif", "ogg"
        ],
    ):
    paths = map(pathlib.Path, path_list)
    audios = []
    for p in paths:
        for ext in extensions:
            audios.append(p.rglob(f"*.{ext}"))
    audios = flatten(audios)
    audios = [str(a) for a in audios if 'MACOS' not in str(a)]

    return audios


def simple_parser(audio_folder, filters=None):
    audios = search_for_audios([audio_folder])
    audios = map(str, audios)
    audios = map(os.path.abspath, audios)
    audios = [*audios]

    if filters is not None:
        audios = [
            a for a in audios if any([s.lower() in a.lower() for s in filters])
        ]

    random.shuffle(audios)

    metadatas = [{"path": audio} for audio in audios]
    print(len(audios), " files found")

    return audios, metadatas, None


def medley_solos_mono_parser(audio_folder):
    audios = search_for_audios([audio_folder])

    metadata_filepath = os.path.join(
        str(pathlib.Path(audios[0]).parent.parent),
        "Medley-solos-DB_metadata.csv")

    # all_inst_list = [
    #     'clarinet',
    #     'distorted electric guitar',
    #     'female singer',
    #     'flute',
    #     'piano',
    #     'tenor saxophone',
    #     'trumpet',
    #     'violin'
    # ]

    filter_ployphonic_instruments = ['distorted electric guitar', 'piano']

    raw_metadata = {}
    with open(metadata_filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_metadata[row['uuid4']] = {'instrument': row['instrument']}

    metadata = []
    audios_filtered = []

    for audio in tqdm(audios):

        uuid = pathlib.Path(audio).stem.split("_")[-1]

        inst_name = raw_metadata[uuid]['instrument']

        if inst_name not in filter_ployphonic_instruments:
            audios_filtered.append(audio)

            data = {
                "path": audio,
                "instrument": inst_name,
            }
            metadata.append(data)

    print(len(audios), " files found")
    print(f"selected files : {len(audios_filtered)} / {len(audios)}")
    print("sanity check: ", audios_filtered[0], metadata[0])

    instruments_values = ['female singer', 'violin', 'flute', 'clarinet', 'tenor saxophone', 'trumpet']

    return audios_filtered, metadata, instruments_values


def musedb_drums_parser(audio_folder):
    audios = search_for_audios([audio_folder])

    metadata = []
    audios_filtered = []

    for audio in tqdm(audios):

        inst_name = pathlib.Path(audio).stem

        if inst_name == 'drums':
            audios_filtered.append(audio)

            data = {"path": audio}
            metadata.append(data)

    print(len(audios), " files found")
    print(f"selected files : {len(audios_filtered)} / {len(audios)}")
    print("sanity check: ", audios_filtered[0], metadata[0])

    return audios_filtered, metadata, None


def get_parser(parser_name):
    if parser_name == "simple_parser":
        parser = simple_parser
    elif parser_name == "medley_solos_mono_parser":
        parser = medley_solos_mono_parser
    elif parser_name == "musedb_drums_parser":
        parser = musedb_drums_parser
    else:
        raise NotImplementedError(f'No parser method named : {parser_name}.')
    return parser
