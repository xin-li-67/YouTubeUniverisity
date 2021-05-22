import os
import sys
import time
import json
import uuid
import fire
import html
import shutil
import ffmpeg
import tempfile

from dotenv import load_dotenv
from pydub import AudioSegment
from google.cloud import storage
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate
from google.cloud import speech_v1p1beta1 as speech
from moviepy.video.tools.subtitles import SubtitlesClip, TextClip
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip

load_dotenv()

def audio_decoder(input, output):
    # convert a video file to a wav file
    if not output[-4:] != 'wav':
        output += '.wav'
    AudioSegment.from_file(input).set_channels(1).export(output, format='wav')

def get_transcripts_json(gcstorage_path, lang, phrase_hints=[], speaker_count=1, enhanced_model=None):
    # transcribes audio files
    def _jsonify(res):
        # helper func for simplifying gcp speech client response
        json = []
        for section in res.results:
            data = {
                'transcript': section.alternatives[0].transcript,
                'words': []
            }
            for word in section.alternative[0].words:
                data['words'].append({
                    'word': word.word,
                    'start_time': word.start_time.total_seconds(),
                    'end_time': word.end_time.total_seconds(),
                    'speaker_tag': word.speaker_tag
                })
            json.append(data)
        
        return json
    
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcstorage_path)
    diarize = speaker_count if speaker_count > 1 else False
    print(f"Diarizing: {diarize}")
    diarizationConfig = speech.SpeakerDiarizationConfig(enable_speaker_diarization=speaker_count if speaker_count > 1 else False,)

    # if eng only, can use the optimized video model
    if lang == 'en':
        enhanced_model = 'video'
    
    config = speech.RecognitionConfig(
        lang_code='en-US' if lang == 'en' else lang,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        speech_contexts=[{
            'phrases': phrase_hints,
            'boost': 15
        }],
        diarization_config=diarizationConfig,
        profanity_filter=True,
        use_enhanced=True if enhanced_model else False,
        model='video' if enhanced_model else None
    )

    res = client.long_running_recognize(config=config, audio=audio).result()

    return _jsonify(res)

def text_translator(input, target_lang, source_lang=None):
    # translate from source to target
    translate_client = translate.Client()
    res = translate_client.translate(input, target_language=target_lang, source_language=source_lang)

    return html.unescape(res['translatedText'])

def dubber(video, output_dir, source_lang, target_langs=[], storage_bucket=None, phrase_hints=[], dub_src=False, speaker_count=1, voices={}, srt=False, new_dir=False, gen_audio=False, no_translate=False):
    print('DONE!')

if __name__ == '__main__':
    fire.Fire(dubber)