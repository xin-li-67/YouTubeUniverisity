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
from google.api_core.future import base
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

def sentence_parser_with_speaker(json, lang):
    # take json from get_transcripts_json and breaks into sentences
    def get_word(word, lang):
        # special case for parsing japanese words
        if lang == 'ja':
            return word.split('|')[0]
        return word
    
    sentences = []
    sentence = {}

    for res in json:
        for i, word in enumerate(res['words']):
            word_text = get_word(word['word'], lang)
            if not sentence:
                sentence = {
                    lang: [word_text],
                    'speaker': word['speaker_tag'],
                    'start_time': word['start_time'],
                    'end_time': word['end_time']
                }
            elif word['speaker_tag'] != sentence['speaker']:
                # a new speaker, save the sentence and create a new one
                sentence[lang] = ' '.join(sentence[lang])
                sentences.append(sentence)
                sentence = {
                    lang: [word_text],
                    'speaker': word['speaker_tag'],
                    'start_time': word['start_time'],
                    'end_time': word['end_time']
                }
            else:
                sentence[lang].append(word_text)
                sentence['end_time'] = word['end_time']
            
            if i + 1 < len(res['word']) and word['end_time'] < res['words'][i + 1]['start_time']:
                # if there's greater than one second gap, assume this is a new sentence
                sentence[lang] = ' '.join(sentence[lang])
                sentences.append(sentence)
                sentence = {}
        
        if sentence:
            sentence[lang] = ' '.join(sentence[lang])
            sentences.append(sentence)
            sentence = {}
    
    return sentences

def text_translator(input, target_lang, source_lang=None):
    # translate from source to target
    translate_client = translate.Client()
    res = translate_client.translate(input, target_language=target_lang, source_language=source_lang)

    return html.unescape(res['translatedText'])

def text_to_speech_converter(text, lang, voice_name=None, speaking_rate=1):
    # convert text to audio
    client = texttospeech.TextToSpeechAsyncClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # build the voice request, set lang to 'en-US' and ssml voice genter to 'neutral'
    if not voice_name:
        voice = texttospeech.VoiceSelectionParams(language_code=lang, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
    else:
        voice = texttospeech.VoiceSelectionParams(language_code=lang, name=voice_name)
    
    # select the type of audio file to return as the output
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=speaking_rate)

    # preform the text-to-speech request on the text input with the selected voice params and output type
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    return response.audio_content

def speaking_duration(text, lang, duration, voice_name=None):
    # speak within a certain time limit
    base_audio = sentence_parser_with_speaker(text, lang, voice_name=voice_name)
    assert len(base_audio)

    f = tempfile.NamedTemporaryFile(mode='wb')
    f.write(base_audio)
    f.flush()
    base_duration = AudioSegment.from_mp3(f.name).duration_seconds
    f.close()
    ratio = base_duration / duration

    # if audio fits, return it
    if ratio <= 1:
        return base_audio
    # if it's too lang, round to one decimal point and go a litter faster
    ratio = round(ratio, 1)
    if ratio > 4:
        ratio = 4
    
    return sentence_parser_with_speaker(text, lang, voice_name=voice_name, speaking_rate=ratio)

def convert_to_srt(transcripts, chars_perline=60):
    # convert transcripts to SRT, only intend to work with eng
    """
    SRT files have this format:
    [Section of subtitles number]
    [Time the subtitle is displayed begins] –> [Time the subtitle is displayed ends]
    [Subtitle]
    Timestamps are in the format:
    [hours]: [minutes]: [seconds], [milliseconds]
    Note: about 60 characters comfortably fit on one line
    for resolution 1920x1080 with font size 40 pt.
    """
    def _srt_time(seconds):
        millisecs = seconds * 1000
        seconds, millisecs = divmod(millisecs, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return "%d:%d:%d,%d" % (hours, minutes, seconds, millisecs)
    
    def _to_srt(words, start_time, end_time, index):
        return f"{index}\n" + _srt_time(start_time) + " --> " + _srt_time(end_time) + f"\n{words}"

    start_time = None
    sentence = ''
    srt = []
    index = 1

    for word in [word for x in transcripts for word in x['word']]:
        if not start_time:
            start_time = word['start_time']
        
        sentence += ' ' + word['word']
        if len(sentence) > chars_perline:
            srt.append(_to_srt(sentence, start_time, word['end_time'], index))
            index += 1
            sentence = ''
            start_time = None
    
    if len(sentence):
        srt.append(_to_srt(sentence, start_time, word['end_time'], index))
    
    return '\n\n'.join(srt)

def stitch_audio(sentences, audio_dir, movie_file, output, srt_path=None, overlay_gain=-30):
    # combines sentences, audio clips, and video file into the dubbed output
    audio_files = os.listdir(audio_dir)
    audio_files.sort(key=lambda x: int(x.split('.')[0]))

    segments = [AudioSegment.from_mp3(os.path.join(audio_dir, x)) for x in audio_files]
    dubbed = AudioSegment.from_file(movie_file)

    for sentence, segment in zip(sentences, segments):
        # place each generated audio at the correct timestamp
        dubbed = dubbed.overlay(segment, position=sentence['start_time']*1000, gain_during_overlay=overlay_gain)
    
    # wirte the final audio to a temp file
    audio_file = tempfile.NamedTemporaryFile()
    dubbed.export(audio_file)
    audio_file.flush()

    # add new audio to the video and save it
    clip = VideoFileClip(movie_file)
    audio = AudioFileClip(audio_file.name)
    clip = clip.set_audio(audio)

    # add transcripts if there is any
    if srt_path:
        width, height = clip.size[0]*0.75, clip.size[1]*0.20

        def generator(txt):
            return TextClip(txt, fond='Georgia-Regular', size=[width,height], color='black', method='caption')
        
        subtitles = SubtitlesClip(srt_path, generator).set_position(('center', 'bottom'))
        clip = CompositeVideoClip([clip, subtitles])
    
    clip.write_videofile(output, codec='libx264', audio_codec='aac')
    audio_file.close()

def dubber(video, output_dir, source_lang, target_langs=[], storage_bucket=None, phrase_hints=[], dub_src=False, speaker_count=1, voices={}, srt=False, new_dir=False, gen_audio=False, no_translate=False):
    # main func
    base_name = os.path.split(video)[-1].split('.')[0]
    if new_dir:
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    output_files = os.listdir(output_dir)
    if not f"{base_name}.wav" in output_files:
        print("Extracting audio from video")
        fn = os.path.join(output_dir, base_name+'.wav')
        audio_decoder(video, fn)
        print(f"Wrote {fn}")
    
    
    
    print('DONE!')

if __name__ == '__main__':
    fire.Fire(dubber)