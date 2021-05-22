# This is a implementation of AI Video Dubs

### Tech Stacks:
1. transcrible audio to text with Google Speech-to-Text API;  
2. translate the text with Google Translate API;  
3. voice out the translations with Google Text-to-Speech API;  
4. Google Cloud Project, Google Cloud CLI

### Obstacles:
1. The video can be incorrectly transcribed from audio to text by the Speech-to-Text API
2. That text can be incorrectly or awkwardly translated by the Translation API
3. Those translations can be mispronounced by the Text-to-Speech API

### Steps:
1. Extract audio from video files
2. Convert audio to text using the Speech-to-Text API
3. Split transcribed text into sentences/segments for translation
4. Translate text
5. Generate spoken audio versions of the translated text
6. Speed up the generated audio to align with the original speaker in the video
7. Stitch the new audio on top of the fold audio/video