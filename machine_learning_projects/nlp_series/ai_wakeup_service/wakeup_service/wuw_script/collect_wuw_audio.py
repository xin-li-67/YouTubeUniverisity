import os
import time
import wave
import pyaudio
import argparse

class Listener:
    def __init__(self, args):
        self.chunk = 1024
        self.channels = 1
        self.sample_rate = args.sample_rate
        self.record_secs = args.seconds
        self.pFORMAT = pyaudio.paInt16

        self.pAudio = pyaudio.PyAudio()
        
        self.stream = self.pAudio.open(format=self.pFORMAT, channels=self.channels,
                                       rate=self.sample_rate, input=True,
                                       output=True, frames_per_buffer=self.chunk)
        
    def save_audio(self, file_name, frames):
        self.stream.stop_stream()
        self.stream.close()

        self.pAudio.terminate()

        # save audio file
        wf = wave.open(file_name, "wb")
        # set the channels
        wf.setnchannels(self.channels)
        # set the sample format
        wf.setsampwidth(self.pAudio.get_sample_size(self.pFORMAT))
        # set the sample rate
        wf.setframerate(self.sample_rate)
        # write the frames as bytes
        wf.writeframes(b"".join(frames))
        wf.close()

def interactive(args):
    index = 0

    try:
        while True:
            listener = Listener(args)
            frames = []
            
            print('Begin Recording...')
            input('Press ENTER to continue. The recoding will be {} seconds. Press CTRL+C to exit'.format(args.seconds))
            time.sleep(1)
            
            for i in range(int((listener.sample_rate/listener.chunk)*listener.record_secs)):
                data = listener.stream.read(listener.chunk, exception_on_overflow=False)
                frames.append(data)
            
            save_path = os.path.join(args.interactive_save_path, "{}.wav".format(index))
            listener.save_audio(save_path, frames)
            index += 1
    except KeyboardInterrupt:
        print('Keyboard Interrupt -- Terminating')
    except Exception as e:
        print(str(e))

def main(args):
    listener = Listener(args)
    frames = []
    print('Start Recording...')

    try:
        while True:
            if listener.record_secs == None:  # record until keyboard interupt
                print('Recording indefinitely... Press CTRL+C to cancel', end="\r")
                data = listener.stream.read(listener.chunk)
                frames.append(data)
            else:
                for i in range(int((listener.sample_rate/listener.chunk) * listener.record_secs)):
                    data = listener.stream.read(listener.chunk)
                    frames.append(data)
                raise Exception('Recording Finished')
    except KeyboardInterrupt:
        print('Keyboard Interrupt -- Terminating')
    except Exception as e:
        print(str(e))

    print('Recording Finished...')
    listener.save_audio(args.save_path, frames)
    print('Audio save locally')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
        This script is used to collect data for wake up training...
        To record environment sound run set seconds to None. This will record indefinitely until CTRL+C.
        To record for a set amount of time set seconds to whatever you want.
        To record interactively (usually for recording your own wake words N times) use --interactive mode.
        ''')

    parser.add_argument('--sample_rate', type=int, default=8000, help='the sample_rate to record at')
    parser.add_argument('--seconds', type=int, default=None, help='set to None to record forever until keyboard interrupt')
    parser.add_argument('--save_path', type=str, default=None, required=False, help='use full path to save file')
    parser.add_argument('--interactive_save_path', type=str, default=None, required=False, help='directory to save all the interactive 2 second samples')
    parser.add_argument('--interactive', default=False, action='store_true', required=False, help='sets to interactive mode')
    args = parser.parse_args()

    if args.interactive:
        if args.interactive_save_path is None:
            raise Exception('need to set --interactive_save_path')
        interactive(args)
    else:
        if args.save_path is None:
            raise Exception('need to set --save_path')
        main(args)
