import torch
from TTS.api import TTS

emotion_list = ["Neutral", "Happy", "Sad", "Angry", "Dull"]

MODELS_WITH_SEP_TESTS = ["tts_models/multilingual/multi-dataset/bark",
                         "tts_models/en/multi-dataset/tortoise-v2",
                         "tts_models/multilingual/multi-dataset/xtts_v1.1",
                         "tts_models/multilingual/multi-dataset/xtts_v2", ]
# "Downloading model to C:\Users\Administrator\AppData\Local\tts\tts_models--multilingual--multi-dataset--xtts_v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
tts.tts_to_file(
    text="《我爱这土地》写于1938年11月17日，"
         "发表于同年12月桂林出版的《十日文萃》。"
         "1938年10月，武汉失守，日本侵略者的铁蹄猖狂地践踏中国大地。作者和当时文艺界许多人士一同撤出武汉，"
         "汇集于桂林。作者满怀对祖国的挚爱和对侵略者的仇恨便写下了这首诗。",
    emotion="Neutral",
    speed=1,
    language="zh-cn",
    speaker_wav="my/cloning/111.wav",
    file_path="my/output/111.wav")
