from datasets import load_dataset, Audio
from transformers import SeamlessM4Tv2Model, SeamlessM4TProcessor
import torch
import re
from evaluate import load

dataset = load_dataset("google/fleurs", "tr_tr", split="test+validation")

dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

"""dataset = dataset.remove_columns(["accent", "age", "client_id", 
                                  "down_votes", "gender", "locale", 
                                  "segment", "up_votes", "path", "variant"])"""

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\']'
def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_remove_regex, '', batch["transcription"]).lower()
    return batch

dataset = dataset.map(remove_special_characters)

def replace_hatted_characters(batch):
    batch["transcription"] = re.sub('[â]', 'a', batch["transcription"])
    batch["transcription"] = re.sub('[î]', 'i', batch["transcription"])
    batch["transcription"] = re.sub('[ô]', 'o', batch["transcription"])
    batch["transcription"] = re.sub('[û]', 'u', batch["transcription"])
    return batch

dataset = dataset.map(replace_hatted_characters)

processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-v2-large", language="tur", task="transcribe")
model = SeamlessM4Tv2Model.from_pretrained("tgrhn/seamless_turkish").to("cuda")

def map_to_pred(batch):
    audio_tensor = torch.tensor(batch["audio"]["array"] ).float().unsqueeze(0).to("cuda")
        
    audio_inputs = processor(audios=audio_tensor.cpu(), return_tensors="pt", sampling_rate=16000).to("cuda")
        
    output_tokens = model.generate(**audio_inputs, tgt_lang="tur", generate_speech=False)
        
    batch["prediction"]  = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    
    batch["transcription"] = batch["transcription"].lower()
    batch["prediction"] = re.sub(chars_to_remove_regex, '', batch["prediction"]).lower()
    
    return batch

result = dataset.map(map_to_pred)

wer = load("wer")
print("WER: {:2f}".format(100 * wer.compute(predictions=result["prediction"], references=result["transcription"])))
