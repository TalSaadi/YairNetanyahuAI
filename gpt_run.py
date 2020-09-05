import gpt_2_simple as gpt2
import os
import requests
from datetime import datetime

model_name = "355M"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)  # model is saved into current directory under /models/355M/
	
file_name = "tweets.csv"
if not os.path.isfile(file_name):
	print("Tweets file not found")
    exit()
    
with open(file_name, 'w') as f:
	f.write(data.text)

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              dataset=file_name,
              model_name=model_name,
              steps=2000,
              restore_from='fresh',
              run_name='run1',
              print_every=10,
              sample_every=500,
              save_every=500)

gen_file = 'gpt2_gentext_{:%Y%m%d_%H%M%S}.txt'.format(datetime.utcnow())

gpt2.generate_to_file(sess,
                      destination_path=gen_file,
                      length=200,
                      temperature=1.0,
                      top_p=0.9,
                      prefix='<|startoftext|>',
                      truncate='<|endoftext|>',
                      include_prefix=False,
                      nsamples=1000,
                      batch_size=20)
