import numpy as np

import torch
from transformers import AutoModel, AutoProcessor
from SoundCodec.base_codec.general import save_audio, ExtractedUnit

from huggingface_hub import hf_hub_download

class BaseCodec:
	def __init__(self):
		try:
			from ns3_codec import FACodecEncoder, FACodecDecoder
		except:
			raise Exception("Please install Amphion first. see https://github.com/open-mmlab/Amphion")
		
		self.model_name = "facodec_16khz"
		self.model_type = "16khz"
		self.sampling_rate = 16_000	

		self.encoder = FACodecEncoder(
			ngf=32,
			up_ratios=[2, 4, 5, 5],
			out_channels=256,
		).to("cuda")

		self.decoder = FACodecDecoder(
			in_channels=256,
			upsample_initial_channel=1024,
			ngf=32,
			up_ratios=[5, 5, 4, 2],
			vq_num_q_c=2,
			vq_num_q_p=1,
			vq_num_q_r=3,
			vq_dim=256,
			codebook_dim=8,
			codebook_size_prosody=10,
			codebook_size_content=10,
			codebook_size_residual=10,
			use_gr_x_timbre=True,
			use_gr_residual_f0=True,
			use_gr_residual_phone=True,
		).to("cuda")

		encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
		decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

		self.encoder.load_state_dict(torch.load(encoder_ckpt))
		self.decoder.load_state_dict(torch.load(decoder_ckpt))

		self.encoder.eval()
		self.decoder.eval()

	def config(self):
		self.model_type = "16khz"
		self.sampling_rate = 16_000
	@torch.no_grad()
	def synth(self, data, local_save=True):
		extracted_unit = self.extract_unit(data)
		data['unit'] = extracted_unit.unit
		audio_values = self.decode_unit(extracted_unit.stuff_for_synth)
		if local_save:
			audio_path = f"dummy_{self.model_name}/{data['id']}.wav"
			save_audio(audio_values, audio_path, self.sampling_rate)
			data['audio'] = audio_path
		else:
			data['audio']['array'] = audio_values
		return data

	@torch.no_grad()
	def extract_unit(self, data):
		audio_sample = data["audio"]["array"]
		if isinstance(audio_sample, np.ndarray):
			audio_sample = torch.from_numpy(audio_sample).float()
		audio_sample = audio_sample.to("cuda")
		audio_sample = audio_sample.unsqueeze(0).unsqueeze(0)
		print(audio_sample.shape)
		print(type(audio_sample))
		enc_out = self.encoder(audio_sample)
		vq_post_emb, vq_id, _, quantized, spk_embs = self.decoder(enc_out, eval_vq=False, vq=True)

		# vq id shape: torch.Size([6, 1, 628])
		# prosody code shape: torch.Size([1, 1, 628])
		# content code shape: torch.Size([2, 1, 628])
		# residual code shape: torch.Size([3, 1, 628])
		# speaker embedding shape: torch.Size([1, 256])
		# get prosody code
		prosody_code = vq_id[:1]
		print("prosody code shape:", prosody_code.shape)
		
		# get content code
		cotent_code = vq_id[1:3]
		print("content code shape:", cotent_code.shape)
		
		# get residual code (acoustic detail codes)
		residual_code = vq_id[3:]
		print("residual code shape:", residual_code.shape)

		# speaker embedding
		print("speaker embedding shape:", spk_embs.shape)
		# inputs = self.processor(raw_audio=audio_sample, sampling_rate=self.sampling_rate, return_tensors="pt")
		# input_values = inputs["input_values"].to(self.device)
		# padding_mask = inputs["padding_mask"].to(self.device) if inputs["padding_mask"] is not None else None
		# encoder_outputs = self.model.encode(input_values, padding_mask)
		return ExtractedUnit(
			unit=vq_id,
			stuff_for_synth=(vq_post_emb, spk_embs)
		)

	@torch.no_grad()
	def decode_unit(self, stuff_for_synth):
		vq_post_emb, spk_embs = stuff_for_synth
		audio_values = self.decoder.inference(vq_post_emb, spk_embs)
		print("facodec output shape:", audio_values.shape)
		return audio_values[0].cpu().numpy()
