import torch
from transformers import AutoModel, AutoProcessor
from SoundCodec.base_codec.general import save_audio, ExtractedUnit
from ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download

class BaseCodec:
	def __init__(self):
		self.fa_encoder = FACodecEncoder(
			ngf=32,
			up_ratios=[2, 4, 5, 5],
			out_channels=256,
		)

		self.fa_decoder = FACodecDecoder(
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
		)

		encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
		decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

		self.fa_encoder.load_state_dict(torch.load(encoder_ckpt))
		self.fa_decoder.load_state_dict(torch.load(decoder_ckpt))

		self.fa_encoder.eval()
		self.fa_decoder.eval()

	@torch.no_grad()
	def synth(self, data, local_save=True):
		extracted_unit = self.extract_unit(data)
		data['unit'] = extracted_unit.unit
		audio_values = self.decode_unit(extracted_unit.stuff_for_synth)
		if local_save:
			audio_path = f"dummy_{self.pretrained_model_name}/{data['id']}.wav"
			save_audio(audio_values, audio_path, self.sampling_rate)
			data['audio'] = audio_path
		else:
			data['audio']['array'] = audio_values
		return data

	@torch.no_grad()
	def extract_unit(self, data):
		audio_sample = data["audio"]["array"]
		inputs = self.processor(raw_audio=audio_sample, sampling_rate=self.sampling_rate, return_tensors="pt")
		input_values = inputs["input_values"].to(self.device)
		padding_mask = inputs["padding_mask"].to(self.device) if inputs["padding_mask"] is not None else None
		encoder_outputs = self.model.encode(input_values, padding_mask)
		return ExtractedUnit(
			unit=encoder_outputs.audio_codes.squeeze(),
			stuff_for_synth=(encoder_outputs, padding_mask)
		)

	@torch.no_grad()
	def decode_unit(self, stuff_for_synth):
		encoder_outputs, padding_mask = stuff_for_synth
		audio_values = \
			self.model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, padding_mask)[0]
		return audio_values[0].cpu().numpy()
