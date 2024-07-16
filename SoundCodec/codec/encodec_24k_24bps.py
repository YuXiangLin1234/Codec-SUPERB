from SoundCodec.base_codec.facodec import BaseCodec

class Codec(BaseCodec):
    def config(self):
        self.model.set_target_bandwidth(24.0)
        self.setting = "encodec_24khz_24"
        self.sampling_rate = 24_000
