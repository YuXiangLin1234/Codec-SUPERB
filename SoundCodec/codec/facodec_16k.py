from SoundCodec.base_codec.facodec import BaseCodec


class Codec(BaseCodec):
    def config(self):
        self.model_type = "16khz"
        self.sampling_rate = 16_000
