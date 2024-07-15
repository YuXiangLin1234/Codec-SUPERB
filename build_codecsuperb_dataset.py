from datasets import load_dataset
import argparse

def extract_audios(dataset_name):

    dataset = load_dataset(dataset_name, "zh-TW",  split='test')
    dataset = dataset.select([i for i in range(100)])
	



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Run audio encoding-decoding experiments.')
	parser.add_argument('--dataset', type=str, required=True, default="mozilla-foundation/common_voice_11_0",
						help='Name of the dataset to process in huggingface/datasets')
	parser.add_argument('--update_codec', type=str, choices=list_codec(),
						help='Name of the codec to add to the dataset')
	parser.add_argument('--extract_unit_only', required=False, action='store_true')
	parser.add_argument('--push_to_hub', required=True, action='store_true')
	parser.add_argument('--upload_name', required=False, default='Evan-Lin')
	args = parser.parse_args()
	extract_audios(args.dataset, args.push_to_hub)