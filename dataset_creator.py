import argparse
import torch
from datasets import DatasetDict, Audio, load_from_disk
from datasets import load_dataset as hf_load_dataset
from SoundCodec.codec import load_codec, list_codec
# from SoundCodec.dataset import load_dataset
from SoundCodec.dataset.general import extract_unit, apply_audio_cast

def map_file_to_id(data):
    data['id'] = "".join(data['path'].split("/")[-1:])
    return data


def run_experiment(dataset_name, sample_num=None):
    # cleaned_dataset = load_dataset(dataset_name)
    cleaned_dataset = hf_load_dataset(dataset_name, "zh-TW", split='test')
    if sample_num:
        cleaned_dataset = cleaned_dataset.select([i for i in range(sample_num)])

    d_item = next(iter(cleaned_dataset))
    sampling_rate = d_item['audio']['sampling_rate']
    # cleaned_dataset = cleaned_dataset.rename_column("path", "id")
    cleaned_dataset = hf_load_dataset(dataset_name, "zh-TW", split='test')
    if sample_num:
        cleaned_dataset = cleaned_dataset.select([i for i in range(sample_num)])

    cleaned_dataset = cleaned_dataset.map(map_file_to_id)
    # cleaned_dataset = cleaned_dataset.map(lambda example: {"audio": {"path": example["audio"]["path"],
    #                                                         "sampling_rate": example["audio"]["sampling_rate"],
    #                                                         "array": torch.from_numpy(example["audio"]["array"]) }}
    #                                                         , remove_columns=["audio"])


    print("before filter duration", cleaned_dataset)
    cleaned_dataset = cleaned_dataset.filter(
        lambda x: len(x['audio']['array']) / x['audio']['sampling_rate'] <= args.max_duration)
    print("after filter duration", cleaned_dataset)
    cleaned_dataset = apply_audio_cast(cleaned_dataset, sampling_rate)

    if not args.extract_unit_only:
        datasets_dict = DatasetDict({'original': cleaned_dataset})
    else:
        datasets_dict = DatasetDict({})
    # for codec_name in list_codec():
    # for i in range(1):
    for codec_name in ["facodec_16k", "encodec_24k_12bps"]
        # print(codec_name)
        # continue
        print(f"Synthesizing dataset with {codec_name}")
        # load from disk if already synthesized
        try:
            synthesized_dataset = load_from_disk(f"./cached_datasets/{dataset_name}_{codec_name}/")
            datasets_dict[f'{codec_name}'] = synthesized_dataset
            continue
        except:
            pass
        codec = load_codec(codec_name)
        synthesized_dataset = apply_audio_cast(cleaned_dataset, codec.sampling_rate)
        if args.extract_unit_only == 'extract_unit':
            synthesized_dataset = synthesized_dataset.map(extract_unit, fn_kwargs={'extract_unit_class': codec})
        else:
            synthesized_dataset = synthesized_dataset.map(codec.synth)
            synthesized_dataset = synthesized_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
        synthesized_dataset.save_to_disk(f"./cached_datasets/{dataset_name}_{codec_name}/")
        datasets_dict[f'{codec_name}'] = synthesized_dataset

    datasets_dict_unit_only = datasets_dict.remove_columns(['audio'])
    datasets_dict_unit_only.pop('original')
    datasets_dict_unit_only.save_to_disk(f"./datasets/{dataset_name}_unit")
    # remove datasets_dict columns if they have 'unit', and use datasets_dict_synth for saving
    datasets_dict_synth = DatasetDict({})
    for key in datasets_dict.keys():
        if 'unit' not in datasets_dict[key].column_names:
            datasets_dict_synth[key] = datasets_dict[key]
        else:
            datasets_dict_synth[key] = datasets_dict[key].remove_columns(['unit'])
    if not args.extract_unit_only:
        datasets_dict_synth.save_to_disk(f"./datasets/{dataset_name}_synth")

    if args.push_to_hub:
        push_to_hub_org = args.upload_name
        if not args.extract_unit_only:
            if args.sample_num:
                datasets_dict_synth.push_to_hub(f"{push_to_hub_org}/{dataset_name}_synth_{args.sample_num}")
            else: 
                datasets_dict_synth.push_to_hub(f"{push_to_hub_org}/{dataset_name}_synth")

        datasets_dict_unit_only.push_to_hub(f"{push_to_hub_org}/{dataset_name}_unit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run audio encoding-decoding experiments.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset to process in huggingface/datasets')
    parser.add_argument('--extract_unit_only', required=False, action='store_true')
    parser.add_argument('--max_duration', required=False, type=int, default=120)
    parser.add_argument('--sample_num', required=False, type=int, default=None)
    parser.add_argument('--push_to_hub', required=False, action='store_true')
    parser.add_argument('--upload_name', required=False, default='Evan-Lin')
    args = parser.parse_args()
    run_experiment(args.dataset, args.sample_num)
