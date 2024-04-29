import re
import torch
from PIL import Image
from ..smp import *
from ..utils import DATASET_TYPE
import sys
from .base import BaseModel
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from sentencepiece import SentencePieceProcessor

pattern = re.compile(r'[A-Z]')

meta_instruction = """
You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).
- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by
Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language
chosen by the user such as English and 中文.
- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively
based on the provided image.
"""


def encode_image(image_paths):
    images = []
    for path in image_paths:
        image_size = 490
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                    (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        img = Image.open(path).convert("RGB")
        images.append(transform(img))
    return torch.stack(images, dim=0).cuda()


class LLMTokenizer(object):

    def __init__(self, tokenizer, max_seq_len=2048, tokenizer_type='llama', mode='none'):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tokenizer_type = tokenizer_type
        self._set_special_tokens()

        assert mode in ['none', 'mid']
        self.mode = mode

    def __call__(self,
                 prompts,
                 padding=True,
                 right_align=False,
                 return_tensors='pt',
                 need_bos=True,
                 need_eos=False,
                 truncation=False):
        tokens = [self.encode(x, bos=need_bos, eos=need_eos) for x in prompts]
        if truncation:
            if self.mode == 'none':
                tokens = [i[:self.max_seq_len] for i in tokens]
            else:
                half = self.max_seq_len // 2
                new_tokens = []
                for i in tokens:
                    if len(i) <= self.max_seq_len:
                        new_tokens.append(i)
                    else:
                        new_tokens.append(i[:half] + i[-half:])
                tokens = new_tokens

        if padding:
            max_len = max([len(i) for i in tokens])
            if right_align:
                tokens = [[self.pad_token_id] * (max_len - len(i)) + i
                          for i in tokens]
            else:
                tokens = [
                    i + [self.pad_token_id] * (max_len - len(i))
                    for i in tokens
                ]

        tokens = torch.LongTensor(tokens)
        return {
            'tokens': tokens.cuda() if torch.cuda.is_available() else tokens
        }

    def encode(self, s: str, bos: bool, eos: bool):
        assert isinstance(s, str)
        s = self._process_meta_tokens(s)
        t = self._tokenize_list_str(s)
        if bos:
            t = [self.bos_token_id]+[92495] * 1225 + t
        if eos:
            t = t + [self.eos_token_id]
        return t

    def _process_meta_tokens(self, input_string: str):
        # Create a pattern to match the META_TOKEN_{NUM} substrings
        pattern = re.compile(r'<META_TOKEN_(\d+)>')

        # Split the input string using the META_TOKEN_{NUM} substrings
        parts = pattern.split(input_string)

        # Combine the parts and tokens in the correct order
        result = []
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Regular text parts
                if part != '':
                    result.append(part)
            else:  # Meta token parts
                result.append(int(part))

        return result

    def _tokenize_list_str(self, s):
        if isinstance(s, str):
            s = [s]
        assert isinstance(s, list)
        t = []
        for item in s:
            if isinstance(item, str):
                t += self.tokenizer.encode(item)
            elif isinstance(item, int):
                t.append(item)
            else:
                raise ValueError(f'Unsupported type {type(item)}')
        return t

    def decode(self, t) -> str:
        return self.tokenizer.decode(t)

    def _set_special_tokens(self):
        if self.tokenizer_type is None or self.tokenizer_type == "":
            if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                self.bos_token_id = self.tokenizer.bos_token_id
                self.eos_token_id = self.tokenizer.eos_token_id
                self.pad_token_id = self.tokenizer.pad_token_id
            else:
                self.bos_token_id = self.tokenizer.bos_id()
                self.eos_token_id = self.tokenizer.eos_id()
                self.pad_token_id = self.tokenizer.pad_id()
            if self.pad_token_id == -1:
                self.pad_token_id = self.bos_token_id
        elif self.tokenizer_type == 'v4':
            self.bos_token_id = self.pad_token_id = 0
            self.eos_token_id = 1
        elif self.tokenizer_type in ['llama', 'v7', 'baichuan2']:
            self.bos_token_id = self.pad_token_id = 1
            self.eos_token_id = 2
        elif self.tokenizer_type == "deepseek":
            self.bos_token_id = 32013
            self.eos_token_id = 32014
            self.pad_token_id = 32018
        else:
            raise NotImplementedError(f"Unknown tokenizer type {self.tokenizer_type}")

        # This is a hack to fit in with LLama type model
        self.bos_id = self.bos_token_id
        self.eos_id = self.eos_token_id
        self.pad_id = self.pad_token_id


class Internlm_train(BaseModel):

    def __init__(self, model_config=None, **kwargs):
        assert model_config is not None
        self.model_config = model_config

        interntrain_dir = self.model_config.get("module_path",None)
        assert interntrain_dir is not None
        sys.path.insert(0, interntrain_dir)
        sys.path.insert(0, os.path.join(interntrain_dir, "internlm"))
        from tools.score_tools.score_utils import initialize_internlm_model
        from internlm.apis.seq_generator_module import _no_beam_search_generate
        self._no_beam_search_generate = _no_beam_search_generate

        if self.model_config.get("specific_model_config", None) is None:
            self.model_config["specific_model_config"] = torch.load(os.path.join(self.model_config["ckpt_dir"].split(":")[-1], "model_config.pt"))
            self.model_config["specific_model_config"]['multimodal_cfg'] = dict(multimodal_token_ids=dict(image=92495))
            self.model_config["specific_model_config"]['huggingface_operator'] = False
            self.model_config["specific_model_config"]['vit_cfg'] = dict(file_path="/mnt/inspurfs/share_data/llm_data/multimodal/clip_l_336", model_type='ViT-L/14@336px', img_size=490, use_checkpoint=True)
            self.model_config["specific_model_config"]['projection_cfg'] = dict(type='mlp', vision_width=1024)
            self.model_config["specific_model_config"]['plora_cfg'] = dict(lora_r=256, lora_alpha=256)
        
        self.model = initialize_internlm_model(
            interntrain_dir=interntrain_dir,
            model_type=self.model_config.get("model_type", "INTERNLM_XCOMPOSER2"),
            load_type=self.model_config.get("load_type", "internevo"),
            ckpt_dir=self.model_config.get("ckpt_dir"),
            param_dtype=self.model_config.get("param_dtype", "torch.float16"),
            training=False,
            specific_model_config=self.model_config.get("specific_model_config", "llama_7b"),
            del_model_prefix=self.model_config.get("del_model_prefix", True),
        )
        self.model = self.model.cuda().eval()

        tokenizer = SentencePieceProcessor()
        tokenizer.load(self.model_config.get("tokenizer_path"))

        self.tokenizer = LLMTokenizer(tokenizer)

    def no_beam_search_generate(self, prompts, image_path, need_bos=True, padding=True, beams=3, max_token=500, dataset=None):
        images = encode_image(image_path)
        prompts = [prompts]
        images = [images]
        generation_kwargs = {
            'max_gen_len': 100,
            'eos_token_id': None,
            'temperature': 1.0,
            'top_p': 1.0,
            'top_k': 50,
            'do_sample': False,
            'repetition_penalty': 1.0,
            'min_new_tokens': max_token,
        }
        tokenized_data = self.tokenizer(
            prompts,
            padding=padding,
            right_align=True,
            return_tensors='pt',
            need_bos=need_bos,
        )
        tokenized_data_len = tokenized_data["tokens"].shape[0]
        padding_data = tokenized_data['tokens'].tolist()
        eos_token_id = generation_kwargs.get('eos_token_id')

        results = self._no_beam_search_generate(
            self.model,
            tokenized_data['tokens'][...,],
            images,
            do_sample=generation_kwargs['do_sample'],
            max_length=generation_kwargs['max_gen_len'] + tokenized_data_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
            top_p=generation_kwargs['top_p'],
            top_k=generation_kwargs['top_k'],
            temperature=generation_kwargs['temperature'],
            repetition_penalty=generation_kwargs['repetition_penalty'],
        )
        results = results.squeeze(1).tolist()
        # for i in range(len(prompts)):
        single_res = results[0][len(padding_data[0]):]
        if eos_token_id is not None:
            try:
                single_res = single_res[:single_res.index(eos_token_id)]
            except ValueError:
                pass
        results_text = self.tokenizer.tokenizer.decode(single_res)
        return results_text

    def generate_mme(self, image_path, text):
        text = text.split('Please answer')[0].strip()
        text = f'{text} Answer this question briefly'
        text = f'[UNUSED_TOKEN_146]user\n{text}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'

        return self.no_beam_search_generate(text, image_path, need_bos=True, padding=True, beams=5)

    def generate_multichoice(self, image_path, text, dataset):
        out = self.no_beam_search_generate(text, image_path, need_bos=True, padding=False, beams=5, max_token=5)
        if 'mmmu' in dataset.lower():
            return out
        res = pattern.findall(out)
        if len(res) == 0:
            print('Error:', out)
            res = 'Z'
        return res[0]

    def generate_vqa(self, image_path, text):
        out = self.no_beam_search_generate(text, image_path, need_bos=True)
        return out

    def generate_vanilla(self, image_path, text):
        text = (
            '[UNUSED_TOKEN_146]system\n{}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]user\n{}'
            'Answer this question in detail.[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
        ).format(meta_instruction, text)
        out = self.no_beam_search_generate(text, image_path, need_bos=True, max_token=500)
        return out

    def generate_brief(self, image_path, text):
        text = (
            '[UNUSED_TOKEN_146]user\nAnswer the question using a single word or phrase.{}'
            '[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
        ).format(text)
        out = self.no_beam_search_generate(text, image_path, need_bos=True, max_token=10)
        return out

    def generate_driectly(self, image_path, text):
        text = '[UNUSED_TOKEN_146]user\n{}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'.format(text)
        out = self.no_beam_search_generate(text, image_path, need_bos=True, max_token=500)
        return out

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message)
        with torch.cuda.amp.autocast():
            if dataset is None:
                return self.generate_vanilla(image_path, prompt)
            assert isinstance(dataset, str)
            if dataset == 'MME':
                return self.generate_mme(image_path, prompt)

            elif listinstr(['hallu'], dataset.lower()):
                return self.generate_brief(image_path, prompt)

            elif listinstr(['llava'], dataset.lower()):
                return self.generate_vanilla(image_path, prompt)

            elif dataset is not None and DATASET_TYPE(dataset) == 'multi-choice':
                return self.generate_multichoice(image_path, prompt, dataset)

            elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
                return self.generate_vqa(image_path, prompt)

            else:
                return self.generate_vanilla(image_path, prompt)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice' or DATASET_TYPE(dataset) == 'VQA':
            return True
        return False

    def build_mcqa(self, line):
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        img_prompt = '[UNUSED_TOKEN_146]user\n'
        if len(options):
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'{key}. {item} '
            options_prompt = options_prompt.strip()
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None

            context = 'N/A' if hint is None else hint
            mid_prompt = 'Question: ' + question + '\nContext: ' + context + '\nOptions: ' + options_prompt
            ans_prompt = '[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\nThe answer is'
            prompt = img_prompt + mid_prompt + ans_prompt
        else:
            mid_prompt = f'Answer the question using a single word or phrase.{question}'
            ans_prompt = '[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
            prompt = img_prompt + mid_prompt + ans_prompt

        return prompt

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        if DATASET_TYPE(dataset) == 'multi-choice':
            prompt = self.build_mcqa(line)
        elif DATASET_TYPE(dataset) == 'VQA':
            if 'mathvista' in dataset.lower():
                q = line['question']
                prompt = f'[UNUSED_TOKEN_146]user\n{q}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
            else:
                q = line['question']
                prompt = (
                    f'[UNUSED_TOKEN_146]user\nAnswer the question using a single word or phrase.{q}'
                    '[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
                )
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message
    
    def message_to_promptimg(self, message):
        num_images = len([x for x in message if x['type'] == 'image'])
        if num_images == 0:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = []
        elif num_images == 1:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = [x['value'] for x in message if x['type'] == 'image']
        else:
            prompt = '\n'.join([x['value'] if x['type'] == 'text' else '<image>' for x in message])
            image = [x['value'] for x in message if x['type'] == 'image']
        return prompt, image
