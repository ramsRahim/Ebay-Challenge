from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification
from transformers import AutoTokenizer, XLMRobertaXLForTokenClassification
from transformers import BertTokenizer, BertForTokenClassification


def get_tokenizer(model_type):
    if model_type == 'large':
        tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-large')
    elif model_type == 'xlarge':
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-xlarge")
        
    # tokenizer = BertTokenizer.from_pretrained('deepset/gbert-large')
    # tokenizer = AutoTokenizer.from_pretrained('facebook/xlm-roberta-xl')
    return tokenizer

def get_model(model_type, num_labels):
    if model_type == 'large':
        model = XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-large', num_labels=num_labels)
    elif model_type == 'xlarge':
        model = XLMRobertaXLForTokenClassification.from_pretrained("xlm-roberta-xlarge", num_labels=num_labels)
        
        # tokenizer = BertTokenizer.from_pretrained('gbert-large')
    # model = BertForTokenClassification.from_pretrained('deepset/gbert-large', num_labels=num_labels)
    # model = XLMRobertaXLForTokenClassification.from_pretrained("facebook/xlm-roberta-xl", num_labels=num_labels)
    return model

def get_model_tokenizer(model_type, num_labels):
    tokenizer = get_tokenizer(model_type)
    model = get_model(model_type, num_labels)
    return model, tokenizer