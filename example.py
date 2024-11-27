from transformers import AutoTokenizer


class Example:
    def __init__(self, sentence, label, metadata = None):
        self.sentence = sentence
        self.label = label
        tokenizer = AutoTokenizer.from_pretrained("/large_disk/lkh/python_projects/llm/fl_project/roberta-base")
        self.p_sentence = tokenizer(sentence)
        
        self.metadata = metadata
    
    def get_label(self):
        return self.label
    
    def get_sentence(self):
        return self.p_sentence
    
    def get_aux_labels(self):
        return self.metadata
    
    
    def get_training_example(self):
        return self.p_sentence, self.label
    
    def get_aux_training(self):
        return self.p_sentence, self.metadata







