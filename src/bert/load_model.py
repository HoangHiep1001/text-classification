from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

phoBERT_cls = RobertaModel.from_pretrained('PhoBERT_base_fairseq', checkpoint_file='model.pt')
phoBERT_cls.eval()  # disable dropout (or leave in train mode to finetune

# Load BPE
class BPE():
  bpe_codes = 'PhoBERT_base_fairseq/bpe.codes'

args = BPE()
phoBERT_cls.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT

# Add header cho classification với số lượng classes = 10
phoBERT_cls.register_classification_head('new_task', num_classes=10)
tokens = 'Học_sinh được nghỉ học bắt đầu từ tháng 3 do ảnh hưởng của dịch covid-19'
token_idxs = phoBERT_cls.encode(tokens)
logprobs = phoBERT_cls.predict('new_task', token_idxs)  # tensor([[-1.1050, -1.0672, -1.1245]], grad_fn=<LogSoftmaxBackward>)
print(logprobs)