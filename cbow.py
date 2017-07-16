#CBOW

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

CONTEXT_SIZE = 2
EMBED_SIZE = 10
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()


vocab = set(raw_text)

word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
print(word_to_idx)
context_target = [ ([raw_text[i-2], raw_text[i-1], raw_text[i+1] , raw_text[i+2]], raw_text[i]) for i in range(2, len(raw_text)-2)]

# print(context_target)

# print(word_to_idx)
class CBOWClassifier(nn.Module):

	def __init__ (self, vocab_size, embed_size, context_size):
		super(CBOWClassifier,self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embed_size)
		self.linear1 = nn.Linear(embed_size, 128)
		self.linear2 = nn.Linear(128, vocab_size)

	def forward(self, inputs):
		embed = self.embeddings(inputs)
		embed = torch.sum(embed, dim=0)
		out = self.linear1(embed)
		out = F.relu(out)
		out = self.linear2(out)
		log_probs = F.log_softmax(out)
		return log_probs

VOCAB_SIZE = len(word_to_idx)

model = CBOWClassifier(VOCAB_SIZE, EMBED_SIZE, 2*CONTEXT_SIZE)
losses = []
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epochs in range(100):
	total_loss = torch.Tensor([0])
	for context, target in context_target:

		context_idx = [word_to_idx[w] for w in context]
		context_var = autograd.Variable(torch.LongTensor(context_idx))
		model.zero_grad()
		log_probs = model(context_var)
		target_idx = word_to_idx[target]
		loss = loss_function(log_probs, autograd.Variable(torch.LongTensor([target_idx])))
		loss.backward()
		optimizer.step()
		total_loss = total_loss + loss.data
	losses.append(total_loss)

print(losses)
print('Training done..')
# Testing the model
test_sentence = "computer spirits conjure by spells.".split()
context_target = [ ([test_sentence[i-2], test_sentence[i-1], test_sentence[i+1] , test_sentence[i+2]], test_sentence[i]) for i in range(2, len(test_sentence)-2)]
# print (context_target)
for ctx_word, target in context_target:
	test_context_idx  = [word_to_idx[w] for w in ctx_word]

test_context_var = autograd.Variable(torch.LongTensor(test_context_idx))
test_log_probs = model(test_context_var)
# print(test_log_probs)
maximum, index = torch.max(test_log_probs, dim=1)
# print (index[0].data[0])
print ('Predicted word:' + idx_to_word[index[0].data[0]])
