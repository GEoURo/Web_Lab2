#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import io
import jsonlines
import numpy as np
from BiLSTM_CRF import *


# In[ ]:


train_file_path = './subtask1_training_part1.json'
test_file_path = './test_data.json'
with io.open(train_file_path, 'r', encoding='utf-8-sig') as json_file:
    raw_train_data = list(jsonlines.Reader(json_file))


# In[ ]:


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 10
HIDDEN_DIM = 4
torch.manual_seed(1)


# In[ ]:


def select_label_type_from_text(text, text_label, label_type):
    if len(text) != len(text_label):
        print("The text length does not match the label length!")
        return

    for i in range(len(text_label)):
        if text_label[i].endswith(bytes(str(label_type), encoding='utf-8')):
            string = str(text_label[i], encoding='utf-8')
            text_label[i] = bytes(string[0] + '-LOC', encoding='utf-8')
        else:
            text_label[i] = b"O"
    return text, text_label


# In[ ]:


label_dict = {'疾病和诊断': 'disease', '手术':'operation', 
              '解剖部位':'dissection', '药物':'drug', 
              '影像检查':'imgexam', '实验室检验':'labexam'}


# In[ ]:


processed_data = {}
textID = 0

for data in raw_train_data:
    data = raw_train_data[0]   
    raw_text = data['originalText']
    text = np.array(['' for i in range(len(raw_text))], dtype='object')
    text_tag = np.array(['O' for i in range(len(raw_text))], dtype='object')
    entities = data['entities']

    
    for entity in entities:
        # Get the start and end pos of each entity
        startpos = entity['start_pos']
        endpos = entity['end_pos']
        # Mark the label with the pos info from above
        text_tag[startpos] = "B" + '_' + label_dict[entity['label_type']]
        text_tag[startpos+1:endpos] = "I" + '_' + label_dict[entity['label_type']]

    for i, word in enumerate(raw_text):
        text[i] = word
    
    processed_data[textID] = [text, text_tag]
    textID += 1


# In[ ]:


word_to_ix = {}
for sentence, tags in processed_data.values():
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)


# In[ ]:


tag_to_ix = {"B_disease":0, "I_disease":1, "B_operation":2, "I_operation":3, 
             "B_dissection":4, "I_dissection":5, "B_drug":6, "I_drug":7, 
             "B_imgexam":8, "I_imgexam":9, "B_labexam":10, "I_labexam":11, 
             'O':12, START_TAG:13, STOP_TAG:14}


# In[ ]:


model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)


# In[ ]:


train_data = list(processed_data.values())

with torch.no_grad():
    precheck_sent = prepare_sequence(train_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in train_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))


# In[ ]:


EPOCH = 30
if torch.cuda.is_available():
    model.cuda()
    
for epoch in range(EPOCH):
    for sentence, tags in train_data:
        model.zero_grad()

        # Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        if torch.cuda.is_available():
            sentence_in = sentence_in.cuda()
            targets = targets.cuda()
            
        # Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
    if epoch % 5 == 0:
        print("%d/%d epochs of training complete. Loss:%.6f" % (epoch, EPOCH, loss))

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(train_data[0][0], word_to_ix)
    print(model(precheck_sent))

