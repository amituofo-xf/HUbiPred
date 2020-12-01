import numpy as np
from tensorflow.keras import models,layers,optimizers,regularizers
from sklearn.metrics import roc_curve,auc
import pandas as pd

aminoacids='ARNDCQEGHILKMFPSTWYV-'
aaindex=pd.read_table('aaindex31',sep='\s+',header=None)
aaindex=aaindex.subtract(aaindex.min(axis=1),axis=0).divide((aaindex.max(axis=1)-aaindex.min(axis=1)),axis=0)
aa=[x for x in 'ARNDCQEGHILKMFPSTWYV']
aaindex=aaindex.to_numpy().T
index={x:y for x,y in zip(aa,aaindex.tolist())}
index['-']=np.zeros(31).tolist()
index['X']=np.zeros(31).tolist()


def index_encode(file):
    encoding=[]
    label=[]
    f=open(file,'r')
    for line in f:
        col=line.strip().split('\t')
        s=col[0]
        encoding.append([index[x] for x in (s[0:13]+s[14:])])
        label.append(col[-1])
    f.close()
    encoding=np.array(encoding)
    label=np.array(label).astype('float32')
    return encoding,label


def binary_encode(file):        
    aa2v={x:y for x,y in zip(aminoacids,np.eye(21,21).tolist())}
    aa2v['X']=np.zeros(21)
    encoding=[]
    label=[]
    f=open(file,'r')
    for line in f:
        col=line.strip().split('\t')
        s=col[0]
        encoding.append([aa2v[x] for x in (s[0:13]+s[14:])])
        label.append(col[-1])
    f.close()
    encoding=np.array(encoding)
    label=np.array(label).astype('float32')
    return encoding,label

def cnn(depth=21,l1=16,l2=512,gamma=1e-4,lr=1e-3,w1=3,w2=2):
    model=models.Sequential()
    model.add(layers.Conv1D(l1,w1,activation='relu',kernel_regularizer=regularizers.l1(gamma),input_shape=(26,depth),padding='same'))
    model.add(layers.MaxPooling1D(w2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
    model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l1(gamma)))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=lr),
                  metrics=['acc'])
    return model

def rnn(depth=21,l1=16,l2=128,gamma=1e-4,lr=1e-3,w1=3,w2=2):
    model=models.Sequential()
    model.add(layers.LSTM(l1,return_sequences=True,input_shape=(26,depth)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
    model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l1(gamma)))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=lr),
                  metrics=['acc'])
    return model

k=5
np.random.seed(1234)
binaryCoding,label=binary_encode('./trainset')
aaindexCoding,label=index_encode('./trainset')

num=len(label)
mode1=np.arange(num/2)%k
mode2=np.arange(num/2)%k
np.random.shuffle(mode1)
np.random.shuffle(mode2)
mode=np.concatenate((mode1,mode2))
score_binary_cnn=np.zeros(num)
score_binary_rnn=np.zeros(num)
score_aaindex_cnn=np.zeros(num)
score_aaindex_rnn=np.zeros(num)

for fold in range(k):
    
    trainLabel=label[mode!=fold]
    testLabel=label[mode==fold]
    
    trainFeature1=binaryCoding[mode!=fold]
    testFeature1=binaryCoding[mode==fold]

    trainFeature2=aaindexCoding[mode!=fold]
    testFeature2=aaindexCoding[mode==fold]
    #test_aaindex=trainCoding_aaindex[mode==fold]
    
    m1=cnn(l1=32,l2=256)
    m1.fit(trainFeature1,trainLabel,batch_size=100,epochs=50,verbose=0)
    score_binary_cnn[mode==fold]=m1.predict(testFeature1).reshape(len(testFeature1))
    
    m2=rnn(l1=16,l2=512)
    m2.fit(trainFeature1,trainLabel,batch_size=100,epochs=50,verbose=0)
    score_binary_rnn[mode==fold]=m2.predict(testFeature1).reshape(len(testFeature1))



    m3=cnn(depth=31,l1=32,l2=128)
    m3.fit(trainFeature2,trainLabel,batch_size=100,epochs=100,verbose=0)
    score_aaindex_cnn[mode==fold]=m3.predict(testFeature2).reshape(len(testFeature2))

    m4=rnn(depth=31,l1=16,l2=512)
    m4.fit(trainFeature2,trainLabel,batch_size=100,epochs=50,verbose=0)
    score_aaindex_rnn[mode==fold]=m4.predict(testFeature2).reshape(len(testFeature2))
    

np.savez('cvscore.npz',cnn1=score_binary_cnn,rnn1=score_binary_rnn,
         cnn2=score_aaindex_cnn,rnn2=score_aaindex_rnn,label=label)



binaryCodingTest,labelTest=binary_encode('./testset')
aaindexCodingTest,_=index_encode('./testset')

m1=cnn(l1=32,l2=256)
m1.fit(binaryCoding,label,batch_size=100,epochs=50,verbose=0)
score_binary_cnn_test=m1.predict(binaryCodingTest).reshape(len(binaryCodingTest))
    
m2=rnn(l1=16,l2=512)
m2.fit(binaryCoding,label,batch_size=100,epochs=50,verbose=0)
score_binary_rnn_test=m2.predict(binaryCodingTest).reshape(len(binaryCodingTest))

m3=cnn(depth=31,l1=32,l2=128)
m3.fit(aaindexCoding,label,batch_size=100,epochs=100,verbose=0)
score_aaindex_cnn_test=m3.predict(aaindexCodingTest).reshape(len(aaindexCodingTest))

m4=rnn(depth=31,l1=16,l2=512)
m4.fit(aaindexCoding,label,batch_size=100,epochs=50,verbose=0)
score_aaindex_rnn_test=m4.predict(aaindexCodingTest).reshape(len(aaindexCodingTest))

np.savez('testscore.npz',cnn1=score_binary_cnn_test,rnn1=score_binary_rnn_test,
         cnn2=score_aaindex_cnn_test,rnn2=score_aaindex_rnn_test,label=labelTest)
