from sklearn.metrics import roc_curve,auc
import numpy as np
import matplotlib.pyplot as plt

cv=np.load('cvscore.npz')

fpr1,tpr1,t=roc_curve(cv['label'],cv['cnn1'])
fpr2,tpr2,t=roc_curve(cv['label'],cv['cnn2'])
fpr3,tpr3,t=roc_curve(cv['label'],cv['rnn1'])
fpr4,tpr4,t=roc_curve(cv['label'],cv['rnn2'])
fpr5,tpr5,t=roc_curve(cv['label'],(cv['cnn1']+cv['cnn2']+cv['rnn1']+cv['rnn2'])/5)

lw = 1
plt.subplot(121)
plt.plot(fpr1, tpr1, color='green',lw=lw, label='CNN-Binary     AUC = {:.3f}'.format(auc(fpr1,tpr1)))
plt.plot(fpr2, tpr2, color='gold',lw=lw, label= 'CNN-Property  AUC = {:.3f}'.format(auc(fpr2,tpr2)))
plt.plot(fpr3, tpr3, color='red',lw=lw, label=  'RNN-Binary     AUC = {:.3f}'.format(auc(fpr3,tpr3)))
plt.plot(fpr4, tpr4, color='blue',lw=lw,label='RNN-Property  AUC = {:.3f}'.format(auc(fpr4,tpr4)))
plt.plot(fpr5, tpr5, color='black',lw=lw, label='HUbiPred         AUC = {:.3f}'.format(auc(fpr5,tpr5)))
plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")


test=np.load('testscore.npz')
fpr1,tpr1,_=roc_curve(test['label'],test['cnn1'])
fpr2,tpr2,_=roc_curve(test['label'],test['cnn2'])
fpr3,tpr3,_=roc_curve(test['label'],test['rnn1'])
fpr4,tpr4,_=roc_curve(test['label'],test['rnn2'])
fpr5,tpr5,_=roc_curve(test['label'],(test['cnn1']+test['cnn2']+test['rnn1']+test['rnn2'])/5)
plt.subplot(122)
plt.plot(fpr1, tpr1, color='green',lw=lw, label='CNN-Binary     AUC = {:.3f}'.format(auc(fpr1,tpr1)))
plt.plot(fpr2, tpr2, color='gold',lw=lw, label= 'CNN-Property  AUC = {:.3f}'.format(auc(fpr2,tpr2)))
plt.plot(fpr3, tpr3, color='red',lw=lw, label=  'RNN-Binary     AUC = {:.3f}'.format(auc(fpr3,tpr3)))
plt.plot(fpr4, tpr4, color='blue',lw=lw,label='RNN-Property  AUC = {:.3f}'.format(auc(fpr4,tpr4)))
plt.plot(fpr5, tpr5, color='black',lw=lw, label='HUbiPred         AUC = {:.3f}'.format(auc(fpr5,tpr5)))
plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

