import numpy as np
import torch
from CNN import model_clouds,model_shadows
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

#%% load data and run the trained adapted Cloud-Net model
target='clouds' # choose 'clouds' or 'shadows'
folder_root="" #Select root folder

features_testing=np.load(folder_root+"X_testing_2A_"+target)
labels_testing=np.load(folder_root+"Y_testing_2A_"+target)
TC_testing=np.load(folder_root+"Z_testing_2A_"+target)
SEN2COR_pred=np.load(folder_root+"SEN2COR_testing_2A_"+target)

X_testing=torch.tensor(features_testing,dtype=torch.float,requires_grad=True)
Y_testing=torch.tensor(labels_testing,dtype=torch.float,requires_grad=False)
Z_testing=TC_testing

if target=='clouds':
    model=model_clouds 

if target=='shadows':
    model=model_shadows  

model.load_state_dict(torch.load("/pretrained_models/"+target))

y_pred_full=[]

for i in range(X_testing.shape[0]):
    y_test=model(X_testing[[i],:,:,:])
    y_test_round=torch.round(y_test)
    y_test_2=y_test_round.detach().numpy()
    y_pred=np.reshape(y_test_2,(X_testing.shape[2],X_testing.shape[3])) 
    y_pred_full.append(y_pred)
    
    y_pred_plot=np.array(y_pred)
    y_pred_plot[y_pred_plot==0]=np.nan
    
#%% Select index of interest to check for individual performances
test_index=0

plt.figure(0)
plt.title('Adapted Cloud-Net')
plt.imshow(y_pred_full[test_index])

plt.figure(1)
plt.title('Manually annotated labels')
plt.imshow(Y_testing[test_index,:,:])

plt.figure(2)
plt.title('True color image')
plt.imshow(Z_testing[test_index,:,:,:]/255)

plt.figure(3)
plt.title('SEN2COR')
plt.imshow(SEN2COR_pred[test_index,:,:])


#%% Optional: estimation of final accuracy (only possible when image is labelled)

score_test_full=[]
final_acc_full=[]
CNN_precision_full=[]
CNN_recall_full=[]
CNN_F1_full=[]


score_test_SEN2COR_full=[]
final_acc_SEN2COR_full=[]
SEN2COR_precision_full=[]
SEN2COR_recall_full=[]
SEN2COR_F1_full=[]

y_pred_full_column=np.zeros((0,1))
label_full_column=np.zeros((0,1))
SEN2COR_full_column=np.zeros((0,1))

for i in range((X_testing.shape[0])):
    y_pred_column=np.reshape(y_pred_full[i],[X_testing.shape[2]*X_testing.shape[3],1])
    SEN2COR_pred_column=np.reshape(SEN2COR_pred[[i],:,:],[X_testing.shape[2]*X_testing.shape[3],1])
    label_column=np.asarray(np.reshape(Y_testing[[i],:,:],[X_testing.shape[2]*X_testing.shape[3],1]))

    y_pred_full_column=np.concatenate((y_pred_full_column,y_pred_column))
    label_full_column=np.concatenate((label_full_column,label_column))
    SEN2COR_full_column=np.concatenate((SEN2COR_full_column,SEN2COR_pred_column))
    
    cm=confusion_matrix(label_column,y_pred_column,labels=[1, 0])
    cmn = cm.astype('float') /cm.sum(axis=1)[:, np.newaxis]
    CNN_precision=cm[0,0]/(cm[0,0]+cm[1,0])
    CNN_precision_full.append(CNN_precision)
    CNN_recall=cm[0,0]/(cm[0,0]+cm[0,1])
    CNN_recall_full.append(CNN_recall)
    CNN_F1=2*CNN_precision*CNN_recall/(CNN_precision+CNN_recall)
    CNN_F1_full.append(CNN_F1)
    
    cm=confusion_matrix(label_column,SEN2COR_pred_column,labels=[1, 0])
    cmn = cm.astype('float') /cm.sum(axis=1)[:, np.newaxis]
    SEN2COR_precision=cm[0,0]/(cm[0,0]+cm[1,0])
    SEN2COR_precision_full.append(SEN2COR_precision)
    SEN2COR_recall=cm[0,0]/(cm[0,0]+cm[0,1])
    SEN2COR_recall_full.append(SEN2COR_recall)
    SEN2COR_F1=2*SEN2COR_precision*SEN2COR_recall/(SEN2COR_precision+SEN2COR_recall)
    SEN2COR_F1_full.append(SEN2COR_F1)
    
    score_test=0
    for j in range(len(y_pred_column)): 
        if y_pred_column[j]==label_column[j]:
            score_test=score_test+1    
    score_test_full.append(score_test)
    
    score_test_SEN2COR=0
    for j in range(len(SEN2COR_pred_column)): 
        if SEN2COR_pred_column[j]==label_column[j]:
            score_test_SEN2COR=score_test_SEN2COR+1    
    score_test_SEN2COR_full.append(score_test_SEN2COR)        


    final_acc=round(score_test/len(y_pred_column),3) 
    final_acc_str=str(np.round(100*final_acc,3))
    final_acc_full.append(final_acc)
    
    final_acc_SEN2COR=round(score_test_SEN2COR/len(SEN2COR_pred_column),3) 
    final_acc_SEN2CORstr=str(np.round(100*final_acc_SEN2COR,3))
    final_acc_SEN2COR_full.append(final_acc_SEN2COR)

test_acc=100*sum(score_test_full)/(X_testing.shape[0]*X_testing.shape[2]*X_testing.shape[3])
test_SEN2COR_acc=100*sum(score_test_SEN2COR_full)/(X_testing.shape[0]*X_testing.shape[2]*X_testing.shape[3])
test_acc_str=str(np.round(test_acc,1))
test_SEN2COR_str=str(np.round(test_SEN2COR_acc,1))


#%% SEN2COR scores
cm=confusion_matrix(label_full_column,SEN2COR_full_column,labels=[1, 0])
cmn = cm.astype('float') /cm.sum(axis=1)[:, np.newaxis]

SEN2COR_precision=cm[0,0]/(cm[0,0]+cm[1,0])
SEN2COR_recall=cm[0,0]/(cm[0,0]+cm[0,1])
SEN2COR_F1=2*SEN2COR_precision*SEN2COR_recall/(SEN2COR_precision+SEN2COR_recall)

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True ,fmt='.2f',annot_kws={'size':20},cbar=False, xticklabels=['shadow','no_shadow'], yticklabels=['shadow','no_shadow'])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('SEN2COR Final acc.='+test_SEN2COR_str+'%',fontsize=25)
plt.ylabel('Actual label', size=23)
plt.xlabel('Predicted label', size=23)
plt.show(block=False)

#%% CNN scores
cm=confusion_matrix(label_full_column, y_pred_full_column,labels=[1, 0])
cmn = cm.astype('float') /cm.sum(axis=1)[:, np.newaxis]

CNN_precision=cm[0,0]/(cm[0,0]+cm[1,0])
CNN_recall=cm[0,0]/(cm[0,0]+cm[0,1])
CNN_F1=2*CNN_precision*CNN_recall/(CNN_precision+CNN_recall)

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True ,fmt='.2f',annot_kws={'size':20},cbar=False, xticklabels=['shadow','no_shadow'], yticklabels=['shadow','no_shadow'])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Cloud-Net Final acc.='+test_acc_str+'%',fontsize=25)
plt.ylabel('Actual label', size=23)
plt.xlabel('Predicted label', size=23)
plt.show(block=False)


