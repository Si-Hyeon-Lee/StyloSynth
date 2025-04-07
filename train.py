import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def train(img_list, gt_list, model, epochs, learning_rate,criterion,optimizer,batch_size):
    # 디바이스 설정 및 모델 준비
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # numpy 배열을 torch 텐서로 변환
    img_tensor = torch.tensor(np.array(img_list), dtype=torch.float32)
    gt_tensor = torch.tensor(np.array(gt_list), dtype=torch.long)

    # DataLoader 생성
    dataset = TensorDataset(img_tensor[:9900], gt_tensor[:9900]) 
    #dataset = TensorDataset(img_tensor, gt_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_set = TensorDataset(img_tensor[9900:], gt_tensor[9900:])
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    ap_metric = torchmetrics.AveragePrecision(task='multiclass', num_classes=7, average='macro').to(device) # macro 면 mean Average Precision.
    iou_metric = torchmetrics.JaccardIndex(task="multiclass", num_classes=7,).to(device)

    model.train()
    best_mIOU = 0.8
    for epoch in range(epochs):
        running_loss = 0.0

        for batch_img, batch_gt in dataloader:
            batch_img, batch_gt = batch_img.to(device), batch_gt.to(device).squeeze(1)
            #print(batch_img.shape , batch_gt.shape)
             #torch.Size([32, 3, 128, 128]) torch.Size([32, 128, 128])

            optimizer.zero_grad()

            # 순전파 (Forward pass)
            output = model(batch_img)

            loss = criterion(output, batch_gt)

            # 역전파 (Backward pass) 및 최적화
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        
        
        print(f'Epoch [{epoch + 1}/{epochs}] Train Loss : {avg_loss}')
        model.eval()
        with torch.no_grad():
          gt_li = []
          prob_li = []
          pred_li = []
          for img, tar in val_loader :
            imgs = img.to(device)
            tars = tar.to(device).squeeze(1)
            #print(imgs.shape , tars.shape) # torch.Size([100, 3, 128, 128]) torch.Size([100, 128, 128])
            outputs = model(imgs)

            probs = torch.softmax(outputs,dim=1)
            preds = torch.argmax(probs,dim=1)
            val_loss = criterion(outputs, tars)

            gt_li.append(tars)
            prob_li.append(probs)
            pred_li.append(preds)
          all_gt = torch.cat(gt_li,dim=0)
          all_porb = torch.cat(prob_li,dim=0)
          all_pred = torch.cat(pred_li,dim=0)
          #print(all_gt.shape ,all_pred.shape,all_porb.shape)
          mAP = ap_metric(all_porb,all_gt)
          mIOU = iou_metric(all_pred,all_gt)
          print(f'Val Loss: {val_loss:.4f}, Val_mAP:{mAP:.4f}, Val_mIOU : {mIOU:.4f}')
          if mIOU > best_mIOU :
            torch.save(model.state_dict(), f'./model_save/Resnet34_unet.pth')
            best_mIOU =mIOU
          
    print('Training Done.')
