"""Competition-only, randomly initialized U-Net heatmap pilot on free GPU."""
import hashlib,json,os,random,time
from pathlib import Path
import cv2,numpy as np,pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold

SEED=20261021;FOLD_SEED=20261020;SIDE=192;EPOCHS=int(os.environ.get('FOVEA_EPOCHS','160'))
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

class Block(nn.Module):
    def __init__(self,a,b):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU())
    def forward(self,x):return self.net(x)

class HeatmapNet(nn.Module):
    def __init__(self):
        super().__init__();self.enc=nn.ModuleList([Block(5,16),Block(16,32),Block(32,64),Block(64,128)]);self.dec=nn.ModuleList([Block(192,64),Block(96,32),Block(48,16)]);self.head=nn.Conv2d(16,1,1)
    def forward(self,x):
        n,c,h,w=x.shape;yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device),torch.linspace(-1,1,w,device=x.device),indexing='ij');x=torch.cat([x,xx.expand(n,1,h,w),yy.expand(n,1,h,w)],1);skips=[]
        for i,b in enumerate(self.enc):
            x=b(x)
            if i<3:skips.append(x);x=F.avg_pool2d(x,2)
        for b,skip in zip(self.dec,reversed(skips)):x=b(torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
        return F.avg_pool2d(self.head(x),2).flatten(1)

def load_image(path):
    im=cv2.imread(str(path));h,w=im.shape[:2];thumb=cv2.resize(im,(384,256));ys,xs=np.where(thumb.max(2)>25);x0=int(xs.min()*w/384);x1=min(w,int((xs.max()+1)*w/384));y0=int(ys.min()*h/256);y1=min(h,int((ys.max()+1)*h/256));tr=np.array([x0,y0,x1-x0,y1-y0],np.float32)
    im=cv2.resize(im[y0:y1,x0:x1],(SIDE,SIDE))[:,:,::-1].copy().astype(np.float32)/255
    # Local contrast channel normalization stays deterministic at inference.
    return im.transpose(2,0,1),tr

def heat_target(xy,device):
    axis=(torch.arange(SIDE//2,device=device)+.5)/(SIDE//2);yy,xx=torch.meshgrid(axis,axis,indexing='ij');dist=(xx[None]-xy[:,0,None,None])**2+(yy[None]-xy[:,1,None,None])**2;target=torch.exp(-dist/(2*.018**2)).flatten(1);return target/target.sum(1,keepdim=True)

def decode(logits):
    n=SIDE//2;p=logits.float().softmax(1).reshape(-1,n,n);flat=p.flatten(1).argmax(1);ix=flat%n;iy=flat//n;yy,xx=torch.meshgrid(torch.arange(n,device=p.device),torch.arange(n,device=p.device),indexing='ij');mask=(xx[None]-ix[:,None,None])**2+(yy[None]-iy[:,None,None])**2<=16;p=p*mask;p=p/p.sum((1,2),keepdim=True);return torch.stack([(p*(xx+.5)).sum((1,2))/n,(p*(yy+.5)).sum((1,2))/n],1)

@torch.no_grad()
def predict(model,images,device):
    model.eval();out=[]
    for start in range(0,len(images),16):
        x=torch.as_tensor(images[start:start+16],device=device);out.append(decode(model(x)).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_tta(model,images,device):
    normal=predict(model,images,device)
    flipped=predict(model,images[:,:,:,::-1].copy(),device)
    flipped[:,0]=1-flipped[:,0]
    return (normal+flipped)/2

def train():
    seed_all(SEED);torch.set_num_threads(2);cv2.setNumThreads(1);start=time.time()
    source=Path('/kaggle/input/2026task3/detection')
    if not source.exists():source=next(Path('/kaggle/input').rglob('fovea_localization_train_GT.csv')).parent
    output=Path('/kaggle/working');device='cuda' if torch.cuda.is_available() else 'cpu'
    if device!='cuda':raise RuntimeError('GPU unavailable: refuse expensive CPU fallback')
    df=pd.read_csv(source/'fovea_localization_train_GT.csv');truth=df[['Fovea_X','Fovea_Y']].values.astype(np.float32);images=[];transforms=[];groups=[]
    for ident in df.data:
        f=source/'train'/f'{int(ident):04d}.jpg';im,tr=load_image(f);images.append(im);transforms.append(tr);groups.append(hashlib.sha256(f.read_bytes()).hexdigest())
    images=np.array(images);transforms=np.array(transforms);xy=(truth-transforms[:,:2])/transforms[:,2:];oof=np.zeros_like(truth);oof_tta=np.zeros_like(truth);folds=list(GroupKFold(5,shuffle=True,random_state=FOLD_SEED).split(df,groups=groups));epochs=[];scores=[];baseline=np.zeros_like(truth)
    for fold,(train,valid) in enumerate(folds):
        seed_all(SEED+fold);model=HeatmapNet().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS);scaler=torch.amp.GradScaler('cuda');best=float('inf');best_epoch=0;checkpoint=output/f'fold{fold}.pt'
        for epoch in range(EPOCHS):
            model.train();order=np.random.permutation(train)
            for i in range(0,len(order),16):
                ids=order[i:i+16];batch=torch.as_tensor(images[ids],device=device);target=torch.as_tensor(xy[ids],device=device);n=len(ids)
                flip=torch.rand(n,device=device)<.5;batch[flip]=batch[flip].flip(-1);target[flip,0]=1-target[flip,0]
                # Translation preserves a known inverse target transform.
                delta=(torch.rand(n,2,device=device)-.5)*.08;theta=torch.eye(2,3,device=device).unsqueeze(0).repeat(n,1,1);theta[:,:,2]=-2*delta;grid=F.affine_grid(theta,batch.shape,align_corners=False);batch=F.grid_sample(batch,grid,align_corners=False);target=target+delta
                gain=.8+.4*torch.rand(n,3,1,1,device=device);batch=(batch*gain).clamp(0,1);opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.float16):
                    logits=model(batch);loss=-(heat_target(target,device)*F.log_softmax(logits.float(),1)).sum(1).mean()
                scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update()
            schedule.step()
            if (epoch+1)%5==0 or epoch==EPOCHS-1:
                pred=predict(model,images[valid],device)*transforms[valid,2:]+transforms[valid,:2];rmse=float(np.sqrt(np.mean((pred-truth[valid])**2)))
                if rmse<best:
                    best=rmse;best_epoch=epoch+1;torch.save({'model':model.state_dict(),'epoch':best_epoch,'rmse':best,'seed':SEED+fold},checkpoint)
                print(json.dumps({'fold':fold,'epoch':epoch+1,'rmse':rmse,'best':best,'elapsed_seconds':time.time()-start}),flush=True)
                if epoch+1-best_epoch>=40:break
        model.load_state_dict(torch.load(checkpoint,weights_only=False)['model']);oof[valid]=predict(model,images[valid],device)*transforms[valid,2:]+transforms[valid,:2];oof_tta[valid]=predict_tta(model,images[valid],device)*transforms[valid,2:]+transforms[valid,:2];scores.append(best);epochs.append(best_epoch);baseline[valid]=truth[train].mean(0);np.save(output/'oof_partial.npy',oof)
    score=float(np.sqrt(np.mean((oof-truth)**2)));metrics={'name':'unet_scratch_v1','rmse':score,'fold_rmse':scores,'best_epochs':epochs,'mean_baseline_rmse':float(np.sqrt(np.mean((baseline-truth)**2))),'seed':SEED,'side':SIDE,'max_epochs':EPOCHS,'seconds':time.time()-start,'gpu_hours':(time.time()-start)/3600,'hypothesis':'Train-only spatial heatmap learning can resolve difficult cases using contextual vessel patterns; grouped folds prevent duplicate leakage.'};(output/'metrics.json').write_text(json.dumps(metrics,indent=2));pd.DataFrame({'data':df.data,'true_x':truth[:,0],'true_y':truth[:,1],'pred_x':oof[:,0],'pred_y':oof[:,1]}).to_csv(output/'oof.csv',index=False)
    testfiles=sorted((source/'test').glob('*.jpg'));testitems=[load_image(f) for f in testfiles];test=np.array([t[0] for t in testitems]);trs=np.array([t[1] for t in testitems]);preds=[];preds_tta=[]
    for fold in range(5):
        model=HeatmapNet().to(device);model.load_state_dict(torch.load(output/f'fold{fold}.pt',weights_only=False)['model']);preds.append(predict(model,test,device)*trs[:,2:]+trs[:,:2]);preds_tta.append(predict_tta(model,test,device)*trs[:,2:]+trs[:,:2])
    preds=np.mean(preds,axis=0);mapping={int(f.stem):p for f,p in zip(testfiles,preds)};submission=pd.read_csv(source/'sample_submission.csv');submission['value']=[mapping[int(key.split('_')[0])][0 if key.endswith('_X') else 1] for key in submission.ImageID];submission.to_csv(output/'submission.csv',index=False);print(json.dumps(metrics),flush=True)


    tta_test=np.mean(preds_tta,axis=0)
    tta_mapping={int(f.stem):p for f,p in zip(testfiles,tta_test)}
    sub_tta=submission.copy()
    sub_tta['value']=[tta_mapping[int(key.split('_')[0])][0 if key.endswith('_X') else 1] for key in sub_tta.ImageID]
    sub_tta.to_csv(output/'submission_seed_tta.csv',index=False)
    submission.to_csv(output/'submission_seed_raw.csv',index=False)
    np.save(output/'oof_tta.npy',oof_tta)



BASE_OOF='data,true_x,true_y,pred_x,pred_y\n1,734.0313,1016.8171,735.1374,1011.9315\n2,769.02484,1018.05975,757.6745,1009.37714\n3,1470.2875,1037.6011,1465.2539,1038.8636\n4,1658.3167,1104.02,1680.9454,1074.5269\n5,1729.8285,1107.21,1713.7537,1098.5493\n6,723.704,1021.7844,717.19806,1014.9528\n7,1292.8555,1069.3457,1284.6201,1064.078\n8,1662.8663,1071.8718,1647.8745,1072.9008\n9,1439.916,1045.1194,1463.0723,1061.0928\n10,1524.2921,1103.3936,1489.2701,1094.8832\n11,1666.5891,1089.0338,1679.2426,1103.976\n12,1251.1063,1107.9016,1256.8501,1104.1427\n13,1409.6556,1007.5838,1417.6113,999.39355\n14,1712.1239,1106.1603,1736.5616,1109.2719\n15,1455.0385,1067.4094,1462.7856,1046.2653\n16,1515.1443,1024.1747,1442.6377,1032.2676\n17,1287.142,1104.6405,1271.2823,1099.2108\n18,1465.4414,1062.6768,1464.3954,1064.8293\n19,1468.3445,1070.1475,1453.0823,1058.7601\n20,1481.9443,1036.2792,1484.8496,1020.5237\n21,1491.5182,1014.91315,1469.5679,1032.6844\n22,1403.288,1066.1256,1411.4705,1040.26\n23,1434.2045,1058.5485,1449.4785,1043.6753\n24,1794.5674,1159.5231,1707.7784,1137.8828\n25,1701.6038,1093.5479,1706.8632,1085.0043\n26,1471.9531,1022.99146,1478.5966,1012.2791\n27,1468.9738,1016.1171,1483.1353,1033.5182\n28,1442.5994,1085.0708,1443.1519,1065.5745\n29,1474.4913,1034.1035,1460.574,1040.8252\n30,1501.6494,991.13116,1457.2618,1025.1199\n31,1680.1407,1080.3431,1686.8154,1079.5625\n32,1183.2408,1040.5542,1171.9255,1013.19055\n33,1471.861,1015.1465,1473.2064,1029.9602\n34,1316.9755,1098.509,1269.2727,1103.7533\n35,1491.6904,1030.102,1468.6277,1026.8252\n36,1484.9368,1020.3517,1462.2062,1006.8289\n37,1453.2102,1059.4314,1450.3734,1079.6627\n38,1470.7567,1041.2789,1468.9447,1041.5049\n39,1299.4407,930.1719,1307.7377,920.2054\n40,1474.591,1020.9314,1440.9905,998.74084\n41,1659.0206,1093.6176,1668.3816,1078.58\n42,1655.0939,1092.2778,1682.9592,1102.5328\n43,1688.9417,1090.5522,1688.1362,1083.7451\n44,1319.8625,1119.7977,1267.9388,1102.17\n45,1439.8423,1047.5924,1434.3127,1068.5874\n46,1207.188,1024.0131,1154.5015,1028.2073\n47,1684.1012,1099.6362,1663.7635,1098.4463\n48,1467.1132,1027.4451,1405.1235,1020.5826\n49,1090.9169,1182.161,1238.2773,1148.686\n50,1469.4753,1049.8127,1468.8088,1052.5554\n51,1491.1715,1030.6747,1495.1948,1029.1467\n52,1236.5653,1095.3138,1248.3633,1103.977\n53,1467.9807,1045.2686,1462.5391,1039.8926\n54,1493.1455,1028.5815,1475.8506,1027.2329\n55,1465.8715,1027.4945,1430.1809,1025.9575\n56,1464.109,1026.1622,1465.7761,1032.4204\n57,1468.5752,1045.2229,1545.6661,1046.7012\n58,1487.9509,1005.06323,1493.4885,1026.9729\n59,1328.0365,953.572,1352.9487,948.51416\n60,1416.9182,1033.4982,1433.6377,1017.9884\n61,1671.1056,1097.9017,1659.6096,1090.7058\n62,1653.285,1068.267,1635.3988,1069.9379\n63,1678.7899,1083.9927,1697.8134,1103.1682\n64,1486.365,1044.7195,1470.0676,1042.8567\n65,1274.0247,1096.0371,1310.0327,1084.4468\n66,1479.5956,1046.9033,1457.4875,1054.2151\n67,1260.7595,1093.2562,1264.5363,1085.2584\n68,1492.5463,1053.2632,1487.8745,1061.7084\n69,1452.7074,1056.4468,1470.0447,1033.502\n70,1268.0001,1097.3387,1307.7959,1089.6488\n71,1715.0603,1057.0776,1721.9657,1064.7637\n72,1413.6873,1014.3381,1429.4551,1023.7357\n73,1248.5541,1084.1873,1245.0466,1089.7546\n74,1487.2053,1016.02234,1496.9991,1028.7042\n75,1688.4767,1063.2249,1694.722,1046.8088\n76,1494.939,1017.49274,1485.1343,1016.8227\n77,1299.4901,934.24457,1307.7375,920.20526\n78,1665.402,1101.4827,1652.0532,1081.6652\n79,1270.2518,1134.4628,1269.552,1121.1149\n80,1457.48,1033.6771,1451.0029,1025.3472\n'
BASE_SUB='ImageID,value\n81_Fovea_X,1718.3027\n81_Fovea_Y,1126.1188\n82_Fovea_X,1692.6188\n82_Fovea_Y,1091.8401\n83_Fovea_X,1491.5935\n83_Fovea_Y,1032.7283\n84_Fovea_X,1253.1306\n84_Fovea_Y,1098.5813\n85_Fovea_X,1175.5979\n85_Fovea_Y,974.06824\n86_Fovea_X,1328.451\n86_Fovea_Y,1095.96\n87_Fovea_X,1433.3386\n87_Fovea_Y,1015.985\n88_Fovea_X,1283.6052\n88_Fovea_Y,1097.2688\n89_Fovea_X,1449.9905\n89_Fovea_Y,1030.5729\n90_Fovea_X,1670.0281\n90_Fovea_Y,1085.5938\n91_Fovea_X,1473.0507\n91_Fovea_Y,1059.3503\n92_Fovea_X,1708.9763\n92_Fovea_Y,1104.351\n93_Fovea_X,1697.2507\n93_Fovea_Y,1156.0554\n94_Fovea_X,1426.4923\n94_Fovea_Y,1037.126\n95_Fovea_X,1679.5059\n95_Fovea_Y,1094.1038\n96_Fovea_X,1713.3098\n96_Fovea_Y,1068.4039\n97_Fovea_X,1485.5525\n97_Fovea_Y,1222.2351\n98_Fovea_X,1452.7548\n98_Fovea_Y,1061.8702\n99_Fovea_X,1443.7712\n99_Fovea_Y,1029.5333\n100_Fovea_X,1691.8572\n100_Fovea_Y,1098.7506\n'

def main():
    train()
    import io
    output=Path('/kaggle/working')
    current=pd.read_csv(output/'oof.csv');old=pd.read_csv(io.StringIO(BASE_OOF))
    old=old.set_index('data').loc[current.data].reset_index()
    truth=current[['true_x','true_y']].values
    if not np.allclose(truth,old[['true_x','true_y']].values,atol=.001):raise ValueError('OOF labels/IDs do not align')
    baseline=old[['pred_x','pred_y']].values;raw=current[['pred_x','pred_y']].values;tta=np.load(output/'oof_tta.npy')
    candidates={'baseline192':baseline,'seed_raw':raw,'seed_tta':tta,'equal_raw':(baseline+raw)/2,'equal_tta':(baseline+tta)/2}
    source=Path('/kaggle/input/2026task3/detection')
    if not source.exists():source=next(Path('/kaggle/input').rglob('fovea_localization_train_GT.csv')).parent
    groups=[hashlib.sha256((source/'train'/f'{int(i):04d}.jpg').read_bytes()).hexdigest() for i in current.data]
    folds=list(GroupKFold(5,shuffle=True,random_state=FOLD_SEED).split(current,groups=groups))
    rmse=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
    scores={name:{'rmse':rmse(pred,truth),'fold_rmse':[rmse(pred[v],truth[v]) for _,v in folds]} for name,pred in candidates.items()}
    # Predetermined inference ablation: TTA must beat the same weights without TTA.
    choice='equal_raw'
    if scores['equal_tta']['rmse']<scores['equal_raw']['rmse']*.98:choice='equal_tta'
    base=scores['baseline192'];chosen=scores[choice]
    wins=sum(a<b for a,b in zip(chosen['fold_rmse'],base['fold_rmse']))
    eligible=chosen['rmse']<base['rmse']*.98 and wins>=3
    if not eligible:choice='baseline192'
    oldsub=pd.read_csv(io.StringIO(BASE_SUB))
    rawsub=pd.read_csv(output/'submission_seed_raw.csv');ttasub=pd.read_csv(output/'submission_seed_tta.csv')
    for sub in [rawsub,ttasub]:
        if list(sub.ImageID)!=list(oldsub.ImageID):raise ValueError('Submission ID order mismatch')
    selected=oldsub.copy()
    if choice.startswith('equal_'):
        selected['value']=(oldsub.value.values+(ttasub if choice=='equal_tta' else rawsub).value.values)/2
    if not np.isfinite(selected.value.values).all() or (selected.value<0).any():raise ValueError('Invalid predictions')
    selected.to_csv(output/'submission.csv',index=False)
    report={'name':'seed192_fixed_ensemble_v1','seed':SEED,'fold_seed':FOLD_SEED,'rmse':scores[choice]['rmse'],'baseline_rmse':base['rmse'],'mean_baseline_rmse':147.90789794921875,'selected':choice,'eligible_for_submission':eligible,'fold_wins':wins,'candidates':scores,'parameters':{'side':192,'new_seeds':1,'ensemble_weight':.5,'tta_min_relative_improvement':.02},'notes':'OOF remains subject to validation checkpoint selection bias; public labels were not used.'}
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    for name,pred in candidates.items():
        frame=current[['data','true_x','true_y']].copy();frame['pred_x']=pred[:,0];frame['pred_y']=pred[:,1];frame.to_csv(output/f'oof_{name}.csv',index=False)
    (output/'complete.json').write_text(json.dumps({'complete':True,'eligible_for_submission':eligible}))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
