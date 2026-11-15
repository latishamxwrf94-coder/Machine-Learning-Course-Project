"""Competition-only, randomly initialized U-Net heatmap pilot on free GPU."""
import hashlib,json,os,random,time
from pathlib import Path
import cv2,numpy as np,pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold

SEED=20261022;FOLD_SEED=20261020;SIDE=192;EPOCHS=int(os.environ.get('FOVEA_EPOCHS','160'))
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

def train():
    seed_all(SEED);torch.set_num_threads(2);cv2.setNumThreads(1);start=time.time()
    source=Path('/kaggle/input/2026task3/detection')
    if not source.exists():source=next(Path('/kaggle/input').rglob('fovea_localization_train_GT.csv')).parent
    output=Path('/kaggle/working');device='cuda' if torch.cuda.is_available() else 'cpu'
    if device!='cuda':raise RuntimeError('GPU unavailable: refuse expensive CPU fallback')
    df=pd.read_csv(source/'fovea_localization_train_GT.csv');truth=df[['Fovea_X','Fovea_Y']].values.astype(np.float32);images=[];transforms=[];groups=[]
    for ident in df.data:
        f=source/'train'/f'{int(ident):04d}.jpg';im,tr=load_image(f);images.append(im);transforms.append(tr);groups.append(hashlib.sha256(f.read_bytes()).hexdigest())
    images=np.array(images);transforms=np.array(transforms);xy=(truth-transforms[:,:2])/transforms[:,2:];oof=np.zeros_like(truth);folds=list(GroupKFold(5,shuffle=True,random_state=FOLD_SEED).split(df,groups=groups));epochs=[];scores=[];baseline=np.zeros_like(truth)
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
        model.load_state_dict(torch.load(checkpoint,weights_only=False)['model']);oof[valid]=predict(model,images[valid],device)*transforms[valid,2:]+transforms[valid,:2];scores.append(best);epochs.append(best_epoch);baseline[valid]=truth[train].mean(0);np.save(output/'oof_partial.npy',oof)
    score=float(np.sqrt(np.mean((oof-truth)**2)));metrics={'name':'unet_scratch_v1','rmse':score,'fold_rmse':scores,'best_epochs':epochs,'mean_baseline_rmse':float(np.sqrt(np.mean((baseline-truth)**2))),'seed':SEED,'side':SIDE,'max_epochs':EPOCHS,'seconds':time.time()-start,'gpu_hours':(time.time()-start)/3600,'hypothesis':'Train-only spatial heatmap learning can resolve difficult cases using contextual vessel patterns; grouped folds prevent duplicate leakage.'};(output/'metrics.json').write_text(json.dumps(metrics,indent=2));pd.DataFrame({'data':df.data,'true_x':truth[:,0],'true_y':truth[:,1],'pred_x':oof[:,0],'pred_y':oof[:,1]}).to_csv(output/'oof.csv',index=False)
    testfiles=sorted((source/'test').glob('*.jpg'));testitems=[load_image(f) for f in testfiles];test=np.array([t[0] for t in testitems]);trs=np.array([t[1] for t in testitems]);preds=[]
    for fold in range(5):
        model=HeatmapNet().to(device);model.load_state_dict(torch.load(output/f'fold{fold}.pt',weights_only=False)['model']);preds.append(predict(model,test,device)*trs[:,2:]+trs[:,:2])
    preds=np.mean(preds,axis=0);mapping={int(f.stem):p for f,p in zip(testfiles,preds)};submission=pd.read_csv(source/'sample_submission.csv');submission['value']=[mapping[int(key.split('_')[0])][0 if key.endswith('_X') else 1] for key in submission.ImageID];submission.to_csv(output/'submission.csv',index=False);print(json.dumps(metrics),flush=True)




def previous_best():
    # Add the saved seed-192 notebook output as a Kaggle input source.
    matches=[]
    for path in Path('/kaggle/input').rglob('metrics.json'):
        try:
            meta=json.loads(path.read_text())
            if meta.get('name')=='seed192_fixed_ensemble_v1' and meta.get('selected')=='equal_raw' and abs(meta.get('rmse',0)-21.621650353244803)<.001:
                matches.append((path.parent,meta))
        except (ValueError,OSError,TypeError):continue
    if len(matches)!=1:
        raise RuntimeError('Add the saved output of zijin233/seed-192 (score 13.37311) as an input. Required: metrics.json, oof_equal_raw.csv, submission.csv. No training has started.')
    folder,meta=matches[0]
    oof=pd.read_csv(folder/'oof_equal_raw.csv');sub=pd.read_csv(folder/'submission.csv')
    if list(sub.columns)!=['ImageID','value'] or len(sub)!=40 or sub.ImageID.duplicated().any():raise ValueError('Previous submission format invalid')
    if not np.isfinite(sub.value.values).all():raise ValueError('Previous submission contains invalid values')
    if len(oof)!=80 or oof.data.duplicated().any():raise ValueError('Previous OOF IDs invalid')
    return folder,meta,oof,sub



def main():
    folder,prior,previous_oof,previous_sub=previous_best()
    # Validate input labels and OOF consistency before allocating training GPU.
    source=Path('/kaggle/input/2026task3/detection')
    if not source.exists():source=next(Path('/kaggle/input').rglob('fovea_localization_train_GT.csv')).parent
    truth_frame=pd.read_csv(source/'fovea_localization_train_GT.csv')
    check=previous_oof.set_index('data').loc[truth_frame.data]
    if not np.allclose(check[['true_x','true_y']].values,truth_frame[['Fovea_X','Fovea_Y']].values,atol=.001):raise ValueError('Previous OOF labels do not match current data')
    train()
    output=Path('/kaggle/working')
    current=pd.read_csv(output/'oof.csv');current_sub=pd.read_csv(output/'submission.csv')
    previous_oof=previous_oof.set_index('data').loc[current.data].reset_index()
    truth=current[['true_x','true_y']].values
    old=previous_oof[['pred_x','pred_y']].values;new=current[['pred_x','pred_y']].values
    if list(current_sub.ImageID)!=list(previous_sub.ImageID):raise ValueError('Submission ID mismatch')
    if not np.isfinite(current_sub.value.values).all():raise ValueError('New submission contains nonfinite values')
    groups=[hashlib.sha256((source/'train'/f'{int(i):04d}.jpg').read_bytes()).hexdigest() for i in current.data]
    folds=list(GroupKFold(5,shuffle=True,random_state=FOLD_SEED).split(current,groups=groups))
    rmse=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
    # Previous best is the average of two seeds; add one seed with fixed 1/3 weight.
    blended=(2*old+new)/3
    candidates={}
    for name,pred in [('previous_best',old),('new_seed_only',new),('three_seed_equal',blended)]:
        candidates[name]={'rmse':rmse(pred,truth),'fold_rmse':[rmse(pred[v],truth[v]) for _,v in folds]}
    baseline=candidates['previous_best'];candidate=candidates['three_seed_equal']
    if abs(baseline['rmse']-prior['rmse'])>.001:raise ValueError('Previous OOF metric mismatch')
    wins=sum(a<b for a,b in zip(candidate['fold_rmse'],baseline['fold_rmse']))
    eligible=candidate['rmse']<baseline['rmse']*.98 and wins>=3
    selected='three_seed_equal' if eligible else 'previous_best'
    previous_sub.to_csv(output/'submission_previous_best.csv',index=False)
    current_sub.to_csv(output/'submission_new_seed.csv',index=False)
    blend_sub=current_sub.copy();blend_sub['value']=(2*previous_sub.value.values+current_sub.value.values)/3
    blend_sub.to_csv(output/'submission_three_seed.csv',index=False)
    final=blend_sub if eligible else previous_sub
    if not np.isfinite(final.value.values).all() or (final.value<0).any():raise ValueError('Invalid final submission')
    final.to_csv(output/'submission.csv',index=False)
    for name,pred in [('previous_best',old),('new_seed',new),('three_seed',blended)]:
        frame=current[['data','true_x','true_y']].copy();frame['pred_x']=pred[:,0];frame['pred_y']=pred[:,1];frame.to_csv(output/f'oof_{name}.csv',index=False)
    report={'name':'seed192_three_seed_v1','seed':SEED,'fold_seed':FOLD_SEED,'side':SIDE,'rmse':candidates[selected]['rmse'],'baseline_rmse':baseline['rmse'],'baseline_public_score':13.37311,'selected':selected,'eligible_for_submission':eligible,'fold_wins':wins,'candidates':candidates,'weights':[1/3,1/3,1/3],'inference_tta':False,'warning':'Validation uses best checkpoints selected on the same folds; OOF is not an unbiased final estimate. Do not resubmit if eligible_for_submission is false.'}
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    (output/'complete.json').write_text(json.dumps({'complete':True,'eligible_for_submission':eligible}))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
