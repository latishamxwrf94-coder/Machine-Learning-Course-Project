"""Competition-only, randomly initialized U-Net heatmap pilot on free GPU."""
import hashlib,json,os,random,time
from pathlib import Path
import cv2,numpy as np,pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold

SEED=20261020;SIDE=384;EPOCHS=int(os.environ.get('FOVEA_EPOCHS','160'))
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


"""Pure validation protocol shared by local tests and the Kaggle script."""
import numpy as np
from sklearn.model_selection import GroupKFold

def nested_groups(groups,seed):
    groups=np.asarray(groups);rows=np.arange(len(groups));result=[]
    for fold,(train,valid) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(rows,groups=groups)):
        a,b=next(GroupKFold(5,shuffle=True,random_state=seed+100+fold).split(train,groups=groups[train]))
        result.append((train,valid,train[a],train[b]))
    return result

def paired_gate(truth,base,continued,high,folds):
    def score(pred):
        return {'rmse':float(np.sqrt(np.mean((pred-truth)**2))),
                'fold_rmse':[float(np.sqrt(np.mean((pred[v]-truth[v])**2))) for v in folds]}
    scores={k:score(v) for k,v in [('base192',base),('continued192',continued),('progressive384',high)]}
    wins=lambda name:sum(a<b for a,b in zip(scores['progressive384']['fold_rmse'],scores[name]['fold_rmse']))
    allowed=all(scores['progressive384']['rmse']<scores[k]['rmse']*.95 and wins(k)>=4 for k in ['base192','continued192'])
    return {'pass':bool(allowed),'scores':scores,'fold_wins_vs_base':wins('base192'),'fold_wins_vs_continued':wins('continued192')}

"""ImageNet ResNet18 encoder with a dense localization decoder."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

class DecoderBlock(nn.Module):
    def __init__(self,a,b):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(8,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(8,b),nn.SiLU())
    def forward(self,x):return self.net(x)

class TransferHeatmapNet(nn.Module):
    def __init__(self,imagenet_state=None):
        super().__init__();net=resnet18(weights=None)
        if imagenet_state is not None:net.load_state_dict(imagenet_state,strict=True)
        self.pretrained_loaded=imagenet_state is not None
        self.encoder=nn.ModuleDict({'stem':nn.Sequential(net.conv1,net.bn1,net.relu),'pool':net.maxpool,
                                    'layer1':net.layer1,'layer2':net.layer2,'layer3':net.layer3,'layer4':net.layer4})
        self.decoder=nn.ModuleList([DecoderBlock(768,256),DecoderBlock(384,128),DecoderBlock(192,64),DecoderBlock(128,32)])
        self.head=nn.Conv2d(34,1,1)
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).reshape(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).reshape(1,3,1,1))
    def train(self,mode=True):
        super().train(mode)
        for module in self.encoder.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()
        return self
    def optimizer_groups(self,lr):
        return [{'params':list(self.encoder.parameters()),'lr':lr*.1},
                {'params':list(self.decoder.parameters())+list(self.head.parameters()),'lr':lr}]
    def forward(self,x):
        enc=self.encoder;x=(x-self.mean)/self.std;stem=enc['stem'](x);a=enc['layer1'](enc['pool'](stem));b=enc['layer2'](a);c=enc['layer3'](b);x=enc['layer4'](c)
        for block,skip in zip(self.decoder,[c,b,a,stem]):x=block(torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
        n,_,h,w=x.shape;yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device),torch.linspace(-1,1,w,device=x.device),indexing='ij')
        return self.head(torch.cat([x,xx.expand(n,1,h,w),yy.expand(n,1,h,w)],1)).flatten(1)

def event(**values):
    print(json.dumps(values),flush=True)

def fit_transfer(model,images,xy,ids,epochs,seed,device,lr=.001,schedule_epochs=160,
               validation=None,transforms=None,truth=None,directory=None,check_time=None):
    seed_all(seed);opt=torch.optim.AdamW(model.optimizer_groups(lr),weight_decay=.001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,schedule_epochs)
    scaler=torch.amp.GradScaler('cuda');best=float('inf');best_epoch=0;path=directory/'inner_best.pt' if validation is not None else None
    for epoch in range(epochs):
        if check_time:check_time()
        model.train()
        for parameter in model.encoder.parameters():parameter.requires_grad_(epoch>=5)
        order=np.random.permutation(ids)
        for offset in range(0,len(order),16):
            batch_ids=order[offset:offset+16];batch=torch.as_tensor(images[batch_ids],device=device);target=torch.as_tensor(xy[batch_ids],device=device);n=len(batch_ids)
            flip=torch.rand(n,device=device)<.5;batch[flip]=batch[flip].flip(-1);target[flip,0]=1-target[flip,0]
            delta=(torch.rand(n,2,device=device)-.5)*.08;theta=torch.eye(2,3,device=device).unsqueeze(0).repeat(n,1,1);theta[:,:,2]=-2*delta
            grid=F.affine_grid(theta,batch.shape,align_corners=False);batch=F.grid_sample(batch,grid,align_corners=False);target=target+delta
            gain=.8+.4*torch.rand(n,3,1,1,device=device);batch=(batch*gain).clamp(0,1);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                logits=model(batch);loss=-(heat_target(target,device)*F.log_softmax(logits.float(),1)).sum(1).mean()
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update()
        schedule.step()
        if validation is not None and (epoch+1)%5==0:
            pred=predict(model,images[validation],device)*transforms[validation,2:]+transforms[validation,:2]
            score=float(np.sqrt(np.mean((pred-truth[validation])**2)))
            if score<best:
                best=score;best_epoch=epoch+1;torch.save(model.state_dict(),path)
            event(stage='inner_selection',epoch=epoch+1,inner_rmse=score,best=best)
            if epoch+1-best_epoch>=40:break
        if validation is None and directory is not None and epoch+1 in SNAPSHOTS:
            torch.save(model.state_dict(),directory/f'epoch{epoch+1}.pt')
    return best_epoch
SNAPSHOTS=(20,25,30)
BOUND_MARGIN=.08
def support_bounds(targets,train):
    low=np.maximum(0,targets[train].min(0)-BOUND_MARGIN)
    high=np.minimum(1,targets[train].max(0)+BOUND_MARGIN)
    return np.concatenate([low,high]).astype(np.float32)

def outside(points,bounds):
    return ((points<bounds[:2]) | (points>bounds[2:])).any(-1)

def decode_supported(logits,bounds):
    n=SIDE//2;axis=(torch.arange(n,device=logits.device,dtype=torch.float32)+.5)/n
    yy,xx=torch.meshgrid(axis,axis,indexing='ij');b=torch.as_tensor(bounds,device=logits.device)
    allowed=(xx>=b[0])&(yy>=b[1])&(xx<=b[2])&(yy<=b[3])
    if not allowed.any():raise ValueError('Empty training-fold decoding support')
    scores=logits.float().reshape(-1,n,n);peak=scores.masked_fill(~allowed[None],-torch.inf).flatten(1).argmax(1)
    ix=peak%n;iy=peak//n;gy,gx=torch.meshgrid(torch.arange(n,device=logits.device),torch.arange(n,device=logits.device),indexing='ij')
    local=((gx[None]-ix[:,None,None])**2+(gy[None]-iy[:,None,None])**2<=16)&allowed[None]
    prob=scores.masked_fill(~local,-torch.inf).flatten(1).softmax(1).reshape(-1,n,n)
    return torch.stack([(prob*xx).sum((1,2)),(prob*yy).sum((1,2))],1)

@torch.no_grad()
def predict_supported(model,images,device,bounds):
    model.eval();raw=[];supported=[]
    for i in range(0,len(images),16):
        logits=model(torch.as_tensor(images[i:i+16],device=device))
        raw.append(decode(logits).cpu().numpy());supported.append(decode_supported(logits,bounds).cpu().numpy())
    return np.concatenate(raw),np.concatenate(supported)

def fetch_imagenet():
    import socket
    from torchvision.models import ResNet18_Weights
    socket.setdefaulttimeout(45)
    error=None
    for attempt in range(3):
        try:
            state=ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False,check_hash=True)
            path=Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            if not sha.startswith('f37072fd'):raise RuntimeError('Pretrained checksum mismatch')
            return state,{'name':'ResNet18_Weights.IMAGENET1K_V1','url':ResNet18_Weights.IMAGENET1K_V1.url,'sha256':sha}
        except Exception as exc:
            error=exc
            if attempt<2:time.sleep(5*(attempt+1))
    raise RuntimeError('Official pretrained weights unavailable; no random fallback: '+str(error))


import numpy as np

def candidate_gate(truth,old,first,second,folds,pilot_pass,confirmation_pass):
    if not all(np.isfinite(v).all() for v in [truth,old,first,second]):raise ValueError('Nonfinite OOF')
    new=(first+second)/2;blend=(old+new)/2
    def score(pred):return float(np.sqrt(np.mean((pred-truth)**2)))
    old_folds=[float(np.sqrt(np.mean((old[v]-truth[v])**2))) for v in folds]
    new_folds=[float(np.sqrt(np.mean((blend[v]-truth[v])**2))) for v in folds]
    wins=sum(a<b for a,b in zip(new_folds,old_folds))
    return {'rmse':score(blend),'baseline_rmse':score(old),'pretrained_repeat_rmse':[score(first),score(second)],'pretrained_mean_rmse':score(new),
            'old_fold_rmse':old_folds,'candidate_fold_rmse':new_folds,'fold_wins':wins,
            'eligible_for_submission':bool(pilot_pass and confirmation_pass and score(blend)<score(old)*.95 and wins>=4)}

def export_submission(candidate,previous,eligible,output):
    """Only one eligible, genuinely changed CSV may be exported for submission."""
    if list(candidate.columns)!=['ImageID','value'] or list(candidate.ImageID)!=list(previous.ImageID):raise ValueError('Submission schema or IDs mismatch')
    if not np.isfinite(candidate.value).all() or (candidate.value<0).any():raise ValueError('Invalid candidate coordinates')
    signature=lambda frame:[(str(k),format(float(v),'.12g')) for k,v in zip(frame.ImageID,frame.value)]
    changed=signature(candidate)!=signature(previous)
    allowed=bool(eligible and changed)
    # Remove only output names owned by this experiment, including legacy fallback.
    for name in ['submission.csv','submission_context_half_blend.csv','submission_context_only.csv','submission_previous_best.csv']:
        (output/name).unlink(missing_ok=True)
    if allowed:
        candidate.to_csv(output/'submission.csv',index=False)
        (output/'DO_NOT_SUBMIT.json').unlink(missing_ok=True)
    else:
        reason='validation_gate_failed' if not eligible else 'duplicate_previous_predictions'
        (output/'DO_NOT_SUBMIT.json').write_text(json.dumps({'reason':reason,'submission_generated':False,'message':'No submission CSV was generated. Keep the previous best submission.'},indent=2))
    np.savez_compressed(output/'candidate_predictions_diagnostic_only.npz',ImageID=candidate.ImageID.astype(str).values,values=candidate.value.values)
    return allowed
"""Frozen two-repeat transfer ensemble; no training and no fallback CSV."""
def main():
    global SIDE
    SIDE=384;start=time.time();torch.set_num_threads(2);cv2.setNumThreads(1);out=Path('/kaggle/working');sources={};metadata={}
    for p in Path('/kaggle/input').rglob('metrics.json'):
        try:m=json.loads(p.read_text())
        except (ValueError,OSError):continue
        if m.get('name') in ['seed192_three_seed_v1','nested_imagenet384_pilot_v1','nested_imagenet_resolution_confirmation_v1']:
            if m['name'] in sources:raise ValueError('Duplicate source artifact')
            sources[m['name']]=p.parent;metadata[m['name']]=m
    if len(sources)!=3:raise ValueError('Attach old best, ImageNet384 pilot and independent confirmation')
    pilot=metadata['nested_imagenet384_pilot_v1'];confirmation=metadata['nested_imagenet_resolution_confirmation_v1']
    source=next(Path('/kaggle/input').rglob('fovea_localization_train_GT.csv')).parent;frame=pd.read_csv(source/'fovea_localization_train_GT.csv');truth=frame[['Fovea_X','Fovea_Y']].values.astype(np.float32)
    old_folder=sources['seed192_three_seed_v1'];oldframe=pd.read_csv(old_folder/'oof_three_seed.csv').set_index('data').loc[frame.data];old=oldframe[['pred_x','pred_y']].values
    first_folder=sources['nested_imagenet384_pilot_v1'];second_folder=sources['nested_imagenet_resolution_confirmation_v1']
    first=pd.read_csv(first_folder/'oof_transfer.csv').set_index('data').loc[frame.data];second=pd.read_csv(second_folder/'oof_transfer384.csv').set_index('data').loc[frame.data]
    for f in [oldframe,first,second]:
        if not np.allclose(f[['true_x','true_y']].values,truth,atol=.001):raise ValueError('OOF labels mismatch')
    if abs(float(np.sqrt(np.mean((old-truth)**2)))-21.056412636720786)>.001:raise ValueError('Wrong old best source')
    groups=[hashlib.sha256((source/'train'/f'{int(i):04d}.jpg').read_bytes()).hexdigest() for i in frame.data]
    folds=[v for _,v in GroupKFold(5,shuffle=True,random_state=20261020).split(frame,groups=groups)]
    a=first[['pred_x','pred_y']].values;b=second[['pred_x','pred_y']].values
    gate=candidate_gate(truth,old,a,b,folds,pilot.get('pilot_pass',False),confirmation.get('pilot_pass',False))
    gate['strict_confirmation_pass']=bool(confirmation.get('pilot_pass',False));gate['eligible_for_submission']=bool(gate['rmse']<gate['baseline_rmse']*.95 and gate['fold_wins']>=4)
    report={'name':'imagenet384_half_exploratory_v1',**gate,'pilot_pass':bool(pilot.get('pilot_pass') and confirmation.get('pilot_pass')),'training_performed':False,'submission_mode':'user_authorized_exploratory','extra_oof_reproduction_skipped':True,
            'blend_weights':{'old_three_seed':.5,'pretrained_split20261025':.25,'pretrained_split20261026':.25},'source_notebooks':list(sources),'pretrained_source':pilot['pretrained_source'],
            'warning':'Old branch used outer-selected checkpoints. Nested pretrained validation and two-split confirmation improve evidence but do not guarantee Public or Private gains.'}
    if not gate['eligible_for_submission']:
        (out/'DO_NOT_SUBMIT.json').write_text(json.dumps({'reason':'Confirmation or fixed-blend validation failed','metrics':gate}));report['seconds']=time.time()-start
        (out/'metrics.json').write_text(json.dumps(report,indent=2));(out/'complete.json').write_text(json.dumps({'complete':True}));event(stage='final',**report);return
    if pilot['pretrained_source']['sha256']!=confirmation['pretrained_source']['sha256']:raise ValueError('Pretrained sources differ')
    if not torch.cuda.is_available():raise RuntimeError('Free GPU required for frozen inference')
    device='cuda';items=[load_image(source/'train'/f'{int(i):04d}.jpg') for i in frame.data];images=np.array([i[0] for i in items]);trs=np.array([i[1] for i in items]);xy=(truth-trs[:,:2])/trs[:,2:]
    testpaths=sorted((source/'test').glob('*.jpg'));testitems=[load_image(p) for p in testpaths];test=np.array([i[0] for i in testitems]);testtrs=np.array([i[1] for i in testitems]);testpred=[];reproduction=[]
    for index,(folder,fold_seed,expected) in enumerate([(first_folder,20261025,a),(second_folder,20261026,b)]):
        restored=np.zeros_like(truth)
        for fold,(train,valid,_,__) in enumerate(nested_groups(groups,fold_seed)):
            weights=folder/f'fold{fold}'/'transfer384.pt' if index==0 else folder/'side384'/f'fold{fold}'/'transfer384.pt'
            model=TransferHeatmapNet().to(device);model.load_state_dict(torch.load(weights,map_location=device,weights_only=False));bounds=support_bounds(xy,train)
            _,pred=predict_supported(model,test,device,bounds);testpred.append(pred*testtrs[:,2:]+testtrs[:,:2]);del model;torch.cuda.empty_cache()
        reproduction.append(None)
    previous=pd.read_csv(old_folder/'submission.csv');template=pd.read_csv(source/'sample_submission.csv')
    if list(previous.ImageID)!=list(template.ImageID):raise ValueError('Previous submission IDs mismatch')
    points=np.mean(testpred,axis=0);mapping={int(p.stem):v for p,v in zip(testpaths,points)};new=np.array([mapping[int(k.split('_')[0])][0 if k.endswith('_X') else 1] for k in template.ImageID])
    candidate=template.copy();candidate['value']=(previous.value.values+new)/2
    for p in testpaths:
        im=cv2.imread(str(p));vals=candidate[candidate.ImageID.str.startswith(str(int(p.stem))+'_')].value.values
        if len(vals)!=2 or not (0<=vals[0]<im.shape[1] and 0<=vals[1]<im.shape[0]):raise ValueError('Invalid image coordinates')
    allowed=export_submission(candidate,previous,True,out);report['eligible_for_submission']=allowed
    if allowed:report['submission_file_sha256']=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()
    final=(old+(a+b)/2)/2;pd.DataFrame({'data':frame.data,'true_x':truth[:,0],'true_y':truth[:,1],'pred_x':final[:,0],'pred_y':final[:,1]}).to_csv(out/'oof_fixed_blend.csv',index=False)
    report.update(oof_reproduction_max_abs=reproduction,changed_rows=int((np.abs(candidate.value.values-previous.value.values)>1e-9).sum()),seconds=time.time()-start,gpu_hours=(time.time()-start)/3600)
    (out/'metrics.json').write_text(json.dumps(report,indent=2));(out/'complete.json').write_text(json.dumps({'complete':True,'eligible_for_submission':allowed}));event(stage='final',**report)

if __name__=='__main__':main()
