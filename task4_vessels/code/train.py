"""Original competition-only vessel segmentation; free GPU; no external weights.
Runs paired patch U-Net pilots then refits validation-selected configuration.
"""
import os,time,json,random,hashlib,csv
from pathlib import Path
import cv2,numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

SEED=20261031;EPOCHS=40;STEPS=12;BATCH=8;PATCH=256
OUT=Path('/kaggle/working') if Path('/kaggle').exists() else Path('local_outputs')
OUT.mkdir(exist_ok=True);START=time.time();CAP=float('inf')
def event(**kw):print(json.dumps(kw),flush=True)
def seed_all():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
class Block(nn.Module):
    def __init__(self,a,b):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU())
    def forward(self,x):return self.net(x)
class UNet(nn.Module):
    def __init__(self):
        super().__init__();self.enc=nn.ModuleList([Block(4,16),Block(16,32),Block(32,64),Block(64,128)]);self.dec=nn.ModuleList([Block(192,64),Block(96,32),Block(48,16)]);self.head=nn.Conv2d(16,1,1)
    def forward(self,x):
        skips=[]
        for i,b in enumerate(self.enc):
            x=b(x)
            if i<3:skips.append(x);x=F.max_pool2d(x,2)
        for b,s in zip(self.dec,reversed(skips)):x=b(torch.cat([F.interpolate(x,size=s.shape[-2:],mode='bilinear',align_corners=False),s],1))
        return self.head(x)
def preprocess(a):
    g=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(a[:,:,1])
    return np.concatenate([a,g[:,:,None]],2).transpose(2,0,1).astype(np.float32)/127.5-1
def load_data():
    candidates=list(Path('/kaggle/input').rglob('sample_submission.csv')) if Path('/kaggle/input').exists() else [Path('data/segmentation/sample_submission.csv')]
    roots=[p.parent for p in candidates if (p.parent/'train/image').exists()]
    assert len(roots)==1,roots
    root=roots[0];data=[]
    for i in range(1,21):
        f=root/'train/image'/f'{i}.jpg';lf=list((root/'train/label').glob(f'{i}.*'));assert len(lf)==1
        a=np.array(Image.open(f).convert('RGB'));y=np.array(Image.open(lf[0]).convert('L'))
        assert set(np.unique(y))<={0,255} and y.shape==a.shape[:2]
        # Supplied converter: bilinear resize then >0, row-major nonzero foreground.
        truth=np.array(Image.fromarray(y).resize((512,512),Image.Resampling.BILINEAR))>0
        data.append((preprocess(a),(y>0).astype(np.float32),truth))
    return root,data
@torch.no_grad()
def infer(model,x):
    model.eval();h,w=x.shape[-2:];t=torch.from_numpy(x[None]).cuda();t=F.pad(t,(0,(-w)%8,0,(-h)%8),mode='reflect')
    with torch.autocast('cuda',dtype=torch.float16):p=model(t).sigmoid()[0,0,:h,:w].float().cpu().numpy()
    return cv2.resize(p,(512,512),interpolation=cv2.INTER_LINEAR)
def evaluate(model,data,ids):
    probs=[infer(model,data[i][0]) for i in ids];scores=[]
    for threshold in np.arange(.2,.81,.05):
        ds=[float((2*((p>threshold)&data[i][2]).sum()+1e-6)/((p>threshold).sum()+data[i][2].sum()+1e-6)) for p,i in zip(probs,ids)]
        scores.append({'dice':float(np.mean(ds)),'threshold':float(threshold),'per_image_dice':ds})
    return max(scores,key=lambda x:x['dice'])
def batch(data,ids,augment,rng):
    xs=[];ys=[]
    for _ in range(BATCH):
        i=int(rng.choice(ids));x,y,_=data[i];h,w=y.shape;top=int(rng.integers(h-PATCH+1));left=int(rng.integers(w-PATCH+1));x=x[:,top:top+PATCH,left:left+PATCH];y=y[top:top+PATCH,left:left+PATCH]
        if augment:
            k=int(rng.integers(4));x=np.rot90(x,k,axes=(1,2));y=np.rot90(y,k)
            if rng.random()<.5:x=x[:,:,::-1];y=y[:,::-1]
        xs.append(x.copy());ys.append(y[None].copy())
    return torch.from_numpy(np.stack(xs)).cuda(),torch.from_numpy(np.stack(ys)).cuda()
def fit(data,ids,augment,name,epochs,valid=None):
    seed_all();rng=np.random.default_rng(SEED);model=UNet().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000);best={'dice':-1};history=[]
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for step in range(STEPS):
            if time.time()-START>CAP:raise RuntimeError('Bounded GPU runtime exhausted; checkpoint retained')
            x,y=batch(data,ids,augment,rng);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(x);p=logits.sigmoid();dice=1-(2*(p*y).sum((1,2,3))+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1);loss=F.binary_cross_entropy_with_logits(logits,y)+dice.mean()
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
        # Fixed common cosine schedule; identical budget across controlled experiments.
        lr=.001*(.1+.9*(1+np.cos(np.pi*epoch/epochs))/2)
        for group in opt.param_groups:group['lr']=lr
        record={'experiment':name,'epoch':epoch,'loss':float(np.mean(losses)),'seconds':time.time()-START}
        if valid is not None and (epoch%5==0 or epoch==epochs):
            record.update(evaluate(model,data,valid))
            if record['dice']>best['dice']:
                best={**record,'augment':augment};torch.save({'model':model.state_dict(),'configuration':best},OUT/(name+'_best.pt'))
        if epoch%5==0 or epoch==epochs:
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'augment':augment,'seed':SEED,'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},OUT/(name+'_last.pt'))
            event(**record)
        history.append(record);(OUT/(name+'_history.json')).write_text(json.dumps(history,indent=2))
    return model,best
def encode_mask(mask):
    a=mask.astype(np.int8).ravel();r=np.flatnonzero(np.diff(np.r_[0,a,0]))+1;r[1::2]-=r[::2];return ' '.join(map(str,r))

LegacyUNet=UNet
def require_checkpoint_identity(checkpoint,expected):
    if checkpoint.get('identity')!=expected:
        raise ValueError('Checkpoint data/source/config identity mismatch; do not resume')

"""Macro-image Dice, with threshold chosen on other OOF folds only.
This removes direct threshold fitting on each scored fold, but is not nested
model-selection CV because other OOF models can have trained on that fold.
"""
import numpy as np
THRESHOLDS=np.array([.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8])
def crossfit_summary(probabilities,truth,folds):
    p=np.asarray(probabilities);y=np.asarray(truth,dtype=bool)
    if p.shape!=y.shape or not np.isfinite(p).all():raise ValueError('Invalid probabilities/truth')
    if sorted(i for fold in folds for i in fold)!=list(range(len(y))):raise ValueError('Fold coverage must be exact')
    ds=np.array([[(2*((q>t)&z).sum()+1e-6)/((q>t).sum()+z.sum()+1e-6) for q,z in zip(p,y)] for t in THRESHOLDS])
    best=int(ds.mean(1).argmax());scores=[];thresholds=[]
    for valid in folds:
        fit=[i for i in range(len(y)) if i not in valid]
        k=int(ds[:,fit].mean(1).argmax());scores.append(float(ds[k,valid].mean()));thresholds.append(float(THRESHOLDS[k]))
    return {'oof_dice':float(ds[best].mean()),'threshold':float(THRESHOLDS[best]),'image_dice':ds[best].tolist(),'fold_dice':[float(ds[best,v].mean()) for v in folds],'crossfit_dice':float(np.mean(scores)),'crossfit_fold_dice':scores,'crossfit_thresholds':thresholds}


# Controlled follow-up: train image/label geometry aligned with official 512px conversion.
# Definitions above are the exact deployed baseline architecture and optimizer family.
def fit(data,ids,augment,name,epochs,valid=None):
    seed_all();rng=np.random.default_rng(SEED);aug_rng=np.random.default_rng(SEED+999)
    model=UNet().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000)
    history=[];first=1;last=OUT/(name+'_last.pt');resume=None
    # A future recovery notebook attaches the previous private notebook as a data source.
    resume_roots=[Path(os.environ['RESUME_DIR'])] if os.environ.get('RESUME_DIR') else list(Path('/kaggle/input').glob('**/'+name+'_last.pt'))
    for location in resume_roots:
        checkpoint=location/(name+'_last.pt') if location.is_dir() else location
        if checkpoint.exists():resume=checkpoint;break
    identity={'source':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'data':CURRENT_DATA_FINGERPRINT,'patch':PATCH,'steps':STEPS,'batch':BATCH,'augmentation':augment,'architecture':'fr2016','supervision':VARIANT}
    if resume is not None:
        ck=torch.load(resume,map_location='cuda',weights_only=False)
        require_checkpoint_identity(ck,identity)
        assert ck['seed']==SEED and ck['epochs']==epochs and ck['ids']==ids and ck['name']==name
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);rng.bit_generator.state=ck['rng'];aug_rng.bit_generator.state=ck['aug_rng'];torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']]);first=ck['epoch']+1;history=ck['history'];event(stage='resumed',name=name,epoch=first)
    for epoch in range(first,epochs+1):
        model.train();losses=[]
        for _ in range(STEPS):
            if time.time()-START>CAP:raise RuntimeError('Bounded GPU runtime exhausted; resume from last complete epoch')
            xs=[];ys=[]
            for _ in range(BATCH):
                i=int(rng.choice(ids));x,y,_=data[i];h,w=y.shape;top=int(rng.integers(h-PATCH+1));left=int(rng.integers(w-PATCH+1));x=x[:,top:top+PATCH,left:left+PATCH];y=y[top:top+PATCH,left:left+PATCH]
                if augment:
                    k=int(aug_rng.integers(4));x=np.rot90(x,k,(1,2));y=np.rot90(y,k)
                    if aug_rng.random()<.5:x=x[:,:,::-1];y=y[:,::-1]
                xs.append(x.copy());ys.append(y[None].copy())
            x=torch.from_numpy(np.stack(xs)).cuda();y=torch.from_numpy(np.stack(ys)).cuda();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                loss=candidate_loss(model,x,y,VARIANT)
            scaler.scale(loss).backward();scaler.unscale_(opt);finite_gradients(loss,model);scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
        for group in opt.param_groups:group['lr']=.001*(.1+.9*(1+np.cos(np.pi*epoch/epochs))/2)
        history.append({'epoch':epoch,'loss':float(np.mean(losses))});ck={'identity':identity,'name':name,'seed':SEED,'epochs':epochs,'ids':ids,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'rng':rng.bit_generator.state,'aug_rng':aug_rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'history':history}
        if epoch%5==0 or epoch==epochs:
            part=last.with_suffix('.tmp');torch.save(ck,part);part.replace(last);event(experiment=name,epoch=epoch,loss=history[-1]['loss'],seconds=time.time()-START)
    (OUT/(name+'_history.json')).write_text(json.dumps(history,indent=2))
    return model,{}

def aligned_data(root):
    data=[]
    for i in range(1,21):
        a=np.array(Image.open(root/'train/image'/f'{i}.jpg').convert('RGB').resize((512,512),Image.Resampling.BILINEAR))
        label=list((root/'train/label').glob(f'{i}.*'))[0];y=np.array(Image.open(label).convert('L').resize((512,512),Image.Resampling.BILINEAR))>0
        data.append((preprocess(a),y.astype(np.float32),y))
    return data

def summarize(probs,truth,folds):
    candidates=[]
    for t in [.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8]:
        ds=[float((2*((p>t)&y).sum()+1e-6)/((p>t).sum()+y.sum()+1e-6)) for p,y in zip(probs,truth)]
        candidates.append({'threshold':t,'oof_dice':float(np.mean(ds)),'image_dice':ds,'fold_dice':[float(np.mean([ds[i] for i in v])) for v in folds]})
    return max(candidates,key=lambda x:x['oof_dice'])



# Source https://github.com/lseventeen/FR-UNet commit 1856bc206055fdfccde184d0ebf9ac8c3207387d
# Adaptation: self-contained initializer import;4inputchannels;feature_scale4 (base16).
# MIT License
# 
# Copyright (c) 2022 liuwentao
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from torch import nn
from torch.nn.init import trunc_normal_

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.neg_slope)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
import torch
import torch.nn as nn



class conv(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(conv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True))
        

    def forward(self, x):
        return self.conv(x)


class feature_fuse(nn.Module):
    def __init__(self, in_c, out_c):
        super(feature_fuse, self).__init__()
        self.conv11 = nn.Conv2d(
            in_c, out_c, kernel_size=1, padding=0, bias=False)
        self.conv33 = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.conv33_di = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=2, bias=False, dilation=2)
        self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv33(x)
        x3 = self.conv33_di(x)
        out = self.norm(x1+x2+x3)
        return out


class up(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2,
                               padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=False))

    def forward(self, x):
        x = self.up(x)
        return x


class down(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2,
                      padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self.down(x)
        return x


class block(nn.Module):
    def __init__(self, in_c, out_c,  dp=0, is_up=False, is_down=False, fuse=False):
        super(block, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        if fuse == True:
            self.fuse = feature_fuse(in_c, out_c)
        else:
            self.fuse = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1)

        self.is_up = is_up
        self.is_down = is_down
        self.conv = conv(out_c, out_c, dp=dp)
        if self.is_up == True:
            self.up = up(out_c, out_c//2)
        if self.is_down == True:
            self.down = down(out_c, out_c*2)

    def forward(self,  x):
        if self.in_c != self.out_c:
            x = self.fuse(x)
        x = self.conv(x)
        if self.is_up == False and self.is_down == False:
            return x
        elif self.is_up == True and self.is_down == False:
            x_up = self.up(x)
            return x, x_up
        elif self.is_up == False and self.is_down == True:
            x_down = self.down(x)
            return x, x_down
        else:
            x_up = self.up(x)
            x_down = self.down(x)
            return x, x_up, x_down


class FR_UNet(nn.Module):
    def __init__(self,  num_classes=1, num_channels=1, feature_scale=2,  dropout=0.2, fuse=True, out_ave=True):
        super(FR_UNet, self).__init__()
        self.out_ave = out_ave
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.block1_3 = block(
            num_channels, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block1_2 = block(
            filters[0], filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block1_1 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block10 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block11 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block12 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block13 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block2_2 = block(
            filters[1], filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block2_1 = block(
            filters[1]*2, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block20 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block21 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block22 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block3_1 = block(
            filters[2], filters[2],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block30 = block(
            filters[2]*2, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block31 = block(
            filters[2]*3, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block40 = block(filters[3], filters[3],
                             dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.final1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final2 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final3 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final4 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final5 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.fuse = nn.Conv2d(
            5, num_classes, kernel_size=1, padding=0, bias=True)
        self.apply(InitWeights_He)

    def forward(self, x):
        x1_3, x_down1_3 = self.block1_3(x)
        x1_2, x_down1_2 = self.block1_2(x1_3)
        x2_2, x_up2_2, x_down2_2 = self.block2_2(x_down1_3)
        x1_1, x_down1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))
        x2_1, x_up2_1, x_down2_1 = self.block2_1(
            torch.cat([x_down1_2, x2_2], dim=1))
        x3_1, x_up3_1, x_down3_1 = self.block3_1(x_down2_2)
        x10, x_down10 = self.block10(torch.cat([x1_1, x_up2_1], dim=1))
        x20, x_up20, x_down20 = self.block20(
            torch.cat([x_down1_1, x2_1, x_up3_1], dim=1))
        x30, x_up30 = self.block30(torch.cat([x_down2_1, x3_1], dim=1))
        _, x_up40 = self.block40(x_down3_1)
        x11, x_down11 = self.block11(torch.cat([x10, x_up20], dim=1))
        x21, x_up21 = self.block21(torch.cat([x_down10, x20, x_up30], dim=1))
        _, x_up31 = self.block31(torch.cat([x_down20, x30, x_up40], dim=1))
        x12 = self.block12(torch.cat([x11, x_up21], dim=1))
        _, x_up22 = self.block22(torch.cat([x_down11, x21, x_up31], dim=1))
        x13 = self.block13(torch.cat([x12, x_up22], dim=1))
        if self.out_ave == True:
            output = (self.final1(x1_1)+self.final2(x10) +
                      self.final3(x11)+self.final4(x12)+self.final5(x13))/5
        else:
            output = self.final5(x13)

        return output

class UNet(FR_UNet):
    def __init__(self):super().__init__(num_classes=1,num_channels=4,feature_scale=4,dropout=.2)


FRNet=UNet
"""Two isolated supervision changes; deployed FR inference is unchanged."""
import math
import torch
from torch.nn import functional as F

def segmentation_loss(logits,y):
    # FP32 reductions avoid overflow under autocast at batch8/256px.
    z=logits.float();p=z.sigmoid();y=y.float()
    dice=1-(2*(p*y).sum((1,2,3))+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1)
    return F.binary_cross_entropy_with_logits(z,y)+dice.mean()

def forward_heads(model,x):
    heads=[];handles=[]
    try:
        for i in range(1,6):
            handles.append(getattr(model,'final'+str(i)).register_forward_hook(lambda module,args,out:heads.append(out)))
        output=model(x)
        if len(heads)!=5:raise RuntimeError('FR must execute exactly five output heads')
        return heads
    finally:
        for handle in handles:handle.remove()

def deep_supervision_loss(heads,y):
    if len(heads)!=5:raise ValueError('Expected five FR heads')
    return .5*segmentation_loss(sum(heads)/5,y)+.5*sum(segmentation_loss(z,y) for z in heads)/5

def hard_pixel_penalty(logits,y,fraction=.1):
    if not 0<fraction<=1:raise ValueError('Invalid hard pixel fraction')
    errors=F.binary_cross_entropy_with_logits(logits.float(),y.float(),reduction='none').flatten(1)
    k=max(1,int(math.ceil(errors.shape[1]*fraction)))
    return errors.topk(k,dim=1,sorted=False).values.mean()

def candidate_loss(model,x,y,variant):
    if variant=='deep_supervision':return deep_supervision_loss(forward_heads(model,x),y)
    logits=model(x);loss=segmentation_loss(logits,y)
    if variant=='hard_pixels':return loss+.25*hard_pixel_penalty(logits,y)
    if variant!='plain':raise ValueError('Unknown variant '+variant)
    return loss

def pilot_pass(candidate_plain,paired_plain,candidate_blend,paired_blend,historical_blend):
    return bool(candidate_plain>=paired_plain-.002 and candidate_blend>=max(paired_blend,historical_blend)+.001)

def finite_gradients(loss,model):
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not bool(torch.isfinite(loss).all()) or not all(bool(torch.isfinite(g).all()) for g in grads):
        raise RuntimeError('Nonfinite gradient/loss: stop before optimizer step')

"""Prespecified development -> sealed confirmation -> conditional fivefold."""
import traceback
VARIANT='plain'
EXPECTED={'fr':'97a8122c79359cf886284228a0f45a3ff6bfc426adf5f63ff714f70ae0430261','aligned':'6274b8ce6054dd790340d660b2045418bb76f139fe00d877c7c3fe070e33404f','photo':'e64fc8ca895e36b21717999382f6679d3cfc3671dd8efb61b67556881ffccaef'}

def save_json(filename,obj):
    p=OUT/filename;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)

def binary_metrics(probabilities,truth,threshold,ids):
    p=np.asarray(probabilities);y=np.asarray(truth,dtype=bool)
    assert p.shape==y.shape and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    q=p>threshold;tp=(q&y).sum((1,2));fp=(q&~y).sum((1,2));fn=(~q&y).sum((1,2));eps=1e-6
    d=(2*tp+eps)/(2*tp+fp+fn+eps);pr=(tp+eps)/(tp+fp+eps);re=(tp+eps)/(tp+fn+eps)
    return {'dice':float(d.mean()),'precision':float(pr.mean()),'recall':float(re.mean()),'threshold':float(threshold),'per_image':[{'id':int(i+1),'dice':float(a),'precision':float(b),'recall':float(c),'fp':int(f),'fn':int(n)} for i,a,b,c,f,n in zip(ids,d,pr,re,fp,fn)]}

def references(truth,fp):
    refs={}
    for path in Path('/kaggle/input').rglob('metrics.json'):
        m=json.loads(path.read_text())
        for name,sha in EXPECTED.items():
            if m.get('source_sha256')==sha:
                assert name not in refs and m['data_fingerprint']==fp
                refs[name]=path.parent
    assert set(refs)==set(EXPECTED),list(refs)
    files=[(refs['aligned'],'native_control_oof.npz'),(refs['aligned'],'aligned512_oof.npz'),(refs['photo'],'photo_oof.npz'),(refs['fr'],'research_oof.npz')]
    arrays=[np.load(p/f) for p,f in files]
    assert all(np.array_equal(a['truth'],truth) for a in arrays)
    return refs,(arrays[0]['probabilities']+arrays[1]['probabilities']+arrays[2]['probabilities'])/3,arrays[3]['probabilities']

def train_fold(variant,fold,data,truth,folds):
    global VARIANT
    VARIANT=variant;name=variant+'_fold'+str(fold);valid=folds[fold];begin=time.time()
    identity={'source':SOURCE_HASH,'data':CURRENT_DATA_FINGERPRINT,'variant':variant,'fold':fold,'valid_indices':valid}
    cache=OUT/(name+'_predictions.npz');manifest=OUT/(name+'_fit.json')
    if cache.exists() and manifest.exists():
        saved=json.loads(manifest.read_text());assert saved['identity']==identity
        assert saved['prediction_sha256']==hashlib.sha256(cache.read_bytes()).hexdigest()
        z=np.load(cache);assert np.array_equal(z['truth'],truth[valid])
        return z['probabilities'],saved
    torch.cuda.reset_peak_memory_stats()
    model,_=fit(data,[i for i in range(20) if i not in valid],False,name,168)
    training_seconds=time.time()-begin;train_peak=torch.cuda.max_memory_allocated()
    p=np.stack([infer(model,data[i][0]) for i in valid]);torch.cuda.synchronize()
    forward_begin=time.perf_counter()
    for i in valid:infer(model,data[i][0])
    torch.cuda.synchronize();inference_ms=(time.perf_counter()-forward_begin)*1000/len(valid)
    torch.save({'model':model.state_dict(),'data_fingerprint':CURRENT_DATA_FINGERPRINT,'source_sha256':SOURCE_HASH,'variant':variant,'train_ids':[i+1 for i in range(20) if i not in valid]},OUT/(name+'_model.pt'))
    saved={'identity':identity,'training_seconds':training_seconds,'train_peak_allocated_bytes':train_peak,'inference_with_transfer_ms_per_image':inference_ms,'params':sum(t.numel() for t in model.parameters()),'fixed_05':binary_metrics(p,truth[valid],.5,valid)}
    np.savez_compressed(cache,probabilities=p,truth=truth[valid]);saved['prediction_sha256']=hashlib.sha256(cache.read_bytes()).hexdigest();save_json(name+'_fit.json',saved)
    del model;torch.cuda.empty_cache();event(stage='fold_complete',variant=variant,fold=fold,image_ids=[i+1 for i in valid],dice=saved['fixed_05']['dice'],seconds=time.time()-START)
    return p,saved

def trial_row(variant,p,paired,triple,old_best,truth,valid):
    standalone=binary_metrics(p,truth[valid],.5,valid)
    control=binary_metrics(paired,truth[valid],.5,valid)
    blended=binary_metrics((p+triple[valid])/2,truth[valid],.5,valid)
    paired_blend=binary_metrics((paired+triple[valid])/2,truth[valid],.5,valid)
    historical=binary_metrics(old_best[valid],truth[valid],.5,valid)
    return {'variant':variant,'standalone':standalone,'paired_fr':control,'fixed_blend':blended,'paired_fr_blend':paired_blend,'historical_best':historical,'passed':pilot_pass(standalone['dice'],control['dice'],blended['dice'],paired_blend['dice'],historical['dice'])}

"""Training-only input selection and fail-closed numerical recovery guard."""
def training_preflight_arrays(data,valid_indices,seed,batch_size=8,patch=256):
    import numpy as np
    ids=[i for i in range(len(data)) if i not in valid_indices]
    if not ids:raise ValueError('No training images available for preflight')
    rng=np.random.default_rng(seed);xs=[];ys=[];chosen=[]
    for _ in range(batch_size):
        i=int(rng.choice(ids));x,y,_=data[i];h,w=y.shape
        if min(h,w)<patch:raise ValueError('Training image too small')
        top=int(rng.integers(h-patch+1));left=int(rng.integers(w-patch+1))
        xs.append(x[:,top:top+patch,left:left+patch].copy());ys.append(y[None,top:top+patch,left:left+patch].copy());chosen.append(i)
    return np.stack(xs),np.stack(ys),chosen

def require_real_diagnostic(metrics,source_sha256):
    if not metrics.get('complete') or metrics.get('optimizer_steps')!=0 or metrics.get('source_sha256')!=source_sha256:
        raise ValueError('Missing or mismatched diagnostic identity')
    rows=[r for r in metrics.get('results',[]) if r.get('input')=='real']
    expected={(v,a) for v in ['plain','deep_supervision','hard_pixels'] for a in [True,False]}
    if len(rows)!=6 or {(r['variant'],r['amp']) for r in rows}!=expected:
        raise ValueError('Incomplete real-crop diagnostic controls')
    if not all(r['loss_finite'] and r['nonfinite_parameter_gradients']==0 for r in rows):
        raise ValueError('Real crop numerical diagnostic failed; do not launch formal training')

def gpu_preflight(data,valid_indices):
    tests={}
    for variant in ['plain','deep_supervision','hard_pixels']:
        seed_all();net=FRNet().cuda();scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000)
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);records=[]
        for step in range(3):
            a,b,ids=training_preflight_arrays(data,valid_indices,SEED+step,batch_size=BATCH,patch=PATCH)
            assert set(ids).isdisjoint(valid_indices) and float(a.std())>0
            x=torch.from_numpy(a).cuda();y=torch.from_numpy(b).cuda();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=candidate_loss(net,x,y,variant)
            scaler.scale(loss).backward();scaler.unscale_(opt);finite_gradients(loss,net);scaler.step(opt);scaler.update()
            records.append({'step':step,'loss':float(loss.detach()),'image_ids':[i+1 for i in ids],'finite':True})
        tests[variant]=records;del net,opt;torch.cuda.empty_cache()
    save_json('gpu-preflight.json',{'passed':True,'input':'real_training_crops','steps_per_variant':3,'tests':tests,'gpu':torch.cuda.get_device_name(),'source_sha256':SOURCE_HASH})

def refit_predict(variant,selected,root,data,refs):
    global VARIANT
    VARIANT=variant;net,_=fit(data,list(range(20)),False,variant+'_full',168)
    ck={'model':net.state_dict(),'configuration':selected,'seed':SEED,'data_fingerprint':CURRENT_DATA_FINGERPRINT,'source_sha256':SOURCE_HASH}
    torch.save(ck,OUT/'best_model.pt');del net;torch.cuda.empty_cache()
    components=[{'weights':'best_model.pt','geometry':'aligned512','weight':.5,'model_class':'FRNet'}]
    old=json.loads((refs['fr']/'ensemble.json').read_text())
    # Reuse only the old triple components, not the old FR component.
    for entry in old['components'][1:]:
        ck=torch.load(refs['fr']/entry['weights'],map_location='cpu',weights_only=False)
        assert ck['data_fingerprint']==CURRENT_DATA_FINGERPRINT
        filename='legacy_'+entry['weights'];torch.save(ck,OUT/filename)
        components.append({'weights':filename,'geometry':entry['geometry'],'weight':1/6,'model_class':'LegacyUNet'})
    assert len(components)==4 and abs(sum(e['weight'] for e in components)-1)<1e-9
    recipe={'components':components,'threshold':selected['threshold'],'data_fingerprint':CURRENT_DATA_FINGERPRINT,'source_sha256':SOURCE_HASH};save_json('ensemble.json',recipe)
    def regenerate():
        models=[]
        for entry in components:
            model=(FRNet if entry['model_class']=='FRNet' else LegacyUNet)().cuda()
            ck=torch.load(OUT/entry['weights'],map_location='cpu',weights_only=False);assert ck['data_fingerprint']==CURRENT_DATA_FINGERPRINT
            model.load_state_dict(ck['model']);models.append(model)
        rows=[];probs=[]
        for i in range(1,21):
            if time.time()-START>CAP:raise RuntimeError('Bounded inference cap reached')
            image=Image.open(root/'test/image'/f'{i}.jpg').convert('RGB');total=None
            for entry,model in zip(components,models):
                view=image if entry['geometry']=='native' else image.resize((512,512),Image.Resampling.BILINEAR)
                term=infer(model,preprocess(np.array(view)))*entry['weight'];total=term if total is None else total+term
            probs.append(total);rows.append({'Id':str(i),'Predicted':encode_mask(total>recipe['threshold'])})
        del models;torch.cuda.empty_cache();return rows,np.stack(probs)
    rows,p=regenerate();again,q=regenerate();assert rows==again and np.array_equal(p,q),'Saved ensemble reproduction mismatch'
    with (OUT/'submission.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['Id','Predicted']);writer.writeheader();writer.writerows(rows)
    np.savez_compressed(OUT/'test_probabilities.npz',probabilities=p)
    return {'submission_sha256':hashlib.sha256((OUT/'submission.csv').read_bytes()).hexdigest(),'model_files':[e['weights'] for e in components],'saved_ensemble_reproduction_verified':True}

"""Exact paired D4 crop augmentation; independent RNG from crop selection."""
class MultiscaleFR(nn.Module):
    def __init__(self):
        super().__init__();self.branches=nn.ModuleList([FRNet() for _ in range(4)]);self.fusion_logits=nn.Parameter(torch.zeros(4));self.scales=(1.,.75,.5,.25)
    def forward(self,x):
        h,w=x.shape[-2:];outputs=[]
        for scale,branch in zip(self.scales,self.branches):
            size=(max(8,round(h*scale/8)*8),max(8,round(w*scale/8)*8))
            view=x if size==(h,w) else F.interpolate(x,size=size,mode='bilinear',align_corners=False)
            z=branch(view)
            outputs.append(F.interpolate(z,size=(h,w),mode='bilinear',align_corners=False))
        weights=self.fusion_logits.softmax(0)
        return sum(weights[i]*z for i,z in enumerate(outputs))

def select_primary(development):
    return 'equal_triple_blend' if development['equal_triple_blend']['passed'] else None


def find_references(truth,fp):
    refs={}
    for path in Path('/kaggle/input').rglob('metrics.json'):
        m=json.loads(path.read_text())
        for name,sha in EXPECTED.items():
            if m.get('source_sha256')==sha:
                assert name not in refs and m['data_fingerprint']==fp
                refs[name]=path.parent
    assert set(refs)==set(EXPECTED),list(refs)
    arrays=[np.load(refs['aligned']/'native_control_oof.npz'),np.load(refs['aligned']/'aligned512_oof.npz'),np.load(refs['photo']/'photo_oof.npz')]
    parent=np.load(refs['seed_parent']/'research_oof.npz');assert np.array_equal(parent['truth'],truth) and all(np.array_equal(a['truth'],truth) for a in arrays)
    metrics=json.loads((refs['seed_parent']/'metrics.json').read_text());assert metrics['complete'] and metrics['confirmation']['passed'] and abs(metrics['candidate']['crossfit_dice']-.8283275048875648)<1e-12
    return refs,parent['probabilities'],sum(a['probabilities'] for a in arrays)/3

def reused_parent_fold(refs,data,truth,valid,fp,fold):
    path=refs['seed_parent'];meta=json.loads((path/'metrics.json').read_text());assert meta['source_sha256']==EXPECTED['seed_parent'] and meta['data_fingerprint']==fp
    name='multiscale_fr_fold'+str(fold);info=meta['fits'][name];assert info['architecture']=='multiscale_fr' and info['fold']==fold and info['successful_updates']==2016
    z=np.load(path/(name+'_predictions.npz'));assert np.array_equal(z['truth'],truth[valid])
    ck=torch.load(path/(name+'_model.pt'),map_location='cpu',weights_only=False);assert ck['source_sha256']==EXPECTED['seed_parent'] and ck['data_fingerprint']==fp and ck['seed']==20261031 and ck['epochs']==168 and ck['successful_updates']==2016 and ck['architecture']=='multiscale_fr'
    assert ck['train_ids']==[i+1 for i in range(20) if i not in valid]
    model=MultiscaleFR().cuda();model.load_state_dict(ck['model']);p=np.stack([infer(model,data[i][0]) for i in valid]);del model;torch.cuda.empty_cache()
    drift=float(np.max(np.abs(p-z['probabilities'])));dice_drift=abs(binary_metrics(p,truth[valid],.5,valid)['dice']-info['fixed_05']['dice']);assert drift<=4*np.finfo(np.float16).eps and dice_drift<=1e-4
    return p,{'reused':True,'source_sha256':EXPECTED['seed_parent'],'seed':20261031,'fold':fold,'original_fit':info,'checkpoint_sha256':hashlib.sha256((path/(name+'_model.pt')).read_bytes()).hexdigest(),'prediction_sha256':hashlib.sha256((path/(name+'_predictions.npz')).read_bytes()).hexdigest(),'max_abs_drift':drift,'fixed05_dice_drift':dice_drift}

def reused_control(refs,data,truth,valid,fp):
    return reused_parent_fold(refs,data,truth,valid,fp,0)

def reused_confirmation(refs,data,truth,valid,fp):
    return reused_parent_fold(refs,data,truth,valid,fp,4)

def optimizer_groups(model):
    return [{'params':list(model.parameters()),'lr':.001,'base_lr':.001}]

def fit(data,ids,augment,name,epochs,valid=None):
    seed_all();rng=np.random.default_rng(SEED);aug_rng=np.random.default_rng(SEED+999)
    model=(MultiscaleFR() if TRAIN_ARCH=='multiscale_seed32' else FRNet()).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000)
    history=[];first=1;last=OUT/(name+'_last.pt');resume=None
    # A future recovery notebook attaches the previous private notebook as a data source.
    resume_roots=[Path(os.environ['RESUME_DIR'])] if os.environ.get('RESUME_DIR') else list(Path('/kaggle/input').glob('**/'+name+'_last.pt'))
    for location in resume_roots:
        checkpoint=location/(name+'_last.pt') if location.is_dir() else location
        if checkpoint.exists():resume=checkpoint;break
    identity={'source':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'data':CURRENT_DATA_FINGERPRINT,'patch':PATCH,'steps':STEPS,'batch':BATCH,'augmentation':augment,'architecture':TRAIN_ARCH,'supervision':'plain'}
    if resume is not None:
        ck=torch.load(resume,map_location='cuda',weights_only=False)
        require_checkpoint_identity(ck,identity)
        assert ck['seed']==SEED and ck['epochs']==epochs and ck['ids']==ids and ck['name']==name
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);rng.bit_generator.state=ck['rng'];aug_rng.bit_generator.state=ck['aug_rng'];torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']]);first=ck['epoch']+1;history=ck['history'];event(stage='resumed',name=name,epoch=first)
    for epoch in range(first,epochs+1):
        model.train();losses=[]
        for _ in range(STEPS):
            if time.time()-START>CAP:raise RuntimeError('Bounded GPU runtime exhausted; resume from last complete epoch')
            xs=[];ys=[]
            for _ in range(BATCH):
                i=int(rng.choice(ids));x,y,_=data[i];h,w=y.shape;top=int(rng.integers(h-PATCH+1));left=int(rng.integers(w-PATCH+1));x=x[:,top:top+PATCH,left:left+PATCH];y=y[top:top+PATCH,left:left+PATCH]
                if augment:
                    k=int(aug_rng.integers(4));x=np.rot90(x,k,(1,2));y=np.rot90(y,k)
                    if aug_rng.random()<.5:x=x[:,:,::-1];y=y[:,::-1]
                xs.append(x.copy());ys.append(y[None].copy())
            x=torch.from_numpy(np.stack(xs)).cuda();y=torch.from_numpy(np.stack(ys)).cuda();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                loss=candidate_loss(model,x,y,VARIANT)
            scaler.scale(loss).backward();scaler.unscale_(opt);finite_gradients(loss,model);scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
        for group in opt.param_groups:group['lr']=.001*(.1+.9*(1+np.cos(np.pi*epoch/epochs))/2)
        history.append({'epoch':epoch,'loss':float(np.mean(losses))});ck={'identity':identity,'name':name,'seed':SEED,'epochs':epochs,'ids':ids,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'rng':rng.bit_generator.state,'aug_rng':aug_rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'history':history}
        if epoch%5==0 or epoch==epochs:
            part=last.with_suffix('.tmp');torch.save(ck,part);part.replace(last);event(experiment=name,epoch=epoch,loss=history[-1]['loss'],seconds=time.time()-START)
    (OUT/(name+'_history.json')).write_text(json.dumps(history,indent=2))
    return model,{}
"""Candidate architecture/transfer package; frozen recipes and held-out gates."""
import traceback
TRAIN_ARCH='multiscale_seed32';VARIANT='plain'
PARENT_CONTROL='e025427e8164a839e35c1a29489b3a1c3a37ad24dc7d0d112eaabcfdb42b5005'
EXPECTED={'fr':'97a8122c79359cf886284228a0f45a3ff6bfc426adf5f63ff714f70ae0430261','aligned':'6274b8ce6054dd790340d660b2045418bb76f139fe00d877c7c3fe070e33404f','photo':'e64fc8ca895e36b21717999382f6679d3cfc3671dd8efb61b67556881ffccaef','paired':PARENT_CONTROL,'confirmation_control':'b371f69935844d4183115828a66463fab3bef13431c1dfe5aac72f0af0e1a660','seed_parent':'f8c9721411a8644b0faf744fe348a0fc6dff3a6053913d7314c735ebb508e38d'}


def gpu_preflight_wide(data,train_ids):
    seed_all();model=MultiscaleFR().cuda();opt=torch.optim.AdamW(optimizer_groups(model),weight_decay=.0001);scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000);rng=np.random.default_rng(SEED);aug_rng=np.random.default_rng(SEED+999);records=[]
    for step in range(3):
        model.train();x,y=batch(data,train_ids,False,rng);opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=segmentation_loss(model(x),y)
        scaler.scale(loss).backward();scaler.unscale_(opt);finite_gradients(loss,model);scaler.step(opt);scaler.update();records.append({'step':step,'loss':float(loss.detach()),'finite':True})
    p=infer(model,data[train_ids[0]][0]);assert p.shape==(512,512) and np.isfinite(p).all()
    proof={'passed':True,'steps':records,'full_image_inference_finite':True,'source_sha256':SOURCE_SHA,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'pretrained':{'used':False,'architecture':'HMS-inspired four independent FR resolution branches with learned fusion'}}
    save_json('gpu-preflight.json',proof);event(stage='gpu_preflight_passed',steps=3,source_sha256=SOURCE_SHA)
    del model,opt;torch.cuda.empty_cache();return proof


def run_fit(arch,fold,data,truth,folds):
    global TRAIN_ARCH
    TRAIN_ARCH=arch;valid=folds[fold];name=arch+'_fold'+str(fold);begin=time.time();torch.cuda.reset_peak_memory_stats()
    model,_=fit(data,[i for i in range(20) if i not in valid],False,name,168)
    train_seconds=time.time()-begin;peak=torch.cuda.max_memory_allocated();p=np.stack([infer(model,data[i][0]) for i in valid])
    torch.save({'model':model.state_dict(),'architecture':arch,'source_sha256':SOURCE_SHA,'data_fingerprint':CURRENT_DATA_FINGERPRINT,'train_ids':[i+1 for i in range(20) if i not in valid],'epochs':168,'successful_updates':2016,'seed':SEED},OUT/(name+'_model.pt'))
    result={'architecture':arch,'fold':fold,'training_seconds':train_seconds,'train_peak_allocated_bytes':peak,'parameters':sum(t.numel() for t in model.parameters()),'successful_updates':2016,'fixed_05':binary_metrics(p,truth[valid],.5,valid)}
    np.savez_compressed(OUT/(name+'_predictions.npz'),probabilities=p,truth=truth[valid]);save_json(name+'_fit.json',result);del model;torch.cuda.empty_cache();event(stage='fold_complete',architecture=arch,fold=fold,dice=result['fixed_05']['dice'],seconds=time.time()-START)
    return p,result

def pilot_recipes(p,paired,old_fr,triple,truth,valid):
    raw=binary_metrics(p,truth[valid],.5,valid);plain=binary_metrics(paired,truth[valid],.5,valid)
    deployed=binary_metrics((old_fr[valid]+triple[valid])/2,truth[valid],.5,valid);paired_blend=binary_metrics((paired+triple[valid])/2,truth[valid],.5,valid)
    rows={}
    for name,pred in [('standalone',p),('equal_triple_blend',(p+triple[valid])/2)]:
        q=binary_metrics(pred,truth[valid],.5,valid);passed=bool(raw['dice']>=plain['dice']-.002 and q['dice']>=max(deployed['dice'],paired_blend['dice'])+.001)
        rows[name]={'metrics':q,'raw_candidate':raw,'paired_fr':plain,'paired_fr_blend':paired_blend,'deployed_blend':deployed,'passed':passed}
    return rows

def full_refit(selected,recipe,root,data,refs,fp):
    global TRAIN_ARCH
    TRAIN_ARCH='multiscale_seed32';model,_=fit(data,list(range(20)),False,'multiscale_seed32_full',168)
    torch.save({'model':model.state_dict(),'configuration':selected,'architecture':'multiscale_seed32','source_sha256':SOURCE_SHA,'data_fingerprint':fp,'seed':SEED},OUT/'best_model.pt');del model;torch.cuda.empty_cache()
    components=[{'weights':'best_model.pt','geometry':'aligned512','weight':1. if recipe=='standalone' else .5,'model_class':'MultiscaleFR'}]
    if recipe=='equal_triple_blend':
        legacy=json.loads((refs['fr']/'ensemble.json').read_text())['components'][1:];assert len(legacy)==3
        for i,entry in enumerate(legacy):
            ck=torch.load(refs['fr']/entry['weights'],map_location='cpu',weights_only=False);assert ck['data_fingerprint']==fp
            filename='legacy_'+str(i)+'.pt';torch.save(ck,OUT/filename);components.append({'weights':filename,'geometry':entry['geometry'],'weight':1/6,'model_class':'LegacyUNet'})
    save_json('ensemble.json',{'components':components,'threshold':selected['threshold'],'source_sha256':SOURCE_SHA,'data_fingerprint':fp})
    def regenerate():
        nets=[]
        for entry in components:
            net=(MultiscaleFR() if entry['model_class']=='MultiscaleFR' else LegacyUNet()).cuda();ck=torch.load(OUT/entry['weights'],map_location='cpu',weights_only=False);assert ck['data_fingerprint']==fp;net.load_state_dict(ck['model']);nets.append(net)
        rows=[];probabilities=[]
        for i in range(1,21):
            if time.time()-START>CAP:raise RuntimeError('Bounded full inference cap exhausted')
            image=Image.open(root/'test/image'/f'{i}.jpg').convert('RGB');total=None
            for entry,net in zip(components,nets):
                view=image if entry['geometry']=='native' else image.resize((512,512),Image.Resampling.BILINEAR);term=infer(net,preprocess(np.array(view)))*entry['weight'];total=term if total is None else total+term
            probabilities.append(total);rows.append({'Id':str(i),'Predicted':encode_mask(total>selected['threshold'])})
        del nets;torch.cuda.empty_cache();return rows,np.stack(probabilities)
    rows,p=regenerate();again,q=regenerate();assert rows==again and np.array_equal(p,q),'Saved ensemble reconstruction mismatch'
    with (OUT/'submission.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['Id','Predicted']);writer.writeheader();writer.writerows(rows)
    np.savez_compressed(OUT/'test_probabilities.npz',probabilities=p)
    return {'submission_sha256':hashlib.sha256((OUT/'submission.csv').read_bytes()).hexdigest(),'saved_ensemble_reproduction_verified':True,'model_files':[e['weights'] for e in components]}


import copy

DENSE_EPOCHS=120
DENSE_STEPS=100
DENSE_BATCH=32
DENSE_PATCH=64
DENSE_DECAY=.999

class DenseFR(FR_UNet):
    def __init__(self):
        super().__init__(num_classes=1,num_channels=4,feature_scale=2,dropout=.2)

def dense_arrays(data,ids,rng,aug_rng,augment=True):
    xs=[];ys=[]
    assert ids and len(set(ids))==len(ids)
    for _ in range(DENSE_BATCH):
        i=int(rng.choice(ids));x,y,_=data[i];h,w=y.shape
        top=int(rng.integers(h-DENSE_PATCH+1));left=int(rng.integers(w-DENSE_PATCH+1))
        x=x[:,top:top+DENSE_PATCH,left:left+DENSE_PATCH];y=y[top:top+DENSE_PATCH,left:left+DENSE_PATCH]
        if augment:
            k=int(aug_rng.integers(4));flip=bool(aug_rng.integers(2));x=np.rot90(x,k,(1,2));y=np.rot90(y,k)
            if flip:x=x[:,:,::-1];y=y[:,::-1]
        xs.append(x.copy());ys.append(y[None].copy())
    return np.stack(xs),np.stack(ys)

def apply_d4(x,k,flip):
    x=torch.rot90(x,k,(-2,-1))
    return x.flip(-1) if flip else x

def undo_d4(x,k,flip):
    return torch.rot90(x.flip(-1) if flip else x,-k,(-2,-1))

@torch.no_grad()
def dense_infer(model,x):
    model.eval();device=next(model.parameters()).device;t=torch.from_numpy(x[None]).to(device);h,w=t.shape[-2:]
    t=F.pad(t,(0,(-w)%8,0,(-h)%8),mode='reflect');total=torch.zeros_like(t[:,:1],dtype=torch.float32)
    for k in range(4):
        for flip in [False,True]:
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                p=model(apply_d4(t,k,flip)).float().sigmoid()
            total.add_(undo_d4(p,k,flip)/8)
    q=total[0,0,:h,:w].cpu().numpy()
    assert q.shape==(512,512) and np.isfinite(q).all() and q.min()>=0 and q.max()<=1
    return q

@torch.no_grad()
def update_dense_ema(ema,model,decay=DENSE_DECAY):
    for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-decay)
    # Buffers are replaced, never averaged as parameters. Final BN is recalibrated.
    for a,b in zip(ema.buffers(),model.buffers()):a.copy_(b)

@torch.no_grad()
def calibrate_dense_bn(model,batches):
    model.eval();layers=[m for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)];momenta=[m.momentum for m in layers]
    for m in layers:m.reset_running_stats();m.momentum=None;m.train()
    try:
        count=0
        for x in batches:
            device=next(model.parameters()).device
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):model(x.to(device))
            count+=1
        assert count>0
    finally:
        for m,momentum in zip(layers,momenta):m.momentum=momentum
        model.eval()

def require_dense_resume(ck,identity):
    assert ck['identity']==identity,'Resume identity mismatch'
    epoch=ck['epoch'];assert isinstance(epoch,int) and 0<=epoch<=DENSE_EPOCHS
    assert ck['updates']==epoch*DENSE_STEPS,'Resume updates inconsistent'
    history=ck['history'];assert len(history)==epoch and [r['epoch'] for r in history]==list(range(1,epoch+1))
    assert all(np.isfinite(r['loss']) and r['successful_updates']==r['epoch']*DENSE_STEPS for r in history)
    assert all(torch.isfinite(t).all() for key in ['model','ema'] for t in ck[key].values())

def dense_gate(raw,parent,primary,archived,fresh):
    return bool(raw>=parent-.002 and primary>=max(archived,fresh)+.001)

def dense_pilot(p,paired,old_fr,triple,truth,valid):
    best=(old_fr[valid]+triple[valid])/2;fresh=(paired+triple[valid])/2
    raw=binary_metrics(p,truth[valid],.5,valid);control=binary_metrics(paired,truth[valid],.5,valid)
    primary=binary_metrics((p+best)/2,truth[valid],.5,valid);archive=binary_metrics(best,truth[valid],.5,valid);live=binary_metrics(fresh,truth[valid],.5,valid)
    return {'raw_candidate':raw,'paired_parent_raw':control,'metrics':primary,'deployed_best':archive,'fresh_best':live,'passed':dense_gate(raw['dice'],control['dice'],primary['dice'],archive['dice'],live['dice']),'recipe':'halfDense_halfCurrentBest'}

def dense_tensors(data,ids,rng,aug_rng,augment=True):
    x,y=dense_arrays(data,ids,rng,aug_rng,augment)
    return torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda()

def dense_step(model,ema,opt,scaler,x,y):
    model.train();opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):loss=segmentation_loss(model(x),y)
    scaler.scale(loss).backward();scaler.unscale_(opt);finite_gradients(loss,model)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
    old_scale=scaler.get_scale();scaler.step(opt);scaler.update();assert scaler.get_scale()>=old_scale,'Skipped optimizer update'
    update_dense_ema(ema,model)
    return float(loss.detach())

@torch.no_grad()
def dense_probe(model,data,ids):
    # Fixed train-only unaugmented crop bank; never reads validation or selects weights.
    rng=np.random.default_rng(SEED+2001);aug=np.random.default_rng(SEED+2002);values=[];model.eval()
    for _ in range(2):
        x,y=dense_tensors(data,ids,rng,aug,False)
        with torch.autocast('cuda',dtype=torch.float16):loss=segmentation_loss(model(x),y)
        values.append(float(loss))
    assert np.isfinite(values).all();return float(np.mean(values))

def fit_dense(data,ids,name):
    seed_all();rng=np.random.default_rng(SEED);aug_rng=np.random.default_rng(SEED+999)
    model=DenseFR().cuda();ema=copy.deepcopy(model).eval();ema.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000)
    identity={'source':SOURCE_SHA,'data':CURRENT_DATA_FINGERPRINT,'architecture':'denseFR32EMA','seed':SEED,'train_indices':ids,'epochs':DENSE_EPOCHS,'steps':DENSE_STEPS,'batch':DENSE_BATCH,'patch':DENSE_PATCH,'augmentation':'D4','ema':DENSE_DECAY,'inference':'8D4mean'}
    last=OUT/(name+'_last.pt');history=[];probes=[];first=1;updates=0
    locations=[Path(os.environ['RESUME_DIR'])/(name+'_last.pt')] if os.environ.get('RESUME_DIR') else list(Path('/kaggle/input').glob('**/'+name+'_last.pt'))
    assert len(locations)<=1,'Ambiguous resume checkpoints'
    if os.environ.get('RESUME_DIR'):assert locations[0].is_file(),'Explicit resume missing'
    if locations and locations[0].is_file():
        ck=torch.load(locations[0],map_location='cuda',weights_only=False);require_dense_resume(ck,identity)
        model.load_state_dict(ck['model']);ema.load_state_dict(ck['ema']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);rng.bit_generator.state=ck['rng'];aug_rng.bit_generator.state=ck['aug_rng'];torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']]);history=ck['history'];probes=ck['probes'];updates=ck['updates'];first=ck['epoch']+1;event(stage='resumed',experiment=name,epoch=first)
    else:probes.append({'epoch':0,'train_only_loss':dense_probe(model,data,ids)})
    for epoch in range(first,DENSE_EPOCHS+1):
        losses=[]
        for _ in range(DENSE_STEPS):
            k=updates;total=DENSE_EPOCHS*DENSE_STEPS;factor=.1+.9*(k+1)/200 if k<200 else .05+.95*(1+np.cos(np.pi*(k-200)/(total-200)))/2
            for g in opt.param_groups:g['lr']=.0003*factor
            x,y=dense_tensors(data,ids,rng,aug_rng);losses.append(dense_step(model,ema,opt,scaler,x,y));updates+=1
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'successful_updates':updates})
        if epoch%40==0:probes.append({'epoch':epoch,'train_only_loss':dense_probe(model,data,ids)})
        if epoch%5==0 or epoch==DENSE_EPOCHS:
            ck={'identity':identity,'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'updates':updates,'history':history,'probes':probes,'rng':rng.bit_generator.state,'aug_rng':aug_rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':[s.cpu() for s in torch.cuda.get_rng_state_all()]}
            part=last.with_suffix('.tmp');torch.save(ck,part);part.replace(last);event(experiment=name,epoch=epoch,loss=history[-1]['loss'],successful_updates=updates,seconds=time.time()-START)
    assert updates==12000 and len(history)==120
    calibration_rng=np.random.default_rng(SEED+3001);calibration_aug=np.random.default_rng(SEED+3002)
    calibrate_dense_bn(ema,(dense_tensors(data,ids,calibration_rng,calibration_aug)[0] for _ in range(64)))
    assert all(torch.isfinite(v).all() for v in ema.state_dict().values())
    save_json(name+'_history.json',history);save_json(name+'_train-probes.json',probes)
    del model,opt;torch.cuda.empty_cache();return ema

def dense_preflight(data,ids):
    seed_all();model=DenseFR().cuda();ema=copy.deepcopy(model).eval();ema.requires_grad_(False);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000);rng=np.random.default_rng(SEED);aug=np.random.default_rng(SEED+999)
    before=next(model.parameters()).detach().clone();records=[];torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        x,y=dense_tensors(data,ids,rng,aug);records.append({'step':step,'loss':dense_step(model,ema,opt,scaler,x,y),'finite':True})
    assert not torch.equal(before,next(model.parameters())) and not torch.equal(before,next(ema.parameters()))
    calibrate_dense_bn(ema,(dense_tensors(data,ids,rng,aug)[0] for _ in range(3)))
    p=dense_infer(ema,data[ids[0]][0]);assert np.isfinite(p).all()
    result={'passed':True,'optimizer_and_ema_updates_verified':True,'steps':records,'batch_shape':[32,4,64,64],'full_image_8D4_inference_finite':True,'parameters':sum(p.numel() for p in model.parameters()),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'source_sha256':SOURCE_SHA,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'external_weights':False}
    save_json('gpu-preflight.json',result);event(stage='gpu_preflight_passed',parameters=result['parameters'],steps=3);del model,ema,opt;torch.cuda.empty_cache();return result

def dense_fold(fold,data,truth,folds):
    valid=folds[fold];ids=[i for i in range(20) if i not in valid];name='dense64_fold'+str(fold);begin=time.time();torch.cuda.reset_peak_memory_stats();model=fit_dense(data,ids,name);p=np.stack([dense_infer(model,data[i][0]) for i in valid])
    ck={'model':model.state_dict(),'source_sha256':SOURCE_SHA,'data_fingerprint':CURRENT_DATA_FINGERPRINT,'train_ids':[i+1 for i in ids],'seed':SEED,'epochs':120,'successful_updates':12000,'architecture':'DenseFR','ema_decay':DENSE_DECAY,'bn_calibration_batches':64,'inference':'8D4mean'};torch.save(ck,OUT/(name+'_model.pt'))
    again=DenseFR().cuda();again.load_state_dict(torch.load(OUT/(name+'_model.pt'),map_location='cuda',weights_only=False)['model']);q=np.stack([dense_infer(again,data[i][0]) for i in valid]);assert np.array_equal(p,q),'Saved EMA reconstruction mismatch'
    np.savez_compressed(OUT/(name+'_predictions.npz'),probabilities=p,truth=truth[valid]);info={'fold':fold,'parameters':sum(t.numel() for t in model.parameters()),'successful_updates':12000,'patch_draws':384000,'training_and_inference_seconds':time.time()-begin,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'fixed_05':binary_metrics(p,truth[valid],.5,valid),'prediction_sha256':hashlib.sha256((OUT/(name+'_predictions.npz')).read_bytes()).hexdigest(),'saved_prediction_reproduction_verified':True};save_json(name+'_fit.json',info);event(stage='fold_complete',fold=fold,dice=info['fixed_05']['dice'],seconds=time.time()-START);del model,again;torch.cuda.empty_cache();return p,info

import shutil
PILOT_SHA='8638ad1b32023e1e10081f20ec07ed7153893f5a6782c0a85945d31401cff358'
DEPLOYED_SHA='b887a2fe2c1122fc0f76b47e692dad33a07f1e403dc27ab7a492577a46a82b8c'
DEPLOYED_HASHES={'best_model.pt':'b0d92cad3655787d4699787505a22fc445d5a2e2dd228036b1ee52a235ae6f97','legacy_0.pt':'7e7fb5214ba76b6c261e782e4512f5cb8c66635d256858cbaf64fbd40f43c9a3','legacy_1.pt':'aa2cebbadfd20cdb736a3245f3d4bbe3b4912eb5661d3ac0767b65d887dcf5e0','legacy_2.pt':'25064f5766cd589b08d12bc7f6df929c16677f6d7a590d9d0ffac8b6e8f823e6'}

def require_pilot_checkpoint(ck,fp,valid):
    expected={'source_sha256':PILOT_SHA,'data_fingerprint':fp,'train_ids':[i+1 for i in range(20) if i not in valid],'seed':20261031,'epochs':120,'successful_updates':12000,'architecture':'DenseFR','ema_decay':.999,'bn_calibration_batches':64,'inference':'8D4mean'}
    assert all(ck.get(k)==v for k,v in expected.items()),'Pilot checkpoint identity mismatch'

def full_gate(candidate,baseline,fixed,base_fixed):
    gain=candidate['crossfit_dice']-baseline['crossfit_dice'];wins=sum(a>b for a,b in zip(candidate['crossfit_fold_dice'],baseline['crossfit_fold_dice']));first=candidate['fold_dice'][0]-baseline['fold_dice'][0];fixed_gain=fixed-base_fixed
    return {'passed':bool(gain>.002 and wins>=4 and first>.001 and fixed_gain>=.001),'gain':gain,'fold_wins':wins,'first_fold_gain':first,'fixed05_gain':fixed_gain}

def submission_components(legacy):
    assert len(legacy)==4 and legacy[0]['model_class']=='MultiscaleFR'
    assert [e['model_class'] for e in legacy[1:]]==['LegacyUNet']*3
    assert [e['geometry'] for e in legacy]==['aligned512','native','aligned512','aligned512']
    assert [e['weight'] for e in legacy]==[.5,1/6,1/6,1/6]
    c=[{'weights':'best_model.pt','geometry':'aligned512','weight':.5,'model_class':'DenseFR','tta':'8D4mean'}]
    for i,e in enumerate(legacy):c.append({**e,'weights':f'currentbest_{i}.pt','weight':e['weight']/2,'tta':'none'})
    assert abs(sum(e['weight'] for e in c)-1)<1e-12;return c

def require_test_probabilities(p,threshold=.5):
    assert p.shape==(20,512,512) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    fractions=(p>threshold).mean((1,2));assert all(.005<v<.4 for v in fractions),'Implausible vessel fractions'

def unique_metrics(source):
    found=[]
    for path in Path('/kaggle/input').rglob('metrics.json'):
        m=json.loads(path.read_text())
        if m.get('source_sha256')==source:found.append((path.parent,m))
    assert len(found)==1,'Ambiguous/missing source metrics';return found[0]

def reuse_dense_pilot(path,meta,fold,data,truth,folds,fp):
    valid=folds[fold];name='dense64_fold'+str(fold);file=path/(name+'_predictions.npz');info=meta['fits']['fold'+str(fold)]
    assert info['saved_prediction_reproduction_verified'] and info['successful_updates']==12000
    assert hashlib.sha256(file.read_bytes()).hexdigest()==info['prediction_sha256']
    z=np.load(file);p=z['probabilities'];assert np.array_equal(z['truth'],truth[valid]) and p.shape==truth[valid].shape and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    history=json.loads((path/(name+'_history.json')).read_text());assert len(history)==120 and all(r['epoch']==i+1 and r['successful_updates']==100*(i+1) and np.isfinite(r['loss']) for i,r in enumerate(history))
    ck=torch.load(path/(name+'_model.pt'),map_location='cpu',weights_only=False);require_pilot_checkpoint(ck,fp,valid);assert all(torch.isfinite(t).all() for t in ck['model'].values())
    model=DenseFR().cuda();model.load_state_dict(ck['model']);q=np.stack([dense_infer(model,data[i][0]) for i in valid]);drift=float(np.max(np.abs(q-p)));dice_drift=abs(binary_metrics(q,truth[valid],.5,valid)['dice']-info['fixed_05']['dice']);assert drift<=4*np.finfo(np.float16).eps and dice_drift<=1e-4,'Pilot GPU reproduction drift'
    assert abs(binary_metrics(p,truth[valid],.5,valid)['dice']-info['fixed_05']['dice'])<1e-12
    for suffix in ['_predictions.npz','_model.pt','_fit.json','_history.json','_train-probes.json']:shutil.copy2(path/(name+suffix),OUT/(name+suffix))
    proof={'reused':True,'source_sha256':PILOT_SHA,'fold':fold,'checkpoint_sha256':hashlib.sha256((path/(name+'_model.pt')).read_bytes()).hexdigest(),'prediction_sha256':info['prediction_sha256'],'max_abs_drift':drift,'fixed05_dice_drift':dice_drift,'training_repeated':False};save_json(name+'_reuse.json',proof);event(stage='pilot_reused',fold=fold,max_abs_drift=drift)
    del model,ck;torch.cuda.empty_cache();return p,info,proof

def dense_full_refit(selected,root,data,fp):
    model=fit_dense(data,list(range(20)),'dense64_full')
    ck={'model':model.state_dict(),'configuration':selected,'architecture':'DenseFR','source_sha256':SOURCE_SHA,'data_fingerprint':fp,'seed':SEED,'train_ids':list(range(1,21)),'epochs':120,'successful_updates':12000,'ema_decay':.999,'bn_calibration_batches':64,'inference':'8D4mean'};torch.save(ck,OUT/'best_model.pt');del model,ck;torch.cuda.empty_cache()
    deployed,meta=unique_metrics(DEPLOYED_SHA);assert meta['complete'] and meta['data_fingerprint']==fp and meta['saved_ensemble_reproduction_verified']
    old=json.loads((deployed/'ensemble.json').read_text());assert old['source_sha256']==DEPLOYED_SHA and old['data_fingerprint']==fp;components=submission_components(old['components'])
    for i,e in enumerate(old['components']):
        f=deployed/e['weights'];assert hashlib.sha256(f.read_bytes()).hexdigest()==DEPLOYED_HASHES[e['weights']];c=torch.load(f,map_location='cpu',weights_only=False);assert c['data_fingerprint']==fp and all(torch.isfinite(t).all() for t in c['model'].values());shutil.copy2(f,OUT/f'currentbest_{i}.pt')
    save_json('ensemble.json',{'components':components,'threshold':selected['threshold'],'source_sha256':SOURCE_SHA,'data_fingerprint':fp,'composition':'(Dense8D4 + original_currentbest_component_sum)/2'})
    def regenerate():
        nets=[]
        for e in components:
            model=(DenseFR() if e['model_class']=='DenseFR' else MultiscaleFR() if e['model_class']=='MultiscaleFR' else LegacyUNet()).cuda();c=torch.load(OUT/e['weights'],map_location='cpu',weights_only=False);assert c['data_fingerprint']==fp;model.load_state_dict(c['model']);nets.append(model)
        probs=[]
        for i in range(1,21):
            image=Image.open(root/'test/image'/f'{i}.jpg').convert('RGB');dense=None;original=None
            for j,(e,net) in enumerate(zip(components,nets)):
                view=image if e['geometry']=='native' else image.resize((512,512),Image.Resampling.BILINEAR);x=preprocess(np.array(view))
                if j==0:dense=dense_infer(net,x)
                else:
                    term=infer(net,x)*(2*e['weight']);original=term if original is None else original+term
            probs.append((dense+original)/2)
        del nets;torch.cuda.empty_cache();return np.stack(probs)
    p=regenerate();q=regenerate();assert np.array_equal(p,q),'Disk-loaded ensemble reproduction mismatch';require_test_probabilities(p,selected['threshold'])
    rows=[{'Id':str(i+1),'Predicted':encode_mask(a>selected['threshold'])} for i,a in enumerate(p)]
    with (root/'sample_submission.csv').open(newline='') as f:sample=list(csv.DictReader(f));assert [r['Id'] for r in sample]==[r['Id'] for r in rows] and set(sample[0])=={'Id','Predicted'}
    with (OUT/'submission.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['Id','Predicted']);w.writeheader();w.writerows(rows)
    np.savez_compressed(OUT/'test_probabilities.npz',probabilities=p)
    hashes={e['weights']:hashlib.sha256((OUT/e['weights']).read_bytes()).hexdigest() for e in components};save_json('inference-proof.json',{'saved_ensemble_reproduction_verified':True,'model_hashes':hashes,'probabilities_sha256':hashlib.sha256((OUT/'test_probabilities.npz').read_bytes()).hexdigest(),'rows':20,'foreground_fractions':(p>selected['threshold']).mean((1,2)).tolist()})
    return {'submission_sha256':hashlib.sha256((OUT/'submission.csv').read_bytes()).hexdigest(),'saved_ensemble_reproduction_verified':True,'model_files':list(hashes)}

def main():
    global SOURCE_SHA,CURRENT_DATA_FINGERPRINT
    assert torch.cuda.is_available();torch.set_num_threads(2);cv2.setNumThreads(0);SOURCE_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();root,_=load_data();data=aligned_data(root);truth=np.stack([d[2] for d in data]);folds=[[i for i in range(20) if i%5==r] for r in [2,0,1,3,4]]
    fp=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((root/'train').rglob('*')) if p.is_file())).hexdigest();CURRENT_DATA_FINGERPRINT=fp;assert fp=='0c935bf22087c4098143370f99111b88e5b28488f9fcb39a7e0919964625ae2a'
    refs,old_fr,triple=find_references(truth,fp);best=(old_fr+triple)/2;base=crossfit_summary(best,truth,folds);assert abs(base['crossfit_dice']-.8283275048875648)<1e-12
    pilot,meta=unique_metrics(PILOT_SHA);assert meta['complete'] and meta['pilot_only'] and meta['promotion_ready'] and not meta['eligible_for_submission'] and meta['data_fingerprint']==fp and meta['recipe']=='halfDense_halfCurrentBest';assert all(meta[k]['passed'] for k in ['development','confirmation'])
    report={'complete':False,'eligible_for_submission':False,'source_sha256':SOURCE_SHA,'parent_source_sha256':PILOT_SHA,'data_fingerprint':fp,'baseline':base,'recipe':'halfDense_halfCurrentBest','fits':{},'experiments':[],'pilot_folds_retrained':0,'missing_folds_trained':[1,2,3],'external_data':False,'monitoring':False};p=np.zeros_like(old_fr)
    for fold,stage in [(0,'development'),(4,'confirmation')]:
        q,info,proof=reuse_dense_pilot(pilot,meta,fold,data,truth,folds,fp);p[folds[fold]]=q;report['fits']['fold'+str(fold)]=info;report['reuse_fold'+str(fold)]=proof
        measured=dense_pilot(q,old_fr[folds[fold]],old_fr,triple,truth,folds[fold]);assert measured['passed'] and abs(measured['metrics']['dice']-meta[stage]['metrics']['dice'])<1e-12;report[stage]=measured
    save_json('reused-pilot-validation.json',report)
    report['gpu_preflight']=dense_preflight(data,[i for i in range(20) if i not in folds[1]])
    for fold in [1,2,3]:
        q,info=dense_fold(fold,data,truth,folds);p[folds[fold]]=q;report['fits']['fold'+str(fold)]=info;save_json('fullcv-progress.json',report)
    primary=(p+best)/2;candidate=crossfit_summary(primary,truth,folds);fixed=binary_metrics(primary,truth,.5,list(range(20)));base_fixed=binary_metrics(best,truth,.5,list(range(20)));gate=full_gate(candidate,base,fixed['dice'],base_fixed['dice']);np.savez_compressed(OUT/'research_oof.npz',probabilities=p,selected_probabilities=primary,truth=truth)
    selected={'experiment':'dense64_ema8D4_half_currentbest','dice':candidate['fold_dice'][0],'oof_dice':candidate['oof_dice'],'crossfit_dice':candidate['crossfit_dice'],'epoch':120,'threshold':candidate['threshold'],'augment':True,'per_image_dice':[candidate['image_dice'][i] for i in folds[0]]};report.update(selected=selected,experiments=[selected],candidate=candidate,gate=gate,fixed_05=fixed);save_json('validation-selection.json',report);event(stage='fullcv_complete',gate=gate)
    if not gate['passed']:
        report.update(complete=True,stop_reason='Dense64 fullCV gate failed; no automatic refit or submission',gpu_seconds=time.time()-START);save_json('metrics.json',report);return
    report.update(dense_full_refit(selected,root,data,fp));report.update(complete=True,eligible_for_submission=True,gpu_seconds=time.time()-START,stop_reason='FullCV passed; saved CSV ready for authorized single submission');save_json('metrics.json',report);event(stage='complete',eligible=True,submission_sha256=report['submission_sha256'])

if __name__=='__main__':
    try:main()
    except Exception as error:save_json('failure.json',{'error':str(error),'traceback':traceback.format_exc(),'gpu_seconds':time.time()-START});raise
