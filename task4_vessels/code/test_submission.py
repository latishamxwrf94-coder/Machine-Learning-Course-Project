import importlib.util,unittest,copy
from pathlib import Path
import numpy as np
def load():
 s=importlib.util.spec_from_file_location('dense_submit',Path(__file__).with_name('train.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class Tests(unittest.TestCase):
 def test_pilot_identity_and_ids(self):
  m=load();fp='fixture';ids=[i+1 for i in range(20) if i not in [2,7,12,17]];ck={'source_sha256':m.PILOT_SHA,'data_fingerprint':fp,'train_ids':ids,'seed':20261031,'epochs':120,'successful_updates':12000,'architecture':'DenseFR','ema_decay':.999,'bn_calibration_batches':64,'inference':'8D4mean'};m.require_pilot_checkpoint(ck,fp,[2,7,12,17])
  for key,value in [('source_sha256','bad'),('train_ids',list(range(1,21))),('successful_updates',11999),('inference','none')]:
   bad=dict(ck);bad[key]=value
   with self.assertRaises(AssertionError):m.require_pilot_checkpoint(bad,fp,[2,7,12,17])
 def test_gate_boundaries(self):
  m=load();base={'crossfit_dice':.8,'crossfit_fold_dice':[.8]*5,'fold_dice':[.8]*5};s={'crossfit_dice':.803,'crossfit_fold_dice':[.803]*4+[.799],'fold_dice':[.803]*5}
  self.assertTrue(m.full_gate(s,base,.802,.8)['passed'])
  for bad in [dict(s,crossfit_dice=.8019),dict(s,crossfit_fold_dice=[.803]*3+[.799]*2),dict(s,fold_dice=[.8009]*5)]:self.assertFalse(m.full_gate(bad,base,.802,.8)['passed'])
  self.assertFalse(m.full_gate(s,base,.8009,.8)['passed'])
 def test_recipe_weights_and_mask_validation(self):
  m=load();legacy=[{'weights':'best_model.pt','weight':.5,'geometry':'aligned512','model_class':'MultiscaleFR'}]+[{'weights':f'legacy_{i}.pt','weight':1/6,'geometry':'native' if i==0 else 'aligned512','model_class':'LegacyUNet'} for i in range(3)];c=m.submission_components(legacy);self.assertEqual(len(c),5);self.assertAlmostEqual(sum(x['weight'] for x in c),1.);self.assertEqual(c[0]['tta'],'8D4mean');self.assertEqual(c[1]['weight'],.25)
  p=np.zeros((20,512,512),np.float32);p[:,100:200,100:200]=1;m.require_test_probabilities(p)
  with self.assertRaises(AssertionError):m.require_test_probabilities(np.zeros_like(p))
  with self.assertRaises(AssertionError):m.require_test_probabilities(p[:19])
if __name__=='__main__':unittest.main()
