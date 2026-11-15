"""Read-only recomputation before the separately authorized submission POST."""
import csv,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];sys.path.insert(0,str(R))
from protocol import validate_rows,decode

def main():
    out=H/'outputs/v1';m=json.loads((out/'metrics.json').read_text());sha=hashlib.sha256((H/'train.py').read_bytes()).hexdigest()
    assert m['complete'] and m['eligible_for_submission'] and m['source_sha256']==sha
    assert m['parent_source_sha256']=='8638ad1b32023e1e10081f20ec07ed7153893f5a6782c0a85945d31401cff358'
    assert m['data_fingerprint']=='0c935bf22087c4098143370f99111b88e5b28488f9fcb39a7e0919964625ae2a'
    assert json.loads((H/'kernel-spec.json').read_text())['text'].encode()==(H/'train.py').read_bytes()
    s=importlib.util.spec_from_file_location('dense_submission_source',H/'train.py');module=importlib.util.module_from_spec(s);s.loader.exec_module(module)
    parent=np.load(R/'runs/fr_multiscale_v1/audit/research_oof.npz');truth=parent['truth'];best=parent['selected_probabilities'];folds=[[i for i in range(20) if i%5==r] for r in [2,0,1,3,4]];p=np.zeros_like(best)
    for fold,ids in enumerate(folds):
        file=out/f'dense64_fold{fold}_predictions.npz';z=np.load(file);assert np.array_equal(z['truth'],truth[ids]);q=z['probabilities'];assert q.shape==truth[ids].shape and np.isfinite(q).all() and q.min()>=0 and q.max()<=1
        assert hashlib.sha256(file.read_bytes()).hexdigest()==m['fits']['fold'+str(fold)]['prediction_sha256'];p[ids]=q
    primary=(p+best)/2;candidate=module.crossfit_summary(primary,truth,folds);baseline=module.crossfit_summary(best,truth,folds)
    for key in ['oof_dice','crossfit_dice','threshold']:assert abs(candidate[key]-m['candidate'][key])<1e-12
    for key in ['fold_dice','crossfit_fold_dice','crossfit_thresholds']:np.testing.assert_allclose(candidate[key],m['candidate'][key],rtol=0,atol=1e-12)
    fixed=module.binary_metrics(primary,truth,.5,list(range(20)))['dice'];base_fixed=module.binary_metrics(best,truth,.5,list(range(20)))['dice'];gate=module.full_gate(candidate,baseline,fixed,base_fixed);assert gate['passed'] and m['gate']['passed']
    for k in ['gain','fold_wins','first_fold_gain','fixed05_gain']:assert abs(gate[k]-m['gate'][k])<1e-12
    assert m['pilot_folds_retrained']==0 and m['missing_folds_trained']==[1,2,3]
    csvfile=out/'submission.csv';assert hashlib.sha256(csvfile.read_bytes()).hexdigest()==m['submission_sha256']
    with csvfile.open(newline='') as f:rows=list(csv.DictReader(f))
    validate_rows(rows);test=np.load(out/'test_probabilities.npz')['probabilities'];module.require_test_probabilities(test,m['selected']['threshold'])
    for row,q in zip(rows,test):assert np.array_equal(decode(row['Predicted']).astype(bool),q>m['selected']['threshold'])
    proof=json.loads((out/'inference-proof.json').read_text());assert proof['saved_ensemble_reproduction_verified'] and m['saved_ensemble_reproduction_verified'];assert hashlib.sha256((out/'test_probabilities.npz').read_bytes()).hexdigest()==proof['probabilities_sha256']
    for i,name in enumerate(['best_model.pt','legacy_0.pt','legacy_1.pt','legacy_2.pt']):assert proof['model_hashes'][f'currentbest_{i}.pt']==module.DEPLOYED_HASHES[name]
    result={'source_sha256':sha,'fullCV_prediction_metrics_independently_verified':True,'candidate':candidate,'gate':gate,'saved_ensemble_reproduction_verified_remote':True,'CSV_RLE_matches_test_probabilities':True,'submission_sha256':m['submission_sha256'],'rows':20,'pilot_folds_retrained':0,'eligible_for_submission':True}
    (H/'artifact-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps({'verified':True,'crossfit':candidate['crossfit_dice'],'gate':gate,'rows':20,'CSV_sha256':m['submission_sha256']}))

if __name__=='__main__':main()
