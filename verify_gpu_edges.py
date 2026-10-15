"""GPU integration checks for the optimized decoder's boundary behavior."""
import numpy as np
from bert_punct import ROOT, Predictor, read_json, write_json, render


def main():
    p=Predictor(read_json(ROOT/'config.json'))
    phrase='学而时习之不亦说乎甲𰻞A１２3乙'
    lengths=[0,1,299,300,301,399,400,401,599,600,601,999,1000]
    texts=[(phrase*100)[:n] for n in lengths]
    expected=p.fixed(texts,16)
    result=p.fixed_fast(texts,16)
    assert all(np.array_equal(a,b) for a,b in zip(expected,result))
    assert all(len(t)==len(y) for t,y in zip(texts,result))
    assert p.fixed_fast([],16)==[]
    assert len(p.fixed_fast([''],16)[0])==0
    assert render('𰻞',[0])=='𰻞'
    try:
        p.fixed_fast(['甲'],0)
    except ValueError:
        pass
    else:
        raise AssertionError('zero batch accepted')
    write_json(ROOT/'artifacts/v2/gpu-edge-tests.json',{
        'status':'passed','checkpoint':p.cfg['checkpoint'],'lengths':lengths,
        'checks':['empty list and empty document','short/ownership/window/tail boundaries',
                  'mixed document lengths and unknown characters','fixed16/fast16 exact equality',
                  'invalid batch rejected']})
    print('GPU EDGE CHECKS PASSED')


if __name__=='__main__':
    main()
