"""Reproducible, offline engineering smoke tests; NOT a cross-platform benchmark."""
import argparse, json, platform, random, time, os
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent.parent / 'results'; OUT.mkdir(exist_ok=True)


def xnor_dot(a, b):
    """Exact dot product over {-1,+1}; pack sign bits in a Python integer."""
    assert len(a) == len(b) and len(a)>0
    n=len(a)
    assert all(int(x) in (-1,1) for x in a) and all(int(x) in (-1,1) for x in b)
    bits_a = sum((int(v)>0)<<i for i,v in enumerate(a))
    bits_b = sum((int(v)>0)<<i for i,v in enumerate(b))
    matches = (~(bits_a^bits_b) & ((1<<n)-1)).bit_count()
    return 2*matches-n


def binary_proof():
    rng = np.random.default_rng(4107)
    sizes=[1,7,8,9,63,64,65,127,129,257]
    count=0
    for n in sizes:
        for _ in range(100):
            a = rng.choice([-1,1], size=n).astype(np.int64)
            b = rng.choice([-1,1], size=n).astype(np.int64)
            assert xnor_dot(a,b) == int(np.dot(a,b)), (n,a,b)
            count+=1
    report={'test':'exact_binary_dot_correctness','vectors_checked':count,'sizes':sizes,'status':'PASS','note':'Python integer XNOR/popcount only; no speed or model-accuracy claim.'}
    (OUT/'binary_correctness.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(report)

class TinyTransformer(nn.Module):
    def __init__(self,vocab_size,block_size,emb=32,heads=4):
        super().__init__()
        self.block=block_size
        self.tok=nn.Embedding(vocab_size,emb)
        self.pos=nn.Embedding(block_size,emb)
        self.attn=nn.MultiheadAttention(emb,heads,dropout=0,batch_first=True)
        self.norm1=nn.LayerNorm(emb)
        self.ff=nn.Sequential(nn.Linear(emb,emb*2),nn.GELU(),nn.Linear(emb*2,emb))
        self.norm2=nn.LayerNorm(emb)
        self.out=nn.Linear(emb,vocab_size)
    def forward(self,x,y=None):
        B,T=x.shape
        h=self.tok(x)+self.pos(torch.arange(T,device=x.device))
        mask=torch.ones(T,T,device=x.device,dtype=torch.bool).triu(1)
        a,_=self.attn(h,h,h,attn_mask=mask,need_weights=False)
        h=self.norm1(h+a)
        h=self.norm2(h+self.ff(h))
        logits=self.out(h)
        if y is None:return logits
        return F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1))

def cpu_demo(steps=100):
    torch.set_num_threads(2)
    torch.manual_seed(3401); random.seed(3401)
    # Intentionally synthetic training data, no claim to natural-language competence.
    names=['Лиса','Сова','Кот','Ёж','Волк','Кит','Гусь','Уж']
    actions=['нашёл','увидел','запомнил','принёс','укрыл','сохранил','потерял','поднял']
    things=['зонт','снег','камень','лист','ответ','ключ','чертёж','план']
    lines=[f'{a} {b} {c}.\n' for a in names for b in actions for c in things]
    rng=random.Random(3401);rng.shuffle(lines)
    n=len(lines); texts={
        'train':''.join(lines[:int(n*0.8)]),
        'val':''.join(lines[int(n*0.8):int(n*0.9)]),
        'test':''.join(lines[int(n*0.9):])
    }
    alphabet=sorted(set(''.join(texts.values())))
    stoi={c:i for i,c in enumerate(alphabet)}
    arrays={k:torch.tensor([stoi[c] for c in s],dtype=torch.long) for k,s in texts.items()}
    block=32; batch=12
    def batch_sample(part):
        seq=arrays[part]; starts=torch.randint(0,len(seq)-block-1,(batch,))
        x=torch.stack([seq[i:i+block] for i in starts]); y=torch.stack([seq[i+1:i+block+1] for i in starts])
        return x,y
    model=TinyTransformer(len(alphabet),block)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3)
    counts=np.bincount(arrays['train'].numpy(),minlength=len(alphabet)).astype(float)
    probs=(counts+1)/(counts.sum()+len(alphabet))
    unigram_test=float(np.mean(-np.log(probs[arrays['test'].numpy()])))
    history=[]
    t0=time.perf_counter(); wall_start=time.time()
    for i in range(1,steps+1):
        model.train();x,y=batch_sample('train')
        loss=model(x,y)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if i==1 or i%20==0 or i==steps:
            model.eval()
            with torch.no_grad():
                losses=[]
                for _ in range(8):
                    a,b=batch_sample('val');losses.append(float(model(a,b)))
                history.append({'step':i,'train_minibatch_loss':round(float(loss.detach()),5),'val_sampled_loss':round(float(np.mean(losses)),5)})
    train_seconds=time.perf_counter()-t0
    # Fixed, deterministic exhaustive sliding-window evaluation on TEST (no random sample).
    model.eval()
    with torch.no_grad():
        test=arrays['test']; total=0.; batches=0
        for start in range(0,len(test)-block-1,32):
            starts=list(range(start,min(start+32,len(test)-block-1)))
            x=torch.stack([test[j:j+block] for j in starts]); y=torch.stack([test[j+1:j+block+1] for j in starts])
            total+=float(model(x,y))*len(starts);batches+=len(starts)
        test_loss=total/batches
    report={
        'run_type':'small CPU smoke test on artificially templated character strings',
        'cpu':platform.processor() or platform.machine(), 'platform':platform.platform(),
        'torch_version':torch.__version__,'cpu_only':not next(model.parameters()).is_cuda,
        'cuda_available':torch.cuda.is_available(),
        'num_cpu_threads':torch.get_num_threads(),
        'seed':3401,'train_lines':int(n*.8),'val_lines':int(n*.9)-int(n*.8),'test_lines':n-int(n*.9),
        'characters':{k:len(v) for k,v in texts.items()},'vocab_size':len(alphabet),
        'params':sum(p.numel() for p in model.parameters()),'context_chars':block,
        'steps':steps,'batch_size':batch,'training_wall_seconds':round(train_seconds,4),
        'training_tokens_processed':int(steps*batch*block),
        'train_tokens_per_second':round(steps*batch*block/train_seconds,2),
        'unigram_test_cross_entropy':round(unigram_test,5),
        'transformer_test_cross_entropy':round(test_loss,5),
        'unigram_test_perplexity':round(float(np.exp(unigram_test)),4),
        'transformer_test_perplexity':round(float(np.exp(test_loss)),4),
        'history':history,
        'limits':'Very small synthetic templated data; this does NOT establish useful LLM capability, real corpus generalization, cloud economics or other CPU performance.'
    }
    (OUT/'cpu_toy_smoke.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('CPU smoke:',{k:report[k] for k in ('params','training_wall_seconds','training_tokens_processed','train_tokens_per_second','unigram_test_cross_entropy','transformer_test_cross_entropy')})
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=100);args=p.parse_args()
    binary_proof();cpu_demo(steps=args.steps)
