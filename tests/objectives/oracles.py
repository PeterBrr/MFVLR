"""Pure mathematical references, independent of mfvlr production functions.

Python float arithmetic supplies independent FP64 scalar/gradient expectations.
No production CE/KL/CMC or torch log_softmax helper is imported here.
"""
import json
import math
from pathlib import Path

import torch

FIXTURES=json.loads(Path(__file__).with_name('reference_fixtures.json').read_text())
FP32={'atol':1e-6,'rtol':1e-5}
FP64={'atol':1e-10,'rtol':1e-8}
FD={'atol':1e-5,'rtol':1e-3}


def logsumexp(row):
    maximum=max(row)
    return maximum+math.log(math.fsum(math.exp(x-maximum) for x in row))


def probabilities(row):
    normalizer=logsumexp(row)
    return [math.exp(x-normalizer) for x in row]


def categorical(rows,labels):
    return math.fsum(logsumexp(row)-row[target] for row,target in zip(rows,labels))/len(rows)


def fd_gradient(rows,labels):
    return [[(p-(j==target))/len(rows) for j,p in enumerate(probabilities(row))]
            for row,target in zip(rows,labels)]


def cmc_reference(visual,language,theta):
    tau=math.exp(theta)
    scores=[[math.fsum(x*y for x,y in zip(v,l))/tau for l in language] for v in visual]
    reverse=list(map(list,zip(*scores)))
    b=len(scores)
    value=(categorical(scores,list(range(b)))+categorical(reverse,list(range(b))))/2
    rows=[probabilities(row) for row in scores]
    columns=list(map(list,zip(*[probabilities(row) for row in reverse])))
    g=[[(rows[i][j]+columns[i][j]-2*(i==j))/(2*b) for j in range(b)] for i in range(b)]
    gv=[[math.fsum(g[i][j]*language[j][k] for j in range(b))/tau for k in range(len(visual[0]))] for i in range(b)]
    gl=[[math.fsum(g[i][j]*visual[i][k] for i in range(b))/tau for k in range(len(language[0]))] for j in range(b)]
    gt=-math.fsum(g[i][j]*scores[i][j] for i in range(b) for j in range(b))
    return value,scores,gv,gl,gt


def kl_reference(text,adapted):
    logp=[];logq=[];p=[];q=[]
    for a,b in zip(text,adapted):
        a=[x/0.5 for x in a];b=[x/0.5 for x in b]
        na=logsumexp(a);nb=logsumexp(b)
        logp.append([x-na for x in a]);logq.append([x-nb for x in b])
        p.append([math.exp(x) for x in logp[-1]]);q.append([math.exp(x) for x in logq[-1]])
    batch=len(text)
    per=[math.fsum(x*(lp-lq) for x,lp,lq in zip(pp,a,b)) for pp,a,b in zip(p,logp,logq)]
    gt=[[x*((lp-lq)-d)/(batch*0.5) for x,lp,lq in zip(pp,a,b)] for pp,a,b,d in zip(p,logp,logq,per)]
    gs=[[(y-x)/(batch*0.5) for x,y in zip(pp,qq)] for pp,qq in zip(p,q)]
    return math.fsum(per)/batch,gt,gs,p,q


def cmc_fixture(batch=2,dtype=torch.float64,scale=0.1):
    v=torch.zeros(batch,1,512,dtype=dtype)
    l=torch.zeros_like(v)
    v[:,:,:2]=torch.tensor(FIXTURES['cmc']['visual_first_two'][:batch],dtype=dtype)[:,None,:]*scale
    l[:,:,:2]=torch.tensor(FIXTURES['cmc']['language_first_two'][:batch],dtype=dtype)[:,None,:]*scale
    return v.requires_grad_(),l.requires_grad_()


def kl_fixture(batch=1,dtype=torch.float64):
    p=torch.tensor(FIXTURES['kl']['P_pair']*256,dtype=dtype)/256
    q=torch.tensor(FIXTURES['kl']['Q_pair']*256,dtype=dtype)/256
    return (0.5*p.log()).view(1,1,512).repeat(batch,1,1).requires_grad_(),(0.5*q.log()).view(1,1,512).repeat(batch,1,1).requires_grad_()


def central_difference(function,tensor,index,epsilon=1e-6):
    """Mutate only a controlled test fixture; no optimizer/update is involved."""
    with torch.no_grad():
        original=tensor[index].item()
        try:
            tensor[index]=original+epsilon;plus=function().item()
            tensor[index]=original-epsilon;minus=function().item()
        finally:
            tensor[index]=original
    return (plus-minus)/(2*epsilon)
