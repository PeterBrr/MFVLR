"""Stats-only A0 hooks; stage residuals observed at the next module boundary."""
class Capture:
    def __init__(self,root,stats):self.root=root;self.stats=stats;self.values={};self.handles=[];self.cnn_calls=0
    def put(self,name,x):
        if name in self.values:raise RuntimeError('Duplicate capture '+name)
        r=self.stats(x);self.values[name]={k:r[k] for k in ('shape','finite','rms','max_abs','l2')}
    def pre(self,module,name):
        self.handles.append(module.register_forward_pre_hook(lambda m,args:self.put(name,args[0])))
    def post(self,module,name):
        self.handles.append(module.register_forward_hook(lambda m,args,out:self.put(name,out)))
    def __enter__(self):
        r=self.root
        self.original_shift=r.flt.shift_low
        def shift(low):
            value=self.original_shift(low);self.put('LD.shifted_lookup',value);return value
        r.flt.shift_low=shift
        def cnn(m,args,out):
            name=('appearance.local','residual.local')[self.cnn_calls];self.cnn_calls+=1;self.put(name,out.local)
        self.handles.append(r.mve.shared_encoder.cnn.register_forward_hook(cnn))
        self.pre(r.flt.encoder,'LE.input');self.pre(r.flt.decoder,'LD.input')
        for i,b in enumerate(r.flt.encoder.blocks):self.post(b,f'LE.{i}.output')
        for i,b in enumerate(r.flt.decoder.blocks):
            self.pre(b.memory_attention,f'LD.{i}.self');self.pre(b.ff,f'LD.{i}.memory');self.pre(b.vim,f'LD.{i}.ff')
            self.post(b,f'LD.{i}.vim');self.post(b,f'LD.{i}.output')
        return self
    def finish(self,out):
        p=out.predictions;v=p.visual;l=out.language
        values={'appearance.global':v.appearance.global_token,'residual.global':v.residual_global,'Iv':v.fused_global,
            'Ipre':p.decoded.reconstruction,'residual.image':v.residual_image,'logits_fd':p.detection_logits,
            'logits_fl':p.decoded.mask_logits,'adapter':out.predicted_language,'low':l.low,'Tl':l.global_language,'logits_lr':l.logits}
        for n,x in values.items():self.put(n,x)
        from ..d02b1.stages import validate_stages
        validate_stages(self.values,len(out.sample_ids))
        return self.values
    def __exit__(self,*unused):
        for h in self.handles:h.remove()
        self.handles.clear()
        del self.root.flt.shift_low
