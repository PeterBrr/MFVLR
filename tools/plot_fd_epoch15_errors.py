"""Plot stored validation previews with TP/FP/FN colors; no inference or editing."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

root=Path(__file__).resolve().parents[1]/'artifacts/genface_fd_epoch15_kaggle_20261027'
previews=[np.load(root/f'downloaded/fd_validation_results/seed_{s}_fd_0p1/val/preview.npz',allow_pickle=False) for s in range(3)]
for p in previews[1:]:
    for i in range(4):
        for k in ('image','target'):np.testing.assert_array_equal(p[f'{i}_{k}'],previews[0][f'{i}_{k}'])
canvas=Image.new('RGB',(224*5,280*4),'white');draw=ImageDraw.Draw(canvas)
for i in range(4):
    panels=[(previews[0][f'{i}_image'],'Input'),(previews[0][f'{i}_target'],'GT')]
    for seed,p in enumerate(previews):
        gt=p[f'{i}_target'].astype(bool);pred=p[f'{i}_prediction'].astype(bool)
        a=np.zeros((*gt.shape,3),dtype=np.float32)
        a[gt&pred]=[0,1,0];a[~gt&pred]=[1,0,0];a[gt&~pred]=[0,.4,1]
        panels.append((a,f'Seed {seed}: TP green / FP red / FN blue'))
    for j,(a,title) in enumerate(panels):
        draw.text((j*224+3,i*280+20),title,fill='black')
        canvas.paste(Image.fromarray(np.rint(np.clip(a,0,1)*255).astype('uint8')).convert('RGB'),(j*224,i*280+48))
canvas.save(root/'validation_error_previews.png')
print(root/'validation_error_previews.png')
