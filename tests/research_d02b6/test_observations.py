import pytest
from mfvlr_research.d02b6.observations import objective_statistics
from mfvlr_research.d02b1.schema import InvalidRecord

@pytest.mark.parametrize('mutation',(None,'missing','padding','kl_detach','nonfinite','bounded_reconstruction'))
def test_objective_metadata(mutation):
    r=dict(scalars={p:0. for p in ('fd','lr','cmc','fl','ar','kl','total')},
           lr=dict(categories={p:dict(count=n,nll_sum=0.) for p,n in [('bos',4),('eos',4),('pad',200),('content',100)]},all_positions=308,token_reduction='token_sum_then_batch_mean'),
           cmc=dict(temperature=.07,score_min=0.,score_max=0.,row_entropy=[0.],column_entropy=[0.],positive_indices=[0],duplicate_prompt_pairs=[]),
           kl=dict(temperature=.5,direction='text||adapted_vision',teacher_entropy=[1.],student_entropy=[1.],both_distributions_connected=True,temperature_squared_factor=False),
           appearance=dict(image_min=0.,image_max=1.,reconstruction_min=-10.,reconstruction_max=100.))
    if mutation=='missing':del r['scalars']['ar']
    if mutation=='padding':r['lr']['categories']['pad']['count']=0
    if mutation=='kl_detach':r['kl']['both_distributions_connected']=False
    if mutation=='nonfinite':r['cmc']['score_max']=float('inf')
    if mutation=='bounded_reconstruction':r['appearance']['reconstruction_max']=100000.
    if mutation in (None,'bounded_reconstruction'):assert objective_statistics(r,1)['execution']=='NOT RUN'
    else:
        with pytest.raises(InvalidRecord):objective_statistics(r,1)
