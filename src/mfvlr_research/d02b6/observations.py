"""Validate supplied objective-statistics coverage, never evaluate an objective."""
import math
from ..d02b1.schema import exact_keys,InvalidRecord,PASSES

def objective_statistics(record,batch):
    if type(batch) is not int or batch not in (1,2):raise InvalidRecord('B1/B2 only')
    exact_keys(record,('scalars','lr','cmc','kl','appearance'))
    exact_keys(record['scalars'],PASSES)
    exact_keys(record['lr'],('categories','all_positions','token_reduction'))
    exact_keys(record['lr']['categories'],('bos','eos','pad','content'))
    count=0
    for value in record['lr']['categories'].values():
        exact_keys(value,('count','nll_sum'))
        if type(value['count']) is not int or value['count']<0:raise InvalidRecord('Token category count')
        count+=value['count']
    if count!=batch*308 or record['lr']['all_positions']!=batch*308 or record['lr']['token_reduction']!='token_sum_then_batch_mean':
        raise InvalidRecord('LR token coverage/reduction')
    exact_keys(record['cmc'],('temperature','score_min','score_max','row_entropy','column_entropy','positive_indices','duplicate_prompt_pairs'))
    c=record['cmc']
    if c['temperature']<=0 or c['score_min']>c['score_max'] or c['positive_indices']!=list(range(batch)):
        raise InvalidRecord('CMC semantics')
    if len(c['row_entropy'])!=batch or len(c['column_entropy'])!=batch:raise InvalidRecord('CMC coverage')
    exact_keys(record['kl'],('temperature','direction','teacher_entropy','student_entropy','both_distributions_connected','temperature_squared_factor'))
    k=record['kl']
    if k['temperature']!=.5 or k['direction']!='text||adapted_vision' or k['both_distributions_connected'] is not True or k['temperature_squared_factor'] is not False:
        raise InvalidRecord('KL semantics')
    if len(k['teacher_entropy'])!=batch or len(k['student_entropy'])!=batch:raise InvalidRecord('KL coverage')
    exact_keys(record['appearance'],('image_min','image_max','reconstruction_min','reconstruction_max'))
    p=record['appearance']
    if not 0<=p['image_min']<=p['image_max']<=1 or p['reconstruction_min']>p['reconstruction_max']:raise InvalidRecord('Appearance ranges')
    def finite(value):
        if isinstance(value,dict):
            for item in value.values():finite(item)
        elif isinstance(value,list):
            for item in value:finite(item)
        elif isinstance(value,(int,float)) and not math.isfinite(value):raise InvalidRecord('Nonfinite objective record')
    finite(record)
    return {'coverage':'VALID','basis':'Supplied metadata only; no live graph observation','execution':'NOT RUN'}
