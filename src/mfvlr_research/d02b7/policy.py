from ..d02b1.schema import InvalidRecord
CELL='D02B-G01-A0'

def require_cell(cell):
    if type(cell) is not str or cell!=CELL:raise PermissionError('Only D02B-G01-A0 authorized')

def resource_gate(ram,disk):
    if any(type(x) is not int or x<0 for x in (ram,disk)):raise InvalidRecord('Actual resource readings required')
    return ram>=5*1024**3 and disk>=8*1024**3

def status(*,completed=False,numerical='NOT RUN',structural='NOT RUN',provenance='NOT RUN',resource='NOT VERIFIED',error=False):
    if numerical not in ('PASS','FAIL','NOT RUN') or structural not in ('PASS','FAIL','NOT RUN') or provenance not in ('PASS','FAIL','NOT RUN') or resource not in ('PASS','INSUFFICIENT','FAIL/OOM','NOT VERIFIED'):
        raise InvalidRecord('Unknown status')
    if type(completed) is not bool or type(error) is not bool:raise InvalidRecord('Boolean state required')
    if completed and (error or structural!='PASS' or provenance!='PASS' or resource!='PASS' or numerical=='NOT RUN'):
        raise InvalidRecord('Incomplete mandatory evidence cannot be COMPLETE')
    return dict(execution='COMPLETE' if completed else 'HARD STOP' if error else 'WAIT/NOT RUN' if resource=='INSUFFICIENT' else 'NOT RUN',
                numerical=numerical,structural=structural,provenance=provenance,resource=resource,
                dependent_work='BLOCKED',B0_D08='DISABLED')
