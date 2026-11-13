"""Fixed-input conformance against pinned author sources, without Isaac Gym."""
import ast
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
import pytest
from explorer3.fleet_allocation import FleetAllocator,effort_metrics

SOURCE=Path('/home/jetsonnx/.local/share/explorer3/cache/source/ifl_benchmark')


@pytest.mark.parametrize('method,module,klass',[('random','random_allocation','RandomAllocation'),
                                             ('fleet_thrifty','TD_allocation','TDAllocation')])
def test_author_function_and_busy_assignment_conformance(method,module,klass):
    if not SOURCE.exists():pytest.skip('Pinned author source cache unavailable')
    package=types.ModuleType('upstream_fleet');package.__path__=[str(SOURCE/'iflb/allocations')]
    sys.modules['upstream_fleet']=package
    spec=importlib.util.spec_from_file_location('upstream_fleet.'+module,SOURCE/f'iflb/allocations/{module}.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    tree=ast.parse((SOURCE/'iflb/parallel_experiment.py').read_text())
    method_ast=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='assign_humans')
    namespace={'np':np};exec(compile(ast.Module(body=[method_ast],type_ignores=[]),'upstream_assignment','exec'),namespace)
    actual=FleetAllocator(3,method,seed=123,min_steps=3,total_steps=12,action_budget=8)
    reference=types.SimpleNamespace(exp_cfg=actual.exp_cfg,assignments=np.zeros((3,1)),
        human_timers=[0],blocked_envs=[False]*3)
    allocation=getattr(original,klass)(actual.exp_cfg);rng=np.random.RandomState(123)
    saved=np.random.get_state()
    try:
        for t in range(20):
            risk=np.array([.1,.8,.4]) if t%2 else np.array([.9,.2,.3])
            novelty=np.array([.3,.7,.9]);blocked=[False,t in (2,3,4),False]
            metrics={'td_risk':risk,'uncertainty':novelty,'constraint_violation':blocked,
                'assignments':reference.assignments.copy(),'human_timers':reference.human_timers.copy(),
                'blocked_envs':blocked,'time':t}
            np.random.set_state(rng.get_state());priorities=allocation.allocate(metrics);rng.set_state(np.random.get_state())
            reference.blocked_envs=blocked;namespace['assign_humans'](reference,priorities)
            if reference.assignments.sum():reference.human_timers[0]+=1
            actual.allocate(risk,novelty,blocked)
            assert np.array_equal(actual.assignments,reference.assignments)
            assert actual.human_timers==reference.human_timers
            assert actual.assignments.sum()<=1
    finally:np.random.set_state(saved)


def test_rohe_paper_definition_and_separate_wall_cost():
    assignments=np.array([[[1],[0],[0]],[[0],[1],[0]]])
    result=effort_metrics(9,assignments,3,successes=4,correction_s=20,switch_s=20,reset_s=20)
    assert result['rohe']==1
    assert result['tasks_per_supervision_minute']==4
    assert not result['measured_human_cost']
