import os
from typing import Dict, Tuple, Any

from .jobs import BenchmarkJob

def slurm_time_and_partition(
    time: int,
    buffer: float = 0.5
) -> Tuple[str, str]:
    """
    Params
    ======
    time: int
        The amount of time in minutes for the job

    buffer: float
        The percent of extra time to add on as a buffer

    Returns
    =======
    Dict[str, str]
        The partition to use and the allocated time as a str
    """
    allocated_time = int(time * (1 + buffer))
    d = int(allocated_time / (60*24))
    h = int((allocated_time - d * 24 * 60) / 60)
    m = int(allocated_time % 60)

    short_max = (2 * 60)
    medium_max = (1 * 24 * 60)
    defq_max = (4 * 24 * 60)

    # zfill(2) just pre-pads with 0's to a str length of 2
    time_str = f'{d}-{str(h).zfill(2)}:{str(m).zfill(2)}:00'

    if allocated_time < short_max:
        return (time_str, 'short')
    elif allocated_time < medium_max:
        return (time_str, 'medium')
    elif allocated_time < defq_max:
        return (time_str, 'defq')

    raise ValueError('Requires too much time,'
                     + f'{allocated_time}, possible to queue'
                     + 'on the smp partition if required')

def slurm_job_options(
    job: BenchmarkJob,
    force: bool = False,
    time_buffer: float = 0.25
) -> Dict[str, Any]:
    time, partition = slurm_time_and_partition(job.time, time_buffer)
    return {
        'slurm_args' : {
            'job-name': job.name(),
            'nodes': 1,
            'ntasks': 1,
            'cpus-per-task': job.cpus,
            'output': os.path.join(job.basedir, 'slurm_out'),
            'error': os.path.join(job.basedir, 'slurm_err'),
            'export': 'ALL',
            'mem': job.memory,
            'time': time,
            'partition' : partition,
        },
        'slurm_opts': [],
        'slurm_script_path': os.path.join(job.basedir, 'slurm_script.sh'),
        'force': force
    }
