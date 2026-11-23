"""Partial-credit GenEval training reward, computed by the official GenEval detector in a subprocess."""
import json
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENEVAL_PYTHON = os.path.join(REPO, 'env_geneval', 'bin', 'python')
DETECTOR_CONFIG = os.path.join(REPO, 'external', 'mmdetection', 'configs', 'mask2former',
                               'mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py')
DETECTOR_WEIGHTS = os.path.join(REPO, 'external', 'geneval', 'weights')


def normalize_prompt(prompt):
    return ' '.join(prompt.split()).strip()


def load_metadata_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_metadata(path):
    """GenEval metadata records keyed by prompt (the rows of a prompt carry the same record)."""
    return {normalize_prompt(r['prompt']): r for r in load_metadata_rows(path)}


class GenEvalReward:
    """reward(image, prompt) -> (partial credit, official verdict)."""

    def __init__(self, metadata_path, tmpdir, deterministic_seed=None):
        self.meta = {k.lower(): v for k, v in load_metadata(metadata_path).items()}
        self.tmpdir = tmpdir
        os.makedirs(tmpdir, exist_ok=True)
        self.stderr = open(os.path.join(tmpdir, f'worker_stderr_{os.getpid()}.log'), 'w')
        # The worker runs torch 1.12, which rejects the training environment's allocator option.
        env = {k: v for k, v in os.environ.items()
               if k not in ('VIRTUAL_ENV', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTORCH_CUDA_ALLOC_CONF')}
        env['PYTHONNOUSERSITE'] = '1'
        command = [GENEVAL_PYTHON, '-u', os.path.join(REPO, 'sprout', 'geneval_worker.py'),
                   '--model-config', DETECTOR_CONFIG, '--model-path', DETECTOR_WEIGHTS]
        if deterministic_seed is not None:
            command += ['--deterministic', '--seed', str(deterministic_seed)]
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                                     cwd=REPO, text=True, bufsize=1, env=env)
        hello = self.proc.stdout.readline()
        if not hello.strip() or not json.loads(hello).get('ready'):
            raise RuntimeError(f'GenEval worker failed to start; see {self.stderr.name}')
        self.count = 0

    def __call__(self, image, prompt):
        meta = self.meta.get(prompt.strip().lower())
        if meta is None:
            raise KeyError(f'no GenEval metadata for prompt {prompt!r}')
        self.count += 1
        path = os.path.join(self.tmpdir, f'{os.getpid()}_{self.count}.png')
        image.save(path)
        try:
            self.proc.stdin.write(json.dumps({'image': path, 'meta': meta}) + '\n')
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        finally:
            os.remove(path)
        if not line:
            raise RuntimeError(f'GenEval worker exited; see {self.stderr.name}')
        out = json.loads(line)
        if 'error' in out:
            raise RuntimeError(f"GenEval worker error: {out['error']}")
        return float(out['reward']), int(out['correct'])
