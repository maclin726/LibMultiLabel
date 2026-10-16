import re, glob, sys, datetime, statistics
ansi = re.compile(r'\x1b\[[0-9;]*m')
ts = re.compile(r'(\d\d-\w\w\w-\d\d \d\d:\d\d:\d\d) : (.*)')
for path in sys.argv[1:]:
    try: text = ansi.sub('', open(path, errors='ignore').read())
    except Exception: continue
    lines = [l for l in re.split(r'[\r\n]', text) if ' : ' in l]
    name = ds = None; gpus = None; host = text.split('\n',1)[0].strip()
    epochs = []; start = None; evals = []
    for l in lines:
        m = ts.search(l)
        if not m: continue
        t = datetime.datetime.strptime(m.group(1), '%d-%b-%y %H:%M:%S'); msg = m.group(2)
        if start is None: start = t
        mm = re.search(r'Experiment name: (\S+), Dataset: (\S+)', msg)
        if mm: name, ds = mm.groups()
        mg = re.search(r"'num_gpu': (\d+)", msg)
        if mg: gpus = int(mg.group(1))
        me = re.search(r'Mean loss after epoch (\d+)/(\d+)', msg)
        if me: epochs.append((t, int(me.group(1)), int(me.group(2))))
    if not epochs: continue
    diffs = [(b[0]-a[0]).total_seconds() for a, b in zip(epochs, epochs[1:])]
    first = (epochs[0][0]-start).total_seconds()
    med = statistics.median(diffs) if diffs else first
    print(f"{path.split('/')[-1]:32s} {host:7s} {str(ds):20s} gpus={gpus} epochs_logged={len(epochs)} last={epochs[-1][1]}/{epochs[-1][2]} first_epoch_s={first:.0f} median_epoch_s={med:.0f} total_h={(epochs[-1][0]-start).total_seconds()/3600:.2f} {name}")
